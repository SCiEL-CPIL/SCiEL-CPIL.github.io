import numpy as np
import time
import random

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    N_max = 100000
    N_current = 3000
    N_expansion_factor = 2
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts
    
    def get_score(counts, N):
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1
    
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
    
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
    
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
    
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]
    
    def apply_move(counts, residues, i, r_new, N):
        p = primes[i]
        r_old = int(residues[i])
        if r_old == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_old <= N:
                counts[r_old::p] -= 1
        if r_new == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_new <= N:
                counts[r_new::p] += 1
        residues[i] = r_new
    
    def revert_move(counts, residues, i, r_old, N):
        p = primes[i]
        r_new = int(residues[i])
        if r_new == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_new <= N:
                counts[r_new::p] -= 1
        if r_old == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_old <= N:
                counts[r_old::p] += 1
        residues[i] = r_old
    
    def is_safe_move(counts, residues, i, N, L):
        """Check if changing residue i from old to new_r = x_fail % p is safe.
        Safe means: no position y < L with counts[y] == 2 loses its last non-target coverage."""
        p = primes[i]
        r_old = int(residues[i])
        x_fail = L + 1
        r_new = x_fail % p
        
        if r_old == r_new:
            return False
        
        # Check positions covered by old residue that are < L
        # If any such position has count == 2, and we remove coverage, it becomes 1 -> bad
        if r_old == 0:
            start = p
        else:
            start = r_old
        
        if start > L:
            return True
        
        # Iterate through positions covered by old residue, up to L
        y = start
        while y <= L:
            if counts[y] == 2:
                return False
            y += p
        return True
    
    def find_safe_candidates(counts, residues, L, N):
        """Find all primes i where changing to cover L+1 is safe."""
        x_fail = L + 1
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            r_old = int(residues[i])
            r_new = x_fail % p
            if r_old == r_new:
                continue
            if is_safe_move(counts, residues, i, N, L):
                candidates.append(i)
        return candidates
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        N_cur = N_current
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        stall_count = 0
        max_stall = 200
        consecutive_kicks = 0
        max_kicks = 3
        
        while time.time() < deadline - 5.0:
            if current_score == N_cur:
                if N_cur * N_expansion_factor > N_max:
                    break
                N_cur = min(N_cur * N_expansion_factor, N_max)
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                stall_count = 0
                consecutive_kicks = 0
                continue
            
            L = current_score
            
            # Try safe moves first
            candidates = find_safe_candidates(counts, residues, L, N_cur)
            
            if candidates:
                consecutive_kicks = 0
                # Pick a random safe candidate
                i = candidates[int(rng.integers(len(candidates)))]
                p = primes[i]
                r_old = int(residues[i])
                r_new = (L + 1) % p
                
                apply_move(counts, residues, i, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                
                if new_score > current_score:
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_old, N_cur)
                    stall_count += 1
                    if stall_count > max_stall:
                        # Try a kick
                        consecutive_kicks += 1
                        if consecutive_kicks > max_kicks:
                            break
                        # Random kick
                        i_kick = int(rng.integers(n_primes))
                        p_kick = primes[i_kick]
                        r_old_kick = int(residues[i_kick])
                        r_new_kick = int(rng.integers(0, p_kick))
                        apply_move(counts, residues, i_kick, r_new_kick, N_cur)
                        current_score = get_score(counts, N_cur)
                        stall_count = 0
            else:
                # No safe candidates - need a kick to escape
                consecutive_kicks += 1
                if consecutive_kicks > max_kicks:
                    break
                
                # Pick a random prime and change its residue
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = int(residues[i])
                r_new = int(rng.integers(0, p))
                if r_new == r_old:
                    r_new = (r_new + 1) % p
                
                apply_move(counts, residues, i, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                
                if new_score >= current_score:
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    # Accept with some probability to escape local minima
                    if rng.random() < 0.2:
                        current_score = new_score
                        stall_count = 0
                    else:
                        revert_move(counts, residues, i, r_old, N_cur)
                        stall_count += 1
                        if stall_count > max_stall:
                            break

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
    
    if best_residues is None:
        return [0] * 168
        
    return best_residues.tolist()
