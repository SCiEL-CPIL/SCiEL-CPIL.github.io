import numpy as np
import time
import random

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
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
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Dynamic window strategy
    N_current = 5000
    N_max = 100000
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

    # Greedy frontier: try all residue changes for non-covering primes, pick best
    def greedy_frontier_step(counts, residues, x_fail, N, rng):
        """Try all possible residue changes for primes that don't cover x_fail.
        Return (best_score, best_i, best_r_new) or None if no improvement."""
        best_new_score = -1
        best_i = -1
        best_r_new = -1
        
        for i in range(n_primes):
            p = primes[i]
            r_old = int(residues[i])
            if x_fail % p == r_old:
                continue  # This prime already covers x_fail
            
            # Try setting residue to x_fail % p
            r_new = x_fail % p
            if r_new == r_old:
                continue
            
            # Apply move
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
            
            new_score = get_score(counts, N)
            
            # Revert
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
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_i = i
                best_r_new = r_new
        
        if best_i >= 0 and best_new_score > 0:
            return (best_new_score, best_i, best_r_new)
        return None

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
        greedy_stall = 0
        max_greedy_stall = 5
        
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
                continue
            
            x_fail = current_score
            
            # Find non-covering primes
            non_covering_indices = []
            for i in range(n_primes):
                p = primes[i]
                r = int(residues[i])
                if x_fail % p != r:
                    non_covering_indices.append(i)
            
            if not non_covering_indices:
                i = int(rng.integers(n_primes))
                r_new = int(rng.integers(0, primes[i]))
                r_old = int(residues[i])
                if r_new == r_old:
                    continue
                apply_move(counts, residues, i, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                if new_score > current_score:
                    current_score = new_score
                    stall_count = 0
                    greedy_stall = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_old, N_cur)
                    stall_count += 1
                    if stall_count > max_stall:
                        break
                continue
            
            # Use greedy frontier when stalled
            if stall_count > 30 or greedy_stall < max_greedy_stall:
                result = greedy_frontier_step(counts, residues, x_fail, N_cur, rng)
                if result is not None:
                    new_score, i, r_new = result
                    r_old = int(residues[i])
                    apply_move(counts, residues, i, r_new, N_cur)
                    if new_score > current_score:
                        current_score = new_score
                        stall_count = 0
                        greedy_stall += 1
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                    else:
                        revert_move(counts, residues, i, r_old, N_cur)
                        greedy_stall += 1
                        stall_count += 1
                        if stall_count > max_stall:
                            break
                    continue
            
            # Random move
            i = non_covering_indices[int(rng.integers(len(non_covering_indices)))]
            p = primes[i]
            r_new = int(x_fail % p)
            r_old = int(residues[i])
            
            apply_move(counts, residues, i, r_new, N_cur)
            new_score = get_score(counts, N_cur)
            
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                if rng.random() < 0.3:
                    accepted = True
                else:
                    accepted = False
            else:
                accepted = False
            
            if accepted:
                current_score = new_score
                stall_count = 0
                greedy_stall = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                revert_move(counts, residues, i, r_old, N_cur)
                stall_count += 1
                if stall_count > max_stall:
                    break

    # Final polish: use remaining time to try to extend the best solution
    if best_residues is not None and time.time() < deadline - 2.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        
        # Try to extend further with a focused search
        if final_score < N_verify and time.time() < deadline - 1.0:
            residues = best_residues.copy()
            N_cur = min(final_score + 1000, N_verify)
            counts = build_counts(residues, N_cur)
            current_score = get_score(counts, N_cur)
            
            polish_stall = 0
            max_polish_stall = 50
            
            while time.time() < deadline - 0.5 and polish_stall < max_polish_stall:
                if current_score == N_cur:
                    break
                
                x_fail = current_score
                non_covering = [i for i in range(n_primes) if x_fail % primes[i] != int(residues[i])]
                
                if not non_covering:
                    break
                
                # Try greedy
                result = greedy_frontier_step(counts, residues, x_fail, N_cur, rng)
                if result is not None:
                    new_score, i, r_new = result
                    r_old = int(residues[i])
                    apply_move(counts, residues, i, r_new, N_cur)
                    if new_score > current_score:
                        current_score = new_score
                        polish_stall = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                    else:
                        revert_move(counts, residues, i, r_old, N_cur)
                        polish_stall += 1
                else:
                    # Random
                    i = non_covering[int(rng.integers(len(non_covering)))]
                    p = primes[i]
                    r_new = int(x_fail % p)
                    r_old = int(residues[i])
                    apply_move(counts, residues, i, r_new, N_cur)
                    new_score = get_score(counts, N_cur)
                    if new_score > current_score:
                        current_score = new_score
                        polish_stall = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                    else:
                        revert_move(counts, residues, i, r_old, N_cur)
                        polish_stall += 1
    
    return best_residues.tolist()
