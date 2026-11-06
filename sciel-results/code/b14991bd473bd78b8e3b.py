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
    
    N_max = 100000
    
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

    def apply_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
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

    def revert_move(counts, residues, i, r_old, r_new, N):
        p = primes[i]
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

    def init_greedy():
        """Greedy: for each prime, pick residue that maximizes coverage of prefix."""
        residues = np.zeros(n_primes, dtype=np.int64)
        counts = np.zeros(N_max + 1, dtype=np.int32)
        # Start with all zeros, then greedily improve
        for i, p in enumerate(primes):
            # Try all residues, pick the one that gives best score
            best_r = 0
            best_sc = -1
            # Only try a subset of residues for speed
            candidates = [0, p // 2, (i * 7 + 3) % p, (i * 13 + 5) % p]
            for r in candidates:
                r = r % p
                if r == 0:
                    if p <= N_max:
                        counts[p::p] += 1
                else:
                    if r <= N_max:
                        counts[r::p] += 1
                sc = get_score(counts, N_max)
                if sc > best_sc:
                    best_sc = sc
                    best_r = r
                # Revert
                if r == 0:
                    if p <= N_max:
                        counts[p::p] -= 1
                else:
                    if r <= N_max:
                        counts[r::p] -= 1
            # Apply best
            residues[i] = best_r
            if best_r == 0:
                if p <= N_max:
                    counts[p::p] += 1
            else:
                if best_r <= N_max:
                    counts[best_r::p] += 1
        return residues

    def init_structured():
        """Structured initialization based on index."""
        return np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64)

    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)

    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)

    def init_one():
        return np.ones(n_primes, dtype=np.int64)

    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)

    initial_strategies = [
        init_greedy,
        init_structured,
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]

    for strat in initial_strategies:
        if time.time() > deadline - 15.0:
            break
            
        residues = strat()
        N_cur = 5000
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Dynamic Expansion
        stall_count = 0
        max_stall = 50
        
        while time.time() < deadline - 8.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                if N_cur >= N_max:
                    break
                N_cur = min(N_cur * 2, N_max)
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                continue
            
            # Find the first failure
            x_fail = current_score
            
            # Move Selection: pick a random prime
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            
            # Does p cover x_fail?
            covers_x = (x_fail % p == r_old)
            
            if covers_x:
                # This prime is contributing to x_fail.
                # Try to pick a non-covering prime
                if rng.random() < 0.7:
                    i2 = int(rng.integers(n_primes))
                    p2 = primes[i2]
                    r_old2 = int(residues[i2])
                    covers_x2 = (x_fail % p2 == r_old2)
                    if not covers_x2:
                        i = i2
                        p = p2
                        r_old = r_old2
                        covers_x = False
                    else:
                        # Perturb the covering prime
                        r_new = int(rng.integers(0, p))
                        if r_new == r_old:
                            continue
                        apply_move(counts, residues, i, r_old, r_new, N_cur)
                        new_score = get_score(counts, N_cur)
                        if new_score > current_score:
                            accepted = True
                        elif new_score == current_score and rng.random() < 0.2:
                            accepted = True
                        else:
                            accepted = False
                        if accepted:
                            current_score = new_score
                            stall_count = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_residues = residues.copy()
                        else:
                            revert_move(counts, residues, i, r_new, r_old, N_cur)
                            stall_count += 1
                            if stall_count > max_stall:
                                break
                        continue
                else:
                    # Perturb the covering prime
                    r_new = int(rng.integers(0, p))
                    if r_new == r_old:
                        continue
                    apply_move(counts, residues, i, r_old, r_new, N_cur)
                    new_score = get_score(counts, N_cur)
                    if new_score > current_score:
                        accepted = True
                    elif new_score == current_score and rng.random() < 0.2:
                        accepted = True
                    else:
                        accepted = False
                    if accepted:
                        current_score = new_score
                        stall_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                    else:
                        revert_move(counts, residues, i, r_new, r_old, N_cur)
                        stall_count += 1
                        if stall_count > max_stall:
                            break
                    continue
            else:
                # p does NOT cover x_fail.
                # Set r_new to cover x_fail.
                r_new = int(x_fail % p)
                
                apply_move(counts, residues, i, r_old, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                
                if new_score > current_score:
                    accepted = True
                elif new_score == current_score and rng.random() < 0.3:
                    accepted = True
                else:
                    accepted = False
                
                if accepted:
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_new, r_old, N_cur)
                    stall_count += 1
                    if stall_count > max_stall:
                        break

    # Final polish: if we have time, do a focused search on the best solution
    if best_residues is not None and time.time() < deadline - 3.0:
        residues = best_residues.copy()
        N_cur = N_max
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        polish_stall = 0
        max_polish_stall = 30
        while time.time() < deadline - 1.0 and polish_stall < max_polish_stall:
            if current_score == N_cur:
                break
            x_fail = current_score
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            covers_x = (x_fail % p == r_old)
            if not covers_x:
                r_new = int(x_fail % p)
                apply_move(counts, residues, i, r_old, r_new, N_cur)
                new_score = get_score(counts, N_cur)
                if new_score > current_score:
                    current_score = new_score
                    polish_stall = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    revert_move(counts, residues, i, r_new, r_old, N_cur)
                    polish_stall += 1
            else:
                polish_stall += 1

    # Final Verification
    if best_residues is not None and time.time() < deadline - 0.5:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        
    return best_residues.tolist()
