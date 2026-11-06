import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _first_start(r, p):
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    start_time = time.time()
    deadline = start_time + 140
    
    def _compute_counts(residues, max_check):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = _first_start(r, p)
            if start <= max_check:
                counts[start::p] += 1
        return counts

    def _find_L(counts):
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return int(below[0])

    def _update_counts(counts, old_r, new_r, p, max_check):
        old_start = _first_start(old_r, p)
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = _first_start(new_r, p)
        if new_start <= max_check:
            counts[new_start::p] += 1

    def _greedy_init(max_check):
        residues = [0] * n
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            best_r = 0
            best_gain = -1
            # Try a subset of residues if p is large? No, p <= 1000, so p is small enough.
            # But iterating all p for each prime is O(N * AvgP * M) which is 168 * 500 * 10000 = 8.4e8. Too slow.
            # We need a faster heuristic.
            # Randomly sample 10 residues?
            candidates = np.random.choice(p, size=min(10, p), replace=False)
            for r in candidates:
                _update_counts(counts, 0, int(r), p, max_check)
                # Gain: number of positions that became double-covered or more
                # Actually, just count how many positions in [1, max_check] have count >= 2
                # This is expensive to compute for every candidate.
                # Proxy: count of positions with count == 1 that become count == 2?
                # Let's just pick the one that covers the most "currently uncovered" spots?
                # Or just pick random?
                _update_counts(counts, int(r), 0, p, max_check)
            # This greedy init is too slow if we compute exact gain.
            # Let's just use r=1 for all, or random.
            residues[i] = int(np.random.randint(0, p))
        return residues

    # Start with r_p = 1
    residues = [1] * n
    current_M = 2000
    counts = _compute_counts(residues, current_M)
    current_L = _find_L(counts)
    best_L = current_L
    best_residues = list(residues)
    
    # Local Search
    def _local_search(residues, max_check, time_limit):
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        current_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if current_L >= max_check:
                break
            
            # Find the first position with count < 2
            target_x = current_L + 1
            
            # Try to fix target_x
            # Find primes that do NOT cover target_x
            # If we change one of them to cover target_x, count[target_x] increases by 1
            # We need count[target_x] >= 2.
            # So if count[target_x] == 1, we need to add one more.
            # If count[target_x] == 0, we need to add two more.
            
            current_count_at_x = counts[target_x]
            
            # Find primes p where residues[i] != target_x % p
            # For each such prime, try changing residues[i] to target_x % p
            # and see if it improves L elsewhere or just fixes x.
            
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            # Sample a few primes to try to avoid O(N) scan if N is large? N=168 is small.
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = target_x % p
                
                if new_r == old_r:
                    continue
                
                _update_counts(counts, old_r, new_r, p, max_check)
                new_L = _find_L(counts)
                _update_counts(counts, new_r, old_r, p, max_check)
                
                delta = new_L - current_L
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_r = new_r
                    best_old_r = old_r
            
            if best_i != -1 and best_delta > 0:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                current_L += best_delta
                no_improve = 0
            else:
                no_improve += 1
                
                # If stuck, try a random move to escape local minimum
                if no_improve % 10 == 0:
                    i = int(np.random.randint(0, n))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r != old_r:
                        _update_counts(counts, old_r, new_r, p, max_check)
                        residues[i] = new_r
                        current_L = _find_L(counts)
        
        return residues, current_L

    # Main Loop
    while time.time() < deadline:
        if current_L < current_M:
            residues, current_L = _local_search(residues, current_M, deadline)
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            
            # If stuck, try to expand M slightly to see if we can make progress?
            # Or just restart with random perturbation?
            # Let's try to expand M if we are close
            if current_L > current_M * 0.9:
                current_M = min(current_M + 100, 10000)
                counts = _compute_counts(residues, current_M)
                current_L = _find_L(counts)
        else:
            # We covered the window, expand it
            current_M = min(current_M * 2, 10000)
            counts = _compute_counts(residues, current_M)
            current_L = _find_L(counts)
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            if current_M >= 10000:
                break

    return [int(r) for r in best_residues]
