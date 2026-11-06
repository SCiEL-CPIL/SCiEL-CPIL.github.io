import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_check = 20000
    
    def evaluate_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]
    
    def build_counts(residues):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        return counts
    
    def apply_change(counts, idx, old_r, new_r):
        p = primes[idx]
        old_start = old_r if old_r > 0 else p
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = new_r if new_r > 0 else p
        if new_start <= max_check:
            counts[new_start::p] += 1
    
    best_residues = None
    best_score = 0
    
    # Strategy 1: Greedy forward construction
    # Initialize with all zeros
    residues = np.zeros(n_primes, dtype=np.int64)
    counts = build_counts(residues)
    current_score = evaluate_counts(counts)
    
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Greedy: extend prefix one step at a time
    max_greedy_iters = 10000
    for _ in range(max_greedy_iters):
        if time.time() > deadline - 10:
            break
        if current_score >= max_check:
            break
        
        x_target = current_score + 1
        count_x = counts[x_target]
        
        if count_x >= 2:
            current_score = evaluate_counts(counts)
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
            continue
        
        # Need to add (2 - count_x) coverage at x_target
        # Try changing residues of primes to cover x_target
        
        applied = False
        # Try primes in increasing order
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
            
            # Check if this change breaks any position < x_target with count == 2
            # Positions losing coverage: y where y % p == old_r and y < x_target
            # We need counts[y] > 2 for all such y
            
            start_pos = old_r if old_r > 0 else p
            if start_pos >= x_target:
                # No positions in [1, x_target-1] covered by old_r
                valid = True
            else:
                positions = np.arange(start_pos, x_target, p)
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                apply_change(counts, i, old_r, new_r)
                residues[i] = new_r
                applied = True
                break
        
        if not applied:
            # Try a different approach: look at positions beyond x_target
            # or allow temporary "damage" if it helps overall
            # For now, break
            break
        
        current_score = evaluate_counts(counts)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
    
    # Strategy 2: Local search / hill climbing on the count vector
    # Objective: maximize the minimum count in [1, L] for some L, i.e., extend the doubly-covered prefix
    # Use a scoring function that rewards high L and also high minimum count near L
    
    def local_search(start_residues, max_iters=5000):
        residues_ls = start_residues.copy()
        counts_ls = build_counts(residues_ls)
        score_ls = evaluate_counts(counts_ls)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            if score_ls >= max_check:
                break
            
            # Pick a random prime to change
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues_ls[i]
            
            # Try a few random new residues
            tried = set()
            for _ in range(10):
                new_r = np.random.randint(p)
                if new_r == old_r or new_r in tried:
                    continue
                tried.add(new_r)
                
                # Apply change temporarily
                apply_change(counts_ls, i, old_r, new_r)
                
                new_score = evaluate_counts(counts_ls)
                
                # Accept if better or equal (to allow plateaus)
                if new_score >= score_ls:
                    residues_ls[i] = new_r
                    score_ls = new_score
                    break
                else:
                    # Revert
                    apply_change(counts_ls, i, new_r, old_r)
        
        return score_ls, residues_ls
    
    # Run local search from the greedy solution
    if time.time() < deadline - 15:
        ls_score, ls_residues = local_search(best_residues, max_iters=20000)
        if ls_score > best_score:
            best_score = ls_score
            best_residues = ls_residues
    
    # Strategy 3: Multi-start with different initializations
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if r % 4 == 0:
            residues_rs = np.array([np.random.randint(p) for p in primes], dtype=np.int64)
        elif r % 4 == 1:
            # Random but biased to small values
            residues_rs = np.array([np.random.randint(max(1, p // 4)) for p in primes], dtype=np.int64)
        elif r % 4 == 2:
            # Start from best and perturb
            residues_rs = best_residues.copy()
            for i in range(n_primes // 3):
                idx = np.random.randint(n_primes)
                residues_rs[idx] = np.random.randint(primes[idx])
        else:
            # Start from greedy-like: assign residues to maximize early coverage
            residues_rs = np.zeros(n_primes, dtype=np.int64)
            counts_rs = np.zeros(max_check + 1, dtype=np.int32)
            for i in range(n_primes):
                p = primes[i]
                # Pick residue that maximizes coverage of the first 100 positions
                best_r = 0
                best_cover = -1
                for trial_r in range(min(p, 20)):
                    start = trial_r if trial_r > 0 else p
                    if start > 100:
                        continue
                    cover = np.sum(1 for k in range(start, min(start + p * 10, 101), p) if counts_rs[k] < 2)
                    if cover > best_cover:
                        best_cover = cover
                        best_r = trial_r
                residues_rs[i] = best_r
                start = best_r if best_r > 0 else p
                if start <= max_check:
                    counts_rs[start::p] += 1
        
        rs_score, rs_residues = local_search(residues_rs, max_iters=5000)
        if rs_score > best_score:
            best_score = rs_score
            best_residues = rs_residues
    
    # Final polish: run local search on the best solution found
    if time.time() < deadline - 5:
        final_score, final_residues = local_search(best_residues, max_iters=10000)
        if final_score > best_score:
            best_score = final_score
            best_residues = final_residues
    
    return [int(r) for r in best_residues]
