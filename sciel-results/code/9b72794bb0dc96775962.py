import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1
    target_range = range(target_lo, target_hi + 1)
    
    # Helper to check if a set A covers all targets and compute peak
    def evaluate(A):
        A_set = set(A)
        counts = np.zeros(target_hi + 1, dtype=np.int32)
        # Use numpy for faster convolution-like updates? 
        # Actually, for k~40, O(k^2) is 1600 ops. 
        # 129 targets * 1600 ops is fast.
        # But we do this many times.
        
        # Let's optimize the evaluation.
        # If we represent A as a boolean array of size 129, 
        # the sumset is the convolution of A with A.
        # But we need unordered pairs (i <= j).
        # Convolution counts (i, j) and (j, i) separately for i != j.
        # So Conv[s] = 2 * (unordered pairs with i != j) + (pairs with i == j)
        # Let U[s] be unordered count.
        # Let D[s] be diagonal count (1 if s/2 in A and s even, else 0).
        # Conv[s] = 2 * (U[s] - D[s]) + D[s] = 2*U[s] - D[s]
        # So U[s] = (Conv[s] + D[s]) / 2
        
        # This is much faster using FFT or direct convolution if A is dense?
        # With k=40, A is sparse. Direct convolution O(N^2) where N=129 is ~16k ops.
        # Python loop for 129*129 is 16k iterations. That's fine.
        
        # Let's stick to the set-based O(k^2) but optimized.
        # Or better, precompute contributions.
        
        # For now, simple O(k^2)
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        
        uncovered = np.count_nonzero(counts[target_lo:target_hi+1] == 0)
        peak = np.max(counts[target_lo:target_hi+1])
        return uncovered, peak

    # Strategy:
    # 1. Binary search on k (size of A) to find minimum k that allows full coverage.
    # 2. For that k (and slightly larger for robustness), use local search to minimize peak.
    # 3. The local search will use a "swap" move but guided by load balancing.
    
    best_A = None
    best_score = (float('inf'), float('inf'))
    
    # First, find a valid A with minimal k.
    # Lower bound on k:
    # Each element in A can form at most k sums.
    # Total sums needed: 129.
    # Total sums provided: k(k+1)/2.
    # k(k+1)/2 >= 129 => k^2 ~ 258 => k ~ 16.
    # But coverage is the constraint.
    # Let's try k from 16 upwards.
    
    min_k_found = None
    for k in range(16, 60):
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid A of size k
        found = False
        attempts = 0
        max_attempts = 50 if k < 25 else 20
        
        while attempts < max_attempts and not found:
            attempts += 1
            A = random.sample(range(n + 1), k)
            unc, peak = evaluate(A)
            if unc == 0:
                found = True
                min_k_found = k
                # We found a valid A. Now we need to minimize peak.
                # We can run local search on this A.
                break
        
        if found:
            break
    
    if min_k_found is None:
        # Fallback if no small k found quickly
        min_k_found = 40
        A = random.sample(range(n + 1), min_k_found)
    
    # Now, run local search to minimize peak, starting from a valid A of size min_k_found (or slightly larger)
    # We will try sizes from min_k_found to min_k_found + 10
    # Larger k allows lower peak because sums are more spread out.
    
    for k in range(min_k_found, min(min_k_found + 15, 60)):
        if time.time() > deadline - 5:
            break
            
        # Generate a good starting point
        # We can use the previous best A and add random elements, or restart.
        # Let's restart with random sampling but bias towards "well-distributed" sets.
        # Or just random.
        
        best_local_A = None
        best_local_score = (float('inf'), float('inf'))
        
        num_restarts = 10
        if k > 30:
            num_restarts = 5
            
        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initial A
            A = random.sample(range(n + 1), k)
            # Ensure it's valid? If not, try to fix it or skip?
            # Local search can handle invalid sets by penalizing uncovered.
            # Score: (uncovered, peak)
            
            A_set = set(A)
            curr_unc, curr_peak = evaluate(A)
            curr_score = (curr_unc, curr_peak)
            
            best_A_local = A[:]
            best_score_local = curr_score
            
            # Local Search
            # Move: Swap x in A with y not in A.
            # We want to reduce uncovered first, then peak.
            
            steps = 2000
            if k < 25:
                steps = 5000
                
            for step in range(steps):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random x from A
                x = random.choice(A)
                
                # Try to find a better y
                # Heuristic: y should help cover uncovered sums or reduce peak.
                # Let's just try a few random y's.
                
                improved = False
                for _ in range(5):
                    y = random.randint(0, n)
                    if y in A_set:
                        continue
                    
                    # Simulate swap
                    A.remove(x)
                    A_set.remove(x)
                    A.append(y)
                    A_set.add(y)
                    
                    new_unc, new_peak = evaluate(A)
                    new_score = (new_unc, new_peak)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score_local:
                            best_score_local = curr_score
                            best_A_local = A[:]
                        break
                    else:
                        # Rollback
                        A.remove(y)
                        A_set.remove(y)
                        A.append(x)
                        A_set.add(x)
                
                if not improved:
                    # If stuck, maybe try a "kick" - random change even if worse?
                    # Or just continue.
                    pass
            
        if best_local_score < best_score:
            best_score = best_local_score
            best_A = best_local_A
            
        # If we achieved peak 1 or 2, we can stop early?
        # Peak 1 means all sums are unique (Sidon set property for the range).
        # Peak 2 is very good.
        if best_score[0] == 0 and best_score[1] <= 3:
            break

    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return best_A

