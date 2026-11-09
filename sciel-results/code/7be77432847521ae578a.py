import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are n*(n+1)/2 = 210 pairs
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    def is_valid_set(A):
        if len(A) != n:
            return False
        if len(set(A)) != n:
            return False
        if min(A) < 1 or max(A) > max_val:
            return False
        return True

    # Helper to generate a candidate set from parameters
    def get_smooth_pool():
        pool = set()
        for a in range(0, 14):
            v2 = 1 << a
            if v2 > max_val: break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > max_val: break
                for c in range(0, 6):
                    v5 = v3 * (5 ** c)
                    if v5 > max_val: break
                    for d in range(0, 5):
                        v7 = v5 * (7 ** d)
                        if v7 > max_val: break
                        pool.add(v7)
        return sorted(pool)
    
    smooth_pool = get_smooth_pool()
    
    # Generate initial candidates from structured families
    candidates = []
    
    # 1. Smallest smooth numbers
    if len(smooth_pool) >= n:
        candidates.append(smooth_pool[:n])
        
    # 2. Geometric progressions with integer steps
    for r in [1.1, 1.2, 1.5, 2.0, 3.0]:
        for start in [1, 2, 5, 10]:
            seq = []
            curr = float(start)
            for _ in range(n):
                val = int(round(curr))
                if val < 1: val = 1
                if val > max_val: break
                if val not in seq:
                    seq.append(val)
                curr *= r
            if len(seq) == n:
                candidates.append(seq)
                
    # 3. Arithmetic progressions
    for start in [1, 2, 5, 10, 50]:
        for step in [1, 2, 5, 10, 50]:
            seq = [start + i * step for i in range(n)]
            if max(seq) <= max_val:
                candidates.append(seq)
                
    # 4. Random candidates
    for _ in range(10):
        candidates.append(sorted(random.sample(range(1, max_val + 1), n)))
        
    # Evaluate initial candidates
    best_score = float('inf')
    best_set = None
    
    for cand in candidates:
        if not is_valid_set(cand):
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = random.sample(range(1, max_val + 1), n)
        best_score = evaluate(best_set)
        
    current_set = best_set[:]
    current_score = best_score
    
    # Incremental Delta Greedy Local Search
    # We precompute the "neighborhood" of candidates for each position.
    # For each position i, we consider a set of candidate values to replace current_set[i] with.
    # To be exhaustive-ish but fast, we sample a large number of candidates per position.
    # However, the directive suggests a structural change. Exhaustive single-element replacement
    # over the entire [1, 10000] range is 10000 * 20 = 200,000 evaluations per full sweep.
    # Each evaluation takes ~50-100 microseconds with numpy? No, unique is expensive.
    # Let's estimate: 200k evals * 100us = 20s per sweep.
    # We have 85s. So we can do ~3-4 full sweeps.
    # This is much more deterministic and thorough than SA.
    
    # To speed up evaluation, we can use the fact that only 2n-1 sums and 2n-1 products change.
    # But implementing incremental unique count is tricky.
    # Let's stick to full evaluation but optimize the candidate set.
    # Instead of random, we can generate a "grid" of candidates.
    # Or simply iterate through a pre-computed list of "interesting" numbers.
    
    # Let's define a candidate pool for replacement.
    # We want to try replacing each element with:
    # 1. Other elements in the current set (swap logic, but we are replacing, so distinctness check handles it)
    # 2. Smooth numbers
    # 3. Numbers close to the current value
    # 4. Random numbers
    
    # Actually, a pure exhaustive search over [1, 10000] is feasible if we are careful.
    # Let's try it.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            old_val = current_set[i]
            best_new_val = None
            best_new_score = current_score
            
            # We will iterate through a subset of [1, 10000] to save time?
            # No, let's try full iteration but break early if time is tight.
            # 10000 iterations * 20 positions = 200,000 evals.
            
            # Optimization: Skip values already in the set (except the one we're replacing)
            # Create a set of current values for O(1) lookup
            current_vals = set(current_set)
            
            # We can't use a set for the loop itself, but we can skip.
            
            # To make this faster, let's only check a fraction of the range if time is running out.
            # But let's start with full range.
            
            # Precompute sums and products for the current set to update them?
            # No, let's just call evaluate. It's vectorized.
            
            # We can optimize by not calling evaluate if the new value is the same.
            
            # Let's try a smaller candidate set first to see if it converges faster.
            # Candidate set: Smooth numbers + neighbors + random sample.
            # If we do full [1, 10000], it's 10000 * 20 = 200k evals.
            # If evaluate takes 100us, that's 20s.
            # If we do 2 sweeps, that's 40s.
            # This is a good use of time.
            
            for new_val in range(1, max_val + 1):
                if new_val == old_val:
                    continue
                if new_val in current_vals:
                    continue
                
                # Apply change
                current_set[i] = new_val
                
                # Evaluate
                new_score = evaluate(current_set)
                
                # Revert
                current_set[i] = old_val
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
            
            if best_new_val is not None:
                current_set[i] = best_new_val
                current_score = best_new_score
                improved = True
                
                # Update current_vals if we were tracking it (we aren't in the loop, we recreate it)
                
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    
                # Check time
                if time.time() > deadline - 5:
                    break
    
    return sorted(best_set)
