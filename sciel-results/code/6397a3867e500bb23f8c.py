import time
import math
import itertools
import random

def _set_sum_diff(U):
    Us = list(U)
    plus = set()
    minus = set()
    for a in Us:
        for b in Us:
            plus.add(a + b)
            minus.add(a - b)
    return len(plus), len(minus)

def _score(U):
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_val = -1e9
    best_set = [0, 1, 3]
    
    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # 1. Baseline
    consider([0, 1, 3])
    
    # 2. Exhaustive search over small diameters
    # We search for subsets of {0, 1, ..., M} containing 0.
    # Since the score depends on max(U), we can iterate over possible max values.
    # For a fixed max M, we want to find the subset U of {0..M} containing 0 and M (usually)
    # that maximizes the score.
    # However, exhaustive subset search is 2^M. We can limit M.
    # Based on the paper and typical small set behaviors, M up to 15-20 is feasible.
    
    # Let's iterate M from 2 to 20.
    # For each M, we consider subsets of {1, ..., M-1} to add to {0, M}.
    # Actually, max(U) doesn't have to be M if we don't include M.
    # But if we fix the "universe" to be {0, ..., M}, any subset U has max(U) <= M.
    # It is better to iterate by size or by specific structure.
    
    # Strategy A: Exhaustive over all subsets of {1, ..., K} for small K.
    # This covers all sets with max <= K.
    # K=15 -> 2^15 = 32768 subsets. Each subset evaluation is O(N^2) where N<=16.
    # 32768 * 256 ops ~ 8M ops. Very fast.
    # K=20 -> 2^20 = 1M subsets. 1M * 400 ops ~ 400M ops. Might be too slow in Python if not optimized.
    # Let's try K=18 first.
    
    max_K = 18
    for K in range(2, max_K + 1):
        if time.time() > deadline:
            break
        # Iterate over all subsets of {1, ..., K-1}
        # We always include 0.
        # We can also choose whether to include K.
        # Actually, let's just iterate all subsets of {1, ..., K}.
        # If K is not in the subset, max is smaller, which was covered in previous iterations?
        # No, previous iterations covered subsets of {1, ..., K-1}.
        # So if we iterate subsets of {1, ..., K}, we cover all sets with max <= K.
        # To avoid redundancy, we can just do it.
        
        # Optimization: Only consider subsets that include K?
        # If max(U) < K, it was considered in the previous loop iteration.
        # So we only need to consider subsets of {1, ..., K-1} that DO include K?
        # No, if we include K, max is K.
        # So: For each K, consider all subsets S of {1, ..., K-1}.
        # U = {0} U S U {K}.
        # This covers all sets with max exactly K.
        
        # Number of subsets of {1, ..., K-1} is 2^(K-1).
        # For K=18, 2^17 = 131k. Fast.
        # For K=20, 2^19 = 524k. Still reasonable.
        
        # Let's push K up to 22 if time permits.
        
        # We'll generate subsets using bitmasks or itertools.
        # itertools is slower than bitmasks for large counts?
        # Bitmask is faster.
        
        # We are iterating K from 2 to max_K.
        # Inside, we iterate masks from 0 to 2^(K-1) - 1.
        
        # Wait, the loop above iterates K.
        # Let's restructure to be efficient.
        
        # Actually, let's just do the bitmask loop for each K.
        
        # To avoid re-evaluating sets with max < K, we ensure K is in the set.
        # So we iterate masks for {1, ..., K-1}.
        
        # Optimization: Precompute sum/diff for small sets?
        # No, just compute directly.
        
        # Let's run K up to 20.
        
        # We already started the loop above, but let's refine the inner logic.
        # The current code structure has the loop outside.
        
        # Let's restart the logic inside the loop for clarity.
        
        # For each mask in 0..2^(K-1)-1:
        #   S = {i in 1..K-1 if mask has bit i-1}
        #   U = {0} U S U {K}
        #   score(U)
        
        # This is correct.
        
        # Let's add a check: if K is small, this is fast.
        
        # We'll handle the time check inside the inner loop too, or just let it run if fast enough.
        # 2^19 is 500k. 500k * (20^2) = 200M operations. In Python, this might take ~20-30 seconds.
        # So K=20 might be tight. K=18 is safe (~3M ops, <1s).
        
        # Let's try to go up to K=20 but break if time runs out.
        
        # We'll use a list comprehension to build U quickly.
        
        # Note: The outer loop in the code snippet above was `for K in range...`.
        # We need to put the bitmask logic inside.
        
        # Since I can't edit the previous block, I will implement the full logic here.
        
        # Re-evaluating the plan:
        # The code above is just the setup. I will write the full construct function now.
        
        pass # Placeholder to ensure syntax is valid in my head, will rewrite below

    # --- Revised Construct Logic ---
    
    # Reset for the actual implementation
    best_val = -1e9
    best_set = [0, 1, 3]
    consider([0, 1, 3])
    
    # Stage 1: Exhaustive search for max(U) <= 20
    # We iterate K (the max element) from 2 to 20.
    # For each K, we iterate all subsets of {1, ..., K-1}.
    # This ensures max(U) == K.
    
    # Optimization: Use numpy? No, standard python is fine for 2^19.
    # Let's try up to K=20.
    
    for K in range(2, 21):
        if time.time() > deadline - 5.0:
            break
            
        # Iterate subsets of {1, ..., K-1}
        # Number of elements is K-1.
        # Masks from 0 to 2^(K-1) - 1.
        
        # We can speed this up by precomputing the list of bits?
        # Or just simple bit operations.
        
        # For very small K, this is trivial.
        
        # Let's use a generator for subsets?
        # itertools.combinations is good but we need all sizes.
        # Bitmask is better.
        
        # To make it faster, we can skip masks that result in very small sets?
        # No, small sets might have high ratios.
        
        # We'll just loop.
        
        # Optimization: 
        # If K is large, the number of subsets is huge.
        # 2^19 = 524,288.
        # For each, we build a list of size ~K and compute sum/diff O(K^2).
        # K=20, K^2=400. 524k * 400 = 200M ops.
        # Python can do ~10-50M simple ops/sec. So 200M might take 4-20 seconds.
        # This is acceptable within 145s.
        
        # We can parallelize? No.
        
        # Let's try to optimize the inner loop.
        
        # Precompute the values 1..K-1.
        # We can represent the set as a bitmask itself?
        # Sum/Diff of a set represented by bitmask?
        # Hard to do efficiently without converting to list.
        
        # Let's just do it.
        
        # One more optimization: 
        # Symmetry? No, sets are not symmetric in sum/diff.
        
        # We'll run the loop.
        
        # To avoid timeout, we check time every 10000 iterations.
        
        # Actually, let's just run it.
        
        # Wait, I need to be careful. 20 seconds is a lot.
        # If K=20 takes 20s, K=19 takes 10s, etc.
        # Total time ~ 40s. Safe.
        
        # Let's refine the inner loop for speed.
        
        # We'll build the set U.
        # U = [0, K] + [i for i in range(1, K) if (mask >> (i-1)) & 1]
        
        # Computing sum/diff:
        # Use sets.
        
        # Let's try.
        
        for mask in range(1 << (K - 1)):
            if time.time() > deadline - 2.0:
                break
                
            # Build U
            # Optimization: list comprehension
            # We can precompute the list of candidates [1, ..., K-1]
            # But the mask changes.
            
            # Let's just do it.
            
            # To speed up, we can use a tuple for U? No, set() is needed.
            
            # Constructing the set:
            # U_list = [0, K]
            # for i in range(1, K):
            #    if mask & (1 << (i-1)):
            #       U_list.append(i)
            
            # This is O(K).
            
            # Then _score is O(K^2).
            
            # Let's inline the score calculation for speed?
            # _score has overhead of function calls.
            
            # Inline version:
            U_list = [0, K]
            for i in range(1, K):
                if mask & (1 << (i - 1)):
                    U_list.append(i)
            
            # Convert to set for unique elements (already unique)
            # But we need to compute sum/diff.
            
            # Fast sum/diff:
            # Using sets is O(N^2) insertions.
            # For N=20, 400 insertions.
            
            plus = set()
            minus = set()
            # Local variables for speed
            Ul = U_list
            for a in Ul:
                for b in Ul:
                    plus.add(a + b)
                    minus.add(a - b)
            
            nplus = len(plus)
            nminus = len(minus)
            
            if nplus == 0:
                continue
                
            log_den = math.log(2 * K + 1) # max is K
            ratio = nminus / nplus
            if ratio <= 0:
                continue
            
            val = 1.0 + math.log(ratio) / log_den
            
            if val > best_val:
                best_val = val
                best_set = sorted(U_list)
                
    # Stage 2: Local Search / Hill Climbing from the best set found
    # The exhaustive search finds the best set with max <= 20.
    # Maybe a set with max > 20 is better?
    # Or maybe a set with max <= 20 but we missed it? No, exhaustive.
    
    # Can we improve by adding elements > 20?
    # If we add a large element M, max becomes M.
    # log(2M+1) increases.
    # Does |U-U|/|U+U| increase enough?
    
    # Let's try to extend the best set by adding larger elements.
    
    if time.time() < deadline - 10.0:
        cur = list(best_set)
        mx = max(cur)
        
        # Try adding elements in a range around mx
        # e.g., mx+1 to mx*2
        
        improved = True
        it = 0
        while improved and time.time() < deadline - 5.0 and it < 100:
            improved = False
            it += 1
            
            # Candidate additions
            # Focus on elements that might increase the ratio.
            # Lacunary sequences are good.
            # Try adding mx * k + 1?
            
            candidates = []
            # Small additions
            for d in range(1, 10):
                candidates.append(mx + d)
            # Large additions (lacunary-like)
            for k in range(2, 10):
                candidates.append(mx * k)
                candidates.append(mx * k + 1)
            
            # Random additions
            rng = random.Random(42)
            for _ in range(20):
                candidates.append(rng.randint(mx + 1, mx * 2))
            
            # Remove duplicates
            candidates = list(set(candidates))
            
            for c in candidates:
                U = cur + [c]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break
            
            # Also try removing elements?
            # Removing elements reduces max, which reduces log_den.
            # Usually adding is better for the ratio if the new element is "independent".
            # But sometimes a bad element hurts.
            # We already found the best set with max<=20.
            # If we added an element, max increased.
            # Removing it returns to previous state.
            # So removal is unlikely to improve if we just added.
            # But maybe we added two elements in previous iterations?
            # The loop breaks on first improvement.
            
    return best_set

# EVOLVE-BLOCK-END
