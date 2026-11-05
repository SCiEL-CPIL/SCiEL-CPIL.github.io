"""Additive bases with few representations - new design using constructive + local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    target_low = 128
    target_high = 256
    target_size = target_high - target_low + 1  # 129 targets
    
    # Represent A as a boolean array of length 129 (indices 0..128)
    # For a given A, compute pair-sum multiplicities for targets [128, 256]
    
    def compute_multiplicities(A_bool):
        """Compute multiplicity of each target sum [128,256] as unordered pair sum from A.
        A_bool is a numpy bool array of length 129.
        Returns numpy int array of length 129 (for sums 128..256)."""
        # For unordered pairs (i,j) with i<=j, sum = i+j
        # We need count of pairs (i,j) in A with i<=j and i+j = s, for s in [128,256]
        # Use convolution-like approach but for unordered with equal allowed
        
        # Method: for each target s, count pairs
        # More efficient: use the fact that for sum s, pairs are (i, s-i) with i <= s-i, i.e., i <= s/2
        # And both i and s-i must be in A, and i <= s-i
        
        # Let's compute via iterating
        mult = np.zeros(target_size, dtype=np.int64)
        indices = np.where(A_bool)[0]
        for k in range(target_size):
            s = target_low + k
            # count pairs (i, j) with i <= j, i+j=s, i,j in A
            # i ranges from max(0, s-128) to min(128, s//2)
            i_min = max(0, s - 128)
            i_max = min(128, s // 2)
            if i_min > i_max:
                continue
            # For each i in [i_min, i_max], j = s - i, need i <= j (guaranteed by i <= s//2)
            # and both i and j in A
            # j = s - i, j >= i (since i <= s/2)
            # j must be <= 128, so s - i <= 128 => i >= s - 128 (already in i_min)
            # j >= 0 => i <= s (always true)
            # Count i in [i_min, i_max] such that A_bool[i] and A_bool[s-i]
            # Vectorized:
            i_vals = np.arange(i_min, i_max + 1)
            j_vals = s - i_vals
            count = np.sum(A_bool[i_vals] & A_bool[j_vals])
            mult[k] = count
        return mult
    
    def is_valid(A_bool):
        """Check that every target in [128,256] has at least one representation."""
        mult = compute_multiplicities(A_bool)
        return np.all(mult >= 1)
    
    def max_mult(A_bool):
        mult = compute_multiplicities(A_bool)
        return int(np.max(mult))
    
    # Initial construction: start with a dense set and refine
    # The seed uses [64,129) which gives sum range [128, 257]
    # For sum s in [128,256], pairs from [64,128]: i in [64,128], j=s-i in [64,128]
    # i >= 64, j >= 64 => i+j >= 128. i <= 128, j <= 128 => i+j <= 256.
    # For s in [128,256], i in [max(64,s-128), min(128,s-64)] and i <= s-i => i <= s/2
    # This gives a valid set. Let's compute its max mult.
    
    A_init = np.zeros(129, dtype=bool)
    A_init[64:129] = True  # indices 64..128
    
    # Verify validity
    if not is_valid(A_init):
        # Fallback: use full range
        A_init = np.ones(129, dtype=bool)
    
    best_A = A_init.copy()
    best_score = max_mult(best_A)
    
    # Local search: try to flip bits to reduce max multiplicity
    # Strategy: identify the target with max multiplicity, try to remove elements
    # that contribute to it, then repair coverage.
    
    def repair(A_bool):
        """Given a possibly invalid A, try to add elements to restore validity.
        Returns repaired A and success flag."""
        A = A_bool.copy()
        # Check which targets are uncovered
        mult = compute_multiplicities(A)
        uncovered = np.where(mult == 0)[0]
        if len(uncovered) == 0:
            return A, True
        
        # For each uncovered target, find a pair to add
        # Greedy: for each uncovered target s, find i,j not both in A, add one of them
        # Prefer adding an element that helps multiple uncovered targets
        remaining = list(uncovered)
        max_iters = 1000
        iters = 0
        while remaining and iters < max_iters:
            iters += 1
            # For each candidate element x in [0,128] not in A, count how many
            # uncovered targets it can help (i.e., there exists y in A or y=x such that x+y is uncovered)
            best_x = -1
            best_gain = 0
            for x in range(129):
                if A[x]:
                    continue
                gain = 0
                for s in remaining:
                    y = s - x
                    if 0 <= y <= 128 and (A[y] or y == x):
                        # Adding x would cover s (pair (x,y) or (y,x))
                        # But we need at least one pair; if y is in A or y==x (and we add x), then (min,max) works
                        # Actually if y==x, we need x in A (which we're adding), so (x,x) works
                        # If y != x, we need y in A (already) or y to be added later
                        # For now, count if y is already in A or y == x
                        if A[y] or y == x:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0 or best_gain == 0:
                # Can't repair with single additions, try adding any element
                # Just pick a random element
                candidates = [x for x in range(129) if not A[x]]
                if not candidates:
                    break
                best_x = random.choice(candidates)
            A[best_x] = True
            # Update uncovered
            mult = compute_multiplicities(A)
            remaining = [s for s in remaining if mult[s - target_low] == 0]
        
        if len(remaining) == 0:
            return A, True
        else:
            return A, False
    
    # Local search loop
    random.seed(42)
    while time.time() < deadline - 5.0:  # 5s safety margin
        A = best_A.copy()
        improved = False
        
        # Try removing elements that contribute to the max-multiplicity targets
        mult = compute_multiplicities(A)
        max_val = int(np.max(mult))
        max_targets = np.where(mult == max_val)[0]
        
        # For each element in A, compute how much it contributes to max targets
        # Try removing one element at a time
        elements_in_A = np.where(A)[0]
        random.shuffle(elements_in_A)
        
        for x in elements_in_A:
            if time.time() > deadline - 5.0:
                break
            # Try removing x
            A_test = A.copy()
            A_test[x] = False
            # Check if still valid
            mult_test = compute_multiplicities(A_test)
            if np.all(mult_test >= 1):
                new_max = int(np.max(mult_test))
                if new_max < best_score:
                    best_A = A_test.copy()
                    best_score = new_max
                    improved = True
                    A = A_test.copy()
                    mult = mult_test
                    max_val = new_max
                    max_targets = np.where(mult == max_val)[0]
                    elements_in_A = np.where(A)[0]
                    random.shuffle(elements_in_A)
        
        # Try adding elements if it helps reduce max multiplicity elsewhere
        # (less likely to help, but try)
        if not improved:
            # Try a random perturbation: remove one, add one
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                A_test = A.copy()
                # Remove a random element
                elems = np.where(A_test)[0]
                if len(elems) == 0:
                    break
                rem = random.choice(elems)
                A_test[rem] = False
                # Add a random element not in A
                not_in = np.where(~A_test)[0]
                if len(not_in) == 0:
                    break
                add = random.choice(not_in)
                A_test[add] = True
                # Check validity
                mult_test = compute_multiplicities(A_test)
                if np.all(mult_test >= 1):
                    new_max = int(np.max(mult_test))
                    if new_max < best_score:
                        best_A = A_test.copy()
                        best_score = new_max
                        A = A_test.copy()
                        improved = True
                        break
        
        # If no improvement after trying, do a larger perturbation
        if not improved and time.time() < deadline - 10.0:
            # Try a different initial structure
            # Random set with density ~0.5
            if random.random() < 0.1:
                A_rand = np.zeros(129, dtype=bool)
                for i in range(129):
                    A_rand[i] = random.random() < 0.5
                A_repaired, success = repair(A_rand)
                if success:
                    score = max_mult(A_repaired)
                    if score < best_score:
                        best_A = A_repaired.copy()
                        best_score = score
    
    # Final result: return as list of integers
    result = [i for i in range(129) if best_A[i]]
    if len(result) == 0:
        result = list(range(64, 129))
    
    return result
