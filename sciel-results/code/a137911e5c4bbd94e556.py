import time
import math
import bisect
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 82  # 82 second budget, 8s margin to 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators for exact integer arithmetic
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
        
    # Weights w_i = L / d_i. We want subset sum S < L maximizing S.
    weights = [L // d for d in denominators]
    
    # Split into two halves for Meet-in-the-Middle
    mid = n // 2
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    # Generate all subset sums for A and B with masks
    # To save memory and time, we store just the sums and masks.
    # 2^23 is ~8M, 2^24 is ~16M. This is heavy but feasible in C-optimized numpy?
    # No, we need exact big ints. Python lists of ints are okay-ish but slow to sort.
    # Let's try to be smart. We only care about sums < L.
    
    def gen_sums(weights_subset):
        sums = [(0, 0)]
        for i, w in enumerate(weights_subset):
            # Use list comprehension for speed
            # new_sums = [(s + w, m | (1 << i)) for s, m in sums]
            # This creates a new list.
            # We can extend in place? No, we need to iterate over original.
            # Let's just do it.
            current_len = len(sums)
            for k in range(current_len):
                s, m = sums[k]
                sums.append((s + w, m | (1 << i)))
        return sums
    
    # This might be slow. Let's optimize generation.
    # Actually, for 23 elements, 8M tuples is a lot of memory (~100MB+).
    # And sorting 8M tuples is slow in Python.
    
    # Alternative: Use a different strategy if MITM is too slow.
    # The current program uses local search.
    # The inspiration uses MITM.
    # Let's try a hybrid: 
    # 1. Try MITM with a time limit. If it finishes, it's optimal (or close).
    # 2. If it's too slow, fall back to local search.
    
    # However, generating 2^23 sums in Python might take > 10-20s.
    # Sorting might take > 10-20s.
    # Total might exceed 80s.
    
    # Let's try to reduce the search space.
    # Note that 1/2 + 1/3 + 1/7 + 1/43 = 1 is a known identity.
    # We want sum < 1.
    
    # Let's try a different structural approach:
    # Branch and Bound with strong pruning.
    # Sort denominators by 1/d descending (i.e., d ascending).
    # This is what the current program does greedily.
    
    # Let's stick to the MITM idea but make it robust.
    # If MITM is too slow, we return the best found so far (which might be partial).
    
    # Actually, let's try to generate sums using arrays of ints if possible?
    # No, L is huge.
    
    # Let's try to prune sums_A.
    # If we have two sums s1, s2 with s1 < s2 and mask1 is a subset of mask2? No.
    # Pareto pruning: if s1 <= s2 and popcount(mask1) >= popcount(mask2)? No, we want max sum.
    # For a given sum value, we don't care about the mask for the *value*, only for reconstruction.
    # So we can just keep the unique sums?
    # No, different masks can yield the same sum. We just need one mask for that sum.
    # So we can use a dict: sum -> mask.
    # This reduces the list size if there are collisions.
    # But with large random-like weights, collisions are rare.
    
    # Let's try the MITM with a strict time check during generation.
    
    def gen_sums_with_time(weights_subset, time_limit):
        sums = [(0, 0)]
        for i, w in enumerate(weights_subset):
            if time.time() > time_limit:
                return None
            current_len = len(sums)
            # Optimize: use list extension
            # new_part = []
            # for k in range(current_len):
            #    s, m = sums[k]
            #    new_part.append((s+w, m | (1<<i)))
            # sums.extend(new_part)
            
            # Faster:
            bit = 1 << i
            for k in range(current_len):
                s, m = sums[k]
                sums.append((s + w, m | bit))
        return sums
    
    # Time budget for generation: 20s each?
    # 2^23 iterations in Python loop... 8M iterations.
    # Python does ~10-50M simple ops/sec?
    # Tuple creation and append is slow. Maybe 1M/sec.
    # So 8M might take 8-80s. Too risky.
    
    # Let's go back to Local Search but improve it significantly.
    # The current program's local search is weak.
    # Let's use Simulated Annealing or Tabu Search.
    # Or, better, use the MITM idea on a *subset* of the problem?
    
    # What if we fix the first few large denominators?
    # 1/2, 1/3, 1/4, 1/5, 1/6 ...
    
    # Let's try a different structural change:
    # Integer Linear Programming? No, no external libs.
    
    # Let's try to make the MITM work by splitting unevenly?
    # Or, use numpy for the sums?
    # If we use numpy arrays of object dtype (big ints), it's slow.
    # If we use 64-bit ints, L is too big (LCM(1..48) is ~ 5e17, which fits in 64-bit signed?
    # 2^63 is ~ 9.2e18. Yes, L fits in int64!
    # Weights fit in int64.
    # Sums fit in int64.
    # So we CAN use numpy!
    
    import numpy as np
    
    # Convert weights to numpy int64 array
    w_np = np.array(weights, dtype=np.int64)
    L_np = np.int64(L)
    
    # Split
    mid = n // 2
    wA = w_np[:mid]
    wB = w_np[mid:]
    
    # Generate all subset sums for A using numpy?
    # This is hard to do efficiently with variable length subsets in numpy without loops.
    # But we can use a trick:
    # For each bit position, add the weight to all existing sums if the bit is set?
    # No, standard way is iterative doubling.
    
    # Let's try to generate sums_A in numpy.
    # Start with [0]
    # sums = np.array([0], dtype=np.int64)
    # for w in wA:
    #    sums = np.concatenate([sums, sums + w])
    # This creates new arrays each time.
    # Size doubles. 2^23 is 8M.
    # 23 iterations.
    # Concatenation is fast in numpy.
    # Let's try this.
    
    def gen_sums_numpy(w):
        sums = np.array([0], dtype=np.int64)
        for weight in w:
            # sums = np.concatenate((sums, sums + weight))
            # This is O(N). Total O(N^2) memory ops?
            # 8M * 8 bytes = 64MB.
            # 23 * 64MB = ~1.5GB memory traffic. Should be fast (<1s).
            sums = np.concatenate((sums, sums + weight))
        return sums
    
    # We need masks too for reconstruction.
    # But if we just find the max sum < L, we need to know which elements.
    # We can store the masks in a separate array?
    # Or, we can reconstruct by checking? No, too slow.
    # We need to store the mask.
    # Masks are 23 bits, fit in int32 or int64.
    
    def gen_sums_numpy_with_masks(w):
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        for i, weight in enumerate(w):
            bit = 1 << i
            new_sums = sums + weight
            new_masks = masks | bit
            sums = np.concatenate((sums, new_sums))
            masks = np.concatenate((masks, new_masks))
        return sums, masks
    
    # Time check
    t_start = time.time()
    try:
        sums_A, masks_A = gen_sums_numpy_with_masks(wA)
        if time.time() > deadline - 40:
            raise TimeoutError
        sums_B, masks_B = gen_sums_numpy_with_masks(wB)
    except Exception:
        # Fallback to local search if MITM fails or is too slow
        return local_search_fallback(start, deadline)
    
    t_gen = time.time() - t_start
    
    # Sort sums_A
    # np.sort is fast.
    # But we need to keep masks aligned.
    # We can use np.lexsort or argsort.
    
    # We want to find max sA + sB < L.
    # Sort sums_A ascending.
    # For each sB, find max sA < L - sB.
    
    # Sort A
    order_A = np.argsort(sums_A)
    sorted_sums_A = sums_A[order_A]
    sorted_masks_A = masks_A[order_A]
    
    # We don't need to sort B, but it might help with two-pointer?
    # Two pointer:
    # i points to end of A (largest).
    # Iterate B.
    # If we sort B ascending, then L - sB decreases.
    # So i can only decrease.
    
    # Sort B
    order_B = np.argsort(sums_B)
    sorted_sums_B = sums_B[order_B]
    sorted_masks_B = masks_B[order_B]
    
    # Two pointer search
    # i = len(A) - 1
    # best_sum = -1
    # best_mask_A = 0
    # best_mask_B = 0
    
    # We want max sum < L.
    # Since we want strict inequality, sum <= L-1.
    
    i = len(sorted_sums_A) - 1
    best_sum = -1
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through B
    # To speed up, we can iterate B in ascending order.
    # As sB increases, max allowed sA decreases.
    
    for j in range(len(sorted_sums_B)):
        sB = sorted_sums_B[j]
        # We need sA <= L - 1 - sB
        target = L_np - 1 - sB
        
        if target < 0:
            # Since B is sorted ascending, all subsequent sB will be larger, so target even smaller.
            break
            
        # Move i down while sA[i] > target
        while i >= 0 and sorted_sums_A[i] > target:
            i -= 1
            
        if i < 0:
            break
            
        sA = sorted_sums_A[i]
        total = sA + sB
        
        if total > best_sum:
            best_sum = total
            best_mask_A = sorted_masks_A[i]
            best_mask_B = sorted_masks_B[j]
            
        # Time check
        if time.time() > deadline - 5:
            break
    
    if best_sum < 0:
        return local_search_fallback(start, deadline)
    
    # Reconstruct indices
    selected = []
    for i in range(mid):
        if best_mask_A & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if best_mask_B & (1 << i):
            selected.append(mid + i)
            
    # Verify with exact fractions
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        return selected
    else:
        return local_search_fallback(start, deadline)

def local_search_fallback(start, deadline):
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # Greedy init
    order = sorted(range(n), key=lambda i: -recips[i])
    selected = set()
    total = Fraction(0)
    for idx in order:
        if total + recips[idx] < 1:
            selected.add(idx)
            total += recips[idx]
    
    best_total = total
    best_selected = set(selected)
    best_deficit = 1 - total
    
    search_deadline = deadline
    
    def try_swap():
        nonlocal best_total, best_selected, best_deficit
        improved = False
        sel_list = list(best_selected)
        unsel_list = [i for i in range(n) if i not in best_selected]
        
        for rem in sel_list:
            rem_r = recips[rem]
            new_total = best_total - rem_r
            for add in unsel_list:
                add_r = recips[add]
                if new_total + add_r < 1:
                    new_def = 1 - (new_total + add_r)
                    if new_def < best_deficit:
                        best_deficit = new_def
                        best_total = new_total + add_r
                        best_selected.discard(rem)
                        best_selected.add(add)
                        improved = True
                        break
                if time.time() > search_deadline:
                    return True
            if time.time() > search_deadline:
                return True
        return improved
    
    def try_add():
        nonlocal best_total, best_selected, best_deficit
        improved = False
        unsel_list = [i for i in range(n) if i not in best_selected]
        for add in unsel_list:
            add_r = recips[add]
            if best_total + add_r < 1:
                new_def = 1 - (best_total + add_r)
                if new_def < best_deficit:
                    best_deficit = new_def
                    best_total += add_r
                    best_selected.add(add)
                    improved = True
            if time.time() > search_deadline:
                break
        return improved
    
    while time.time() < search_deadline:
        improved = False
        for _ in range(5):
            if try_swap():
                improved = True
            if time.time() > search_deadline:
                break
        if time.time() > search_deadline:
            break
        if try_add():
            improved = True
        if not improved:
            break
            
    final_total = sum(recips[i] for i in best_selected)
    if final_total < 1 and len(best_selected) > 0:
        return list(best_selected)
    else:
        return [0]

# Entry point
# The construct function is defined above.
# We need to make sure it's the one called.
