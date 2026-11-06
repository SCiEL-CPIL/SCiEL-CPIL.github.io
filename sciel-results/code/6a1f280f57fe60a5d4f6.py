"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 22
    # Calculate LCM of 1..22 to convert to integer sums
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value
    vals = {x: L // x for x in range(1, n + 1)}
    
    # Sort elements by value descending (smaller x -> larger reciprocal)
    # This tends to create "sparse" sums, which are more likely to be distinct
    elements = sorted(range(1, n + 1), key=lambda x: -vals[x])
    sorted_vals = [vals[x] for x in elements]
    
    best_result = [1]  # fallback
    
    # --- MITM Strategy for High Cardinality ---
    
    def generate_dissociated_subsets(v_list):
        """Generate all dissociated subsets of v_list.
        Returns list of (size, sums_set, diffs_set)."""
        results = []
        
        def dfs(start, current_sums, current_indices):
            if time.time() > deadline:
                return
            # Record current subset
            # Note: current_sums is the set of all subset sums of current_indices
            # We compute diffs
            # Optimization: Only compute diffs if we might need them?
            # For now, compute for all valid subsets found.
            diffs = set()
            for s1 in current_sums:
                for s2 in current_sums:
                    diffs.add(s1 - s2)
            results.append((len(current_indices), current_sums.copy(), diffs))
            
            for i in range(start, len(v_list)):
                v = v_list[i]
                shifted = {s + v for s in current_sums}
                if not (shifted & current_sums):
                    new_sums = current_sums | shifted
                    dfs(i + 1, new_sums, current_indices + [i])
        
        dfs(0, {0}, [])
        return results

    # Split elements into two halves
    mid = n // 2
    half1_vals = sorted_vals[:mid]
    half2_vals = sorted_vals[mid:]
    
    # Generate subsets for both halves
    # We do this once, then reuse for different K
    subsets1 = generate_dissociated_subsets(half1_vals)
    subsets2 = generate_dissociated_subsets(half2_vals)
    
    # Index subsets by size for faster lookup
    from collections import defaultdict
    sub1_by_size = defaultdict(list)
    for size, sums, diffs in subsets1:
        sub1_by_size[size].append((sums, diffs))
        
    sub2_by_size = defaultdict(list)
    for size, sums, diffs in subsets2:
        sub2_by_size[size].append((sums, diffs))

    # Try to find a solution with K > 16 using MITM
    for K in range(n, 16, -1):
        if time.time() > deadline:
            break
        found = False
        # k1 + k2 = K
        for k1 in range(max(0, K - (n - mid)), min(mid, K) + 1):
            k2 = K - k1
            if k1 not in sub1_by_size or k2 not in sub2_by_size:
                continue
            
            for sums1, diffs1 in sub1_by_size[k1]:
                if time.time() > deadline:
                    break
                for sums2, diffs2 in sub2_by_size[k2]:
                    # Check if union is dissociated
                    # Condition: (diffs1 & diffs2) == {0}
                    if not (diffs1 & diffs2) or (diffs1 & diffs2) == {0}:
                        # Reconstruct the set
                        # We need to map indices back to elements
                        # This is tricky because we didn't store indices in the result list
                        # We need to modify generate_dissociated_subsets to store indices
                        pass
                        
        # If found, we need to return the actual elements
        # The current generate function doesn't return indices.
        # Let's modify it.

    # --- Fallback DFS for Lower Cardinality ---
    # ... (Keep existing DFS code but maybe skip if MITM found something)
    
    # Since I need to provide a complete working solution, I will refactor the MITM part to correctly return indices.

    def generate_dissociated_subsets_with_indices(v_list, elem_indices):
        """Generate all dissociated subsets.
        Returns list of (size, sums_set, diffs_set, indices_list)."""
        results = []
        
        def dfs(start, current_sums, current_indices):
            if time.time() > deadline:
                return
            diffs = set()
            for s1 in current_sums:
                for s2 in current_sums:
                    diffs.add(s1 - s2)
            results.append((len(current_indices), current_sums.copy(), diffs, current_indices.copy()))
            
            for i in range(start, len(v_list)):
                v = v_list[i]
                shifted = {s + v for s in current_sums}
                if not (shifted & current_sums):
                    new_sums = current_sums | shifted
                    dfs(i + 1, new_sums, current_indices + [i])
        
        dfs(0, {0}, [])
        return results

    # Re-generate with indices
    # Note: This might be slow if called repeatedly. We call it once.
    # The previous call was a dry run? No, I should just do it once correctly.
    
    # Let's restart the MITM setup properly.
    
    # Clear previous lists
    subsets1 = []
    subsets2 = []
    
    # To save time, we can just use the indices from the sorted list
    # elements[0..mid-1] correspond to half1
    # elements[mid..n-1] correspond to half2
    
    # We need to map local indices in half to global indices in elements
    # local index i in half1 -> global index i
    # local index i in half2 -> global index mid + i
    
    # Let's just use the function above.
    
    # However, generating all subsets of 11 elements might be slow if there are many.
    # 2^11 = 2048.
    # For each, O(2^k^2) for diffs.
    # If k=11, 2^11=2048 sums. 2048^2 = 4M ops.
    # 2048 subsets * 4M ops = 8B ops. Too slow.
    
    # We must prune the generation or the diff calculation.
    # We only care about large K.
    # If K is large, k1 and k2 are large.
    # But we generate ALL subsets.
    
    # Optimization: Only store subsets with size >= min_size_needed?
    # We don't know min_size_needed until we start checking K.
    # But we can guess that if K > 16, then k1 >= K - 11 and k2 >= K - 11.
    # So for K=17, k1 >= 6.
    # We can filter results after generation.
    
    # The bottleneck is the diff calculation.
    # Can we avoid computing diffs for small subsets?
    # Yes. If we are looking for K > 16, we only need subsets with size >= 6 or 7.
    # But we don't know K in advance.
    
    # Let's try to just run the DFS for K=22..17 as well, but with better pruning?
    # The original DFS was slow.
    
    # Alternative: Use the MITM but only for the TOP candidates?
    # No, we need to check all.
    
    # Let's stick to the DFS but improve the pruning.
    # The current DFS uses `shifted & current_sums`.
    
    # What if we use a different ordering?
    # Randomized restarts?
    
    # Given the time constraint and the risk of the MITM being too slow,
    # I will improve the DFS with a better heuristic:
    # Try to add elements that are "far" from existing sums?
    # Or use a greedy approach to fill the rest?
    
    # Let's try a simple improvement:
    # In the DFS, instead of iterating i from start to n,
    # we can prioritize elements that are less likely to cause collisions.
    
    # However, the most robust improvement is likely just ensuring the DFS is as fast as possible.
    # Using `set` operations is already fast.
    
    # Let's try to combine:
    # 1. Greedy to get a good lower bound.
    # 2. DFS to try to exceed it.
    
    # The current code does this.
    
    # Let's try to optimize the `dfs` function in the current code.
    # It creates `new_sums = current_sums | shifted`.
    # This creates a new set.
    
    # We can try to use a list of sums and check collisions manually?
    # No, set is better.
    
    # I will provide a version that uses the MITM idea but with a strict time limit on the generation phase,
    # and if it's too slow, it falls back to DFS.
    
    # Actually, let's just refine the DFS to be more aggressive in pruning.
    # If we have `remaining_needed` elements to pick, and we have `n - start_idx` available,
    # we can check if the "max possible size" from the remaining elements is sufficient.
    # This is already done: `if start_idx + remaining_needed > n: return False`.
    
    # Another pruning:
    # If the sum of the largest `remaining_needed` remaining elements is less than the min difference in current_sums?
    # No, that's not a valid constraint.
    
    # I will implement the MITM with a timeout on the generation of subsets.
    # If generation takes too long, we abort MITM and go to DFS.

    # Re-implementing construct with MITM attempt.
    
    def _mitm_search(min_k):
        """Try to find a set of size >= min_k using MITM.
        Returns list of elements or None."""
        start_mitm = time.time()
        mitm_time_limit = 10.0
        
        # Generate subsets for halves
        # We only care about subsets that could contribute to K >= min_k
        # So size >= min_k - (n - mid)
        
        def gen_subs(v_list, min_size_needed):
            res = []
            def dfs(start, sums, inds):
                if time.time() - start_mitm > mitm_time_limit:
                    return
                if len(inds) >= min_size_needed:
                    # Compute diffs
                    diffs = set()
                    for s1 in sums:
                        for s2 in sums:
                            diffs.add(s1 - s2)
                    res.append((len(inds), sums, diffs, inds))
                
                for i in range(start, len(v_list)):
                    v = v_list[i]
                    shifted = {s + v for s in sums}
                    if not (shifted & sums):
                        dfs(i + 1, sums | shifted, inds + [i])
            dfs(0, {0}, [])
            return res

        min_size_1 = max(0, min_k - (n - mid))
        min_size_2 = max(0, min_k - mid)
        
        subs1 = gen_subs(half1_vals, min_size_1)
        subs2 = gen_subs(half2_vals, min_size_2)
        
        if time.time() - start_mitm > mitm_time_limit:
            return None

        # Check combinations
        for K in range(n, min_k - 1, -1):
            for k1 in range(max(min_size_1, K - (n - mid)), min(mid, K) + 1):
                k2 = K - k1
                if k2 < min_size_2 or k2 > n - mid:
                    continue
                
                # Filter subs by size
                cands1 = [s for s in subs1 if s[0] == k1]
                cands2 = [s for s in subs2 if s[0] == k2]
                
                for s1 in cands1:
                    for s2 in cands2:
                        if not (s1[2] & s2[2]) or (s1[2] & s2[2]) == {0}:
                            # Found
                            idxs1 = s1[3]
                            idxs2 = s2[3]
                            # Map back to elements
                            # idxs1 are indices in half1 (0..mid-1)
                            # idxs2 are indices in half2 (0..n-mid-1)
                            final_indices = []
                            for i in idxs1:
                                final_indices.append(elements[i])
                            for i in idxs2:
                                final_indices.append(elements[mid + i])
                            return final_indices
        return None

    # Try MITM for K > 16
    result = _mitm_search(17)
    if result:
        return result

    # Fallback to DFS for K <= 16
    # ... (Use the existing DFS code)
    
    def _is_dissociated_fast(v_list):
        if not v_list:
            return True
        sums = {0}
        for v in v_list:
            shifted = {s + v for s in sums}
            if shifted & sums:
                return False
            sums |= shifted
        return True

    def _greedy_build():
        result = []
        result_sums = {0}
        for i in range(n):
            v = sorted_vals[i]
            shifted = {s + v for s in result_sums}
            if not (shifted & result_sums):
                result.append(i)
                result_sums |= shifted
        return result

    greedy_result = _greedy_build()
    if len(greedy_result) > len(best_result):
        best_result = [elements[i] for i in greedy_result]
    
    best_k = len(best_result)
    
    for k in range(n, best_k, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        def dfs(start_idx, current_indices, current_sums):
            nonlocal found, best_result, best_k
            
            if time.time() > deadline:
                return True
            
            cur_len = len(current_indices)
            if cur_len == k:
                best_result = [elements[i] for i in current_indices]
                best_k = k
                found = True
                return True
            
            remaining_needed = k - cur_len
            if start_idx + remaining_needed > n:
                return False
            
            for i in range(start_idx, n - remaining_needed + 1):
                v = sorted_vals[i]
                shifted = {s + v for s in current_sums}
                if shifted & current_sums:
                    continue
                new_sums = current_sums | shifted
                new_indices = current_indices + [i]
                
                if dfs(i + 1, new_indices, new_sums):
                    return True
                if found:
                    return True
                    
            return False

        initial_sums = {0}
        if dfs(0, [], initial_sums):
            break
            
    return best_result
# EVOLVE-BLOCK-END
