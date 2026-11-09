import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # 2. Group numbers by sigma value using a dictionary
    # This is O(N) and avoids the O(N log N) sort
    groups = {}
    # sigma[1:] corresponds to n=1..N
    sig_vals = sigma[1:]
    
    # We need to iterate and populate the dictionary.
    # Vectorized population is tricky, so we do it in a loop or use np.unique?
    # np.unique returns unique values and inverse indices.
    # unique_sig, inverse = np.unique(sig_vals, return_inverse=True)
    # Then we can use np.bincount or similar to gather?
    # Actually, building a dict of lists is easier in a loop, but a loop over 10^7 in Python is slow.
    # Let's try to use numpy to group.
    
    # Alternative: Use np.unique to get sorted unique sigma values and counts.
    # But we need the actual n's.
    # We can use the fact that if we sort by sigma, the n's are grouped.
    # But sorting is slow.
    
    # Let's stick to the dictionary approach but optimize the population.
    # Since Python loops are slow, we can use a trick:
    # If we assume the number of unique sigma values is manageable, we can process them.
    # However, there are ~10^7 unique sigma values? No, sigma(n) grows roughly as n log log n.
    # The range of sigma(n) for n<=10^7 is up to ~10^8.
    # So the number of unique sigma values is large.
    
    # Let's reconsider the sort.
    # np.argsort on 10^7 elements might take 5-10 seconds.
    # The original code did that. Why did it score 4077490?
    # 4077490 corresponds to an error of ~1/3?
    # 1/3 is 0.333...
    # 4077490 is not 1/3.
    # Wait, the score is seed/value.
    # If score is 4077490, and seed is likely 1, then value is ~2.4e-7?
    # Or is the score the value itself?
    # "The search score is seed/value"
    # If seed=1, score=4077490, then value = 1/4077490 ~ 2.45e-7.
    # This is a very small error.
    # |a/b - 3/2| = |2a-3b|/(2b).
    # If |2a-3b|=1, error is 1/(2b).
    # For error ~ 2.45e-7, 2b ~ 4e6, so b ~ 2e6.
    # So we found a pair with |2a-3b|=1 and b ~ 2e6.
    # This implies the original code *did* work and found a good pair.
    
    # So why improve?
    # Maybe it can find an even better one (smaller error) if it runs faster and checks more?
    # Or maybe the current code is just slow and we can speed it up to ensure it finishes and potentially allow for more iterations or a larger N?
    # N is fixed at 10^7.
    
    # If the current code already finds a pair with error ~2.45e-7, that's very good.
    # Can we find |2a-3b|=1 with larger b?
    # We want to maximize b such that there exists a with sigma(a)=sigma(b) and 2a=3b+1 or 2a=3b-1.
    # This is a specific number theory problem.
    
    # Let's try to speed up the grouping to ensure we don't hit the timeout, and maybe use the saved time to verify or extend the search if possible?
    # No, N is fixed.
    
    # Let's try the dictionary approach again, but efficiently.
    # We can use `np.unique` with `return_inverse` and then use `np.bincount` to get counts, but we need the indices.
    
    # Actually, the sort approach in the seed is standard.
    # Let's try to optimize the inner loop of the seed.
    # The seed uses `np.searchsorted` for each group.
    
    # Let's try a different strategy for the search:
    # Instead of iterating all groups, we can iterate all b, compute target_a = 1.5 * b,
    # and check if there is an a close to target_a with sigma(a)=sigma(b).
    # This requires a map from sigma value to a sorted list of n's.
    # This is the dictionary approach.
    
    # Building the dictionary:
    # We can do this by iterating over n.
    # 10^7 iterations in Python might take 1-2 seconds if simple.
    
    groups = {}
    # Use a list for speed? No, dict is fine.
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    # This loop might be slow in Python.
    # Let's use NumPy to speed up the grouping?
    # It's hard to build a dict of lists with NumPy directly.
    
    # Let's stick to the Sort method but optimize the search part.
    # The sort method is:
    sort_idx = np.argsort(sigma[1:], kind='stable')
    sorted_sig = sigma[1:][sort_idx]
    sorted_nums = np.arange(1, N + 1)[sort_idx]
    
    changes = np.concatenate(([True], sorted_sig[1:] != sorted_sig[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [N]))
    
    best_a = 11
    best_b = 6
    best_num = 4
    best_den = 12
    
    for gi in range(len(group_starts)):
        if time.time() > deadline - 5:
            break
        
        gs = group_starts[gi]
        ge = group_ends[gi]
        group_size = ge - gs
        
        if group_size < 2:
            continue
        
        group_arr = np.sort(sorted_nums[gs:ge])
        
        # Vectorized search for best pair in this group
        # We want min |2a - 3b| / (2b)
        # For each a, b is near 2a/3.
        
        targets = (2.0 * group_arr) / 3.0
        idxs = np.searchsorted(group_arr, targets, side='left')
        idxs = np.clip(idxs, 0, len(group_arr) - 1)
        
        cand1 = group_arr[idxs]
        cand2 = group_arr[np.clip(idxs - 1, 0, len(group_arr) - 1)]
        
        diff1 = np.abs(2 * group_arr - 3 * cand1)
        diff2 = np.abs(2 * group_arr - 3 * cand2)
        
        score1 = diff1 / (2.0 * cand1)
        score2 = diff2 / (2.0 * cand2)
        
        all_scores = np.minimum(score1, score2)
        best_local_idx = np.argmin(all_scores)
        
        best_score = all_scores[best_local_idx]
        
        if score1[best_local_idx] <= score2[best_local_idx]:
            best_b_local = cand1[best_local_idx]
            best_diff = diff1[best_local_idx]
        else:
            best_b_local = cand2[best_local_idx]
            best_diff = diff2[best_local_idx]
        
        best_a_local = group_arr[best_local_idx]
        
        if best_diff * best_den < best_num * (2 * best_b_local):
            best_num = int(best_diff)
            best_den = int(2 * best_b_local)
            best_a = int(best_a_local)
            best_b = int(best_b_local)
            
    if best_a == best_b:
        best_a = 11
        best_b = 6
        best_num = 4
        best_den = 12
        
    return [int(best_a), int(best_b)]
