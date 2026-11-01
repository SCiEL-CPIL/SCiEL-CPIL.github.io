import time
import random
import numpy as np
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # 1. Generate initial superpermutation using the standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Setup for fast local search
    current = np.array(word, dtype=np.uint8)
    L = len(current)
    
    # Precompute factorial weights for Lehmer code
    # weights[i] = (n - 1 - i)!
    factorials = [1] * (n + 1)
    for i in range(1, n + 1):
        factorials[i] = factorials[i - 1] * i
    
    weights = np.array([factorials[n - 1 - i] for i in range(n)], dtype=np.int64)
    
    # Map permutation tuple to ID (0 to n!-1)
    # We can compute ID on the fly using the Lehmer code formula
    # ID = sum( count of elements < x[i] in x[i+1:] * weights[i] )
    
    def compute_window_ids(arr, n):
        """
        Compute the permutation ID for every window of length n in arr.
        Returns a numpy array of IDs.
        """
        L = len(arr)
        if L < n:
            return np.array([], dtype=np.int64)
        
        num_windows = L - n + 1
        ids = np.zeros(num_windows, dtype=np.int64)
        
        # For each window start i, we need to compute the Lehmer code.
        # This is O(n^2) per window if done naively.
        # However, n is small (7).
        # We can vectorize across windows?
        # Let's just loop over windows for simplicity, but use numpy for the inner sum if possible.
        # Actually, for n=7, a pure Python loop over ~5000 windows is fast enough if we don't create tuples.
        
        for i in range(num_windows):
            window = arr[i:i+n]
            lehmer_code = 0
            for k in range(n):
                # Count elements in window[k+1:] that are smaller than window[k]
                # window[k+1:] is a numpy array slice
                smaller = np.sum(window[k+1:] < window[k])
                lehmer_code += smaller * weights[k]
            ids[i] = lehmer_code
            
        return ids

    # Initial window IDs
    window_ids = compute_window_ids(current, n)
    
    # Count occurrences of each permutation ID
    target_perms = factorials[n]
    counts = np.zeros(target_perms, dtype=np.int32)
    np.add.at(counts, window_ids, 1)
    
    # Verify validity
    if np.sum(counts > 0) < target_perms:
        return "".join(map(str, current.tolist()))
    
    # Deletion Loop
    # We scan for deletable indices.
    # To improve search, we can randomize the scan order or pick random candidates.
    # Given the time limit, a full scan with random tie-breaking is good.
    
    while time.time() < deadline:
        deletable = []
        
        # Iterate through all possible deletion positions
        # We can sample if L is very large, but L is ~5900, so full scan is fine.
        # To avoid bias, we can shuffle the order of checks.
        indices = np.arange(L)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Windows containing idx:
            # Start indices: max(0, idx-n+1) to min(idx, L-n)
            start_w = max(0, idx - n + 1)
            end_w = min(idx, L - n)
            
            if start_w > end_w:
                continue
            
            # Get IDs of old windows containing idx
            old_ids = window_ids[start_w:end_w+1]
            
            # Check if any old window has count 1 globally?
            # If a permutation has count 1, and it's in old_ids, we MUST ensure it's in new_ids.
            # If a permutation has count > 1, we just need to ensure it doesn't drop to 0.
            
            # Calculate new windows formed by deletion at idx
            # New windows start at j in [max(0, idx-n+1), idx-1]
            # Content: current[j:idx] + current[idx+1 : idx+1 + (n - (idx-j))]
            
            new_ids_list = []
            for j in range(max(0, idx - n + 1), idx):
                prefix_len = idx - j
                suffix_len = n - prefix_len
                
                # Bounds check for suffix
                # We need current[idx+1 : idx+1+suffix_len]
                # Max index needed: idx + suffix_len
                # idx + (n - (idx-j)) = n + j
                # We need n + j <= L - 1  => j <= L - 1 - n
                if j > L - n - 1:
                    continue
                
                # Construct the new window content
                # Instead of creating a list and computing Lehmer code,
                # we can just extract the slice and compute.
                # current[j:idx] and current[idx+1:idx+1+suffix_len]
                new_window = np.concatenate((current[j:idx], current[idx+1:idx+1+suffix_len]))
                
                # Compute ID for this specific window
                lehmer_code = 0
                for k in range(n):
                    smaller = np.sum(new_window[k+1:] < new_window[k])
                    lehmer_code += smaller * weights[k]
                new_ids_list.append(lehmer_code)
            
            new_ids = np.array(new_ids_list, dtype=np.int64)
            
            # Check validity
            # For each unique ID in old_ids, check if it survives.
            # Let's aggregate counts for old and new sets involved.
            # Simpler: 
            # 1. Temporarily decrement counts for old_ids
            # 2. Temporarily increment counts for new_ids
            # 3. Check if any count that was > 0 becomes 0.
            # 4. Restore counts.
            
            # To make this fast, we only care about IDs in old_ids.
            # If an ID is not in old_ids, its count doesn't decrease, so it can't become 0 (unless it was 0, which it isn't).
            # So we only need to check IDs in `old_ids`.
            
            valid = True
            unique_old_ids = np.unique(old_ids)
            
            # We need to know how many times each unique old ID appears in old_ids and new_ids.
            # Since n is small, the number of windows is small (<= n).
            
            for pid in unique_old_ids:
                # Current global count
                c = counts[pid]
                
                # How many times does pid appear in old_ids?
                old_count = np.sum(old_ids == pid)
                
                # How many times does pid appear in new_ids?
                if len(new_ids) > 0:
                    new_count = np.sum(new_ids == pid)
                else:
                    new_count = 0
                
                new_global_count = c - old_count + new_count
                
                if new_global_count <= 0:
                    valid = False
                    break
            
            if valid:
                deletable.append(idx)
                
        if not deletable:
            break
            
        # Pick a random deletable index
        idx_to_delete = random.choice(deletable)
        
        # Perform deletion
        # 1. Update current array
        current = np.delete(current, idx_to_delete)
        L -= 1
        
        # 2. Update window_ids and counts
        # This is the expensive part if we recompute everything.
        # Optimization: Only recompute affected windows.
        # Affected windows in old array: [idx-n+1, idx]
        # Affected windows in new array: [idx-n+1, idx-1] (shifted)
        
        # Let's just recompute the whole window_ids array for simplicity and correctness first, 
        # then optimize if needed. For L~5000, computing 5000 IDs takes ~1-2ms in Python loop.
        # 150s / 2ms = 75000 deletions. We only need ~10. So this is plenty fast.
        
        window_ids = compute_window_ids(current, n)
        
        # Update counts
        counts = np.zeros(target_perms, dtype=np.int32)
        np.add.at(counts, window_ids, 1)
        
        # Check if we still have all perms (should be true if deletion was valid)
        # If not, we made a mistake in logic, but we checked validity.
        
    return "".join(map(str, current.tolist()))
