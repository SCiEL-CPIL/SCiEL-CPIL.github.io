import time
import numpy as np

DEADLINE = time.time() + 85

def has_4ap(arr):
    """Check if sorted array arr contains a nonconstant 4-term AP."""
    s = set(arr)
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            d = b - a
            if d == 0:
                continue
            if (b + d) in s and (b + 2 * d) in s:
                return True
    return False

def try_add_bitset(s_bitset, x):
    """
    Check if adding x (1-indexed) to the bitset keeps it 4-AP-free.
    s_bitset is a numpy array of bools, index 0 is unused (or maps to 1).
    We map value v to index v-1.
    """
    # x is 1-indexed. Index ix = x - 1.
    ix = x - 1
    
    # We need to check if x creates a 4-AP.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # x can be at position 0, 1, 2, or 3 in the AP.
    
    # Case 1: x is the 4th term (a+3d).
    # a = x - 3d. Need a, a+d, a+2d in set.
    # a >= 1 => x - 3d >= 1 => 3d <= x - 1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    if max_d > 0:
        # We need to check all d in [1, max_d].
        # For a specific d, we check indices:
        # (x-3d)-1, (x-3d+d)-1, (x-3d+2d)-1
        # Which are: x-3d-1, x-2d-1, x-d-1
        # We can vectorize this check.
        ds = np.arange(1, max_d + 1)
        idx_a = (x - 3*ds) - 1
        idx_b = (x - 2*ds) - 1
        idx_c = (x - 1*ds) - 1
        # Check if all are True in s_bitset
        if np.all(s_bitset[idx_a] & s_bitset[idx_b] & s_bitset[idx_c]):
            return False # Conflict found (actually, we just need ONE conflict to return False)
            # Wait, if ANY d causes a conflict, we cannot add x.
            # So if np.any(...) is True, return False.
            # Let's correct logic: if there exists a d such that the other 3 are present, conflict.
            # So if np.any(s_bitset[idx_a] & s_bitset[idx_b] & s_bitset[idx_c]): return False
    
    # Case 2: x is the 3rd term (a+2d).
    # a = x - 2d. Need a, a+d, a+3d in set.
    # a >= 1 => 2d <= x - 1 => d <= (x-1)//2
    max_d = (x - 1) // 2
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        idx_a = (x - 2*ds) - 1
        idx_b = (x - 1*ds) - 1
        idx_c = (x + 1*ds) - 1 # a+3d = x-2d+3d = x+d
        if np.any(s_bitset[idx_a] & s_bitset[idx_b] & s_bitset[idx_c]):
            return False

    # Case 3: x is the 2nd term (a+d).
    # a = x - d. Need a, a+2d, a+3d in set.
    # a >= 1 => d <= x - 1
    max_d = x - 1
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        idx_a = (x - 1*ds) - 1
        idx_b = (x + 1*ds) - 1 # a+2d = x-d+2d = x+d
        idx_c = (x + 2*ds) - 1 # a+3d = x-d+3d = x+2d
        # Check bounds: a+3d <= 512 => x+2d <= 512
        # If x+2d > 512, that index is out of bounds or we just don't care if it's out of range 
        # because it can't be in the set [1,512].
        # But s_bitset is size 512 (indices 0..511).
        # We must mask out-of-bounds accesses or ensure we only check valid indices.
        # Easier: only consider d such that x+2d <= 512.
        valid_d = ds[ds <= (512 - x) // 2]
        if len(valid_d) > 0:
             ds = valid_d
             idx_a = (x - 1*ds) - 1
             idx_b = (x + 1*ds) - 1
             idx_c = (x + 2*ds) - 1
             if np.any(s_bitset[idx_a] & s_bitset[idx_b] & s_bitset[idx_c]):
                 return False

    # Case 4: x is the 1st term (a).
    # Need a+d, a+2d, a+3d in set.
    # x+d <= 512, x+2d <= 512, x+3d <= 512 => 3d <= 512 - x => d <= (512-x)//3
    max_d = (512 - x) // 3
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        idx_b = (x + 1*ds) - 1
        idx_c = (x + 2*ds) - 1
        idx_d = (x + 3*ds) - 1
        if np.any(s_bitset[idx_b] & s_bitset[idx_c] & s_bitset[idx_d]):
            return False

    return True

def greedy_vectorized(order):
    """
    Greedy construction using vectorized try_add.
    order: list of 1-indexed integers.
    """
    s_bitset = np.zeros(512, dtype=bool)
    for x in order:
        if try_add_bitset(s_bitset, x):
            s_bitset[x-1] = True
    return s_bitset

def local_search_bitset(s_bitset, max_iters=5000, seed=42):
    """
    Local search on bitset representation.
    """
    rng = np.random.RandomState(seed)
    n = 512
    s = s_bitset.copy()
    current_size = np.sum(s)
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Try to add a random element
        candidates = np.where(~s)[0]
        if len(candidates) == 0:
            break
        
        idx = candidates[rng.randint(len(candidates))]
        x = idx + 1
        
        if try_add_bitset(s, x):
            s[idx] = True
            current_size += 1
        else:
            # Try a swap: remove a random existing element, then try to add x
            existing = np.where(s)[0]
            if len(existing) > 0:
                rem_idx = existing[rng.randint(len(existing))]
                s[rem_idx] = False
                if try_add_bitset(s, x):
                    s[idx] = True
                    # Size stays same (removed 1, added 1)
                else:
                    s[rem_idx] = True # Restore
    
    return s

def construct():
    global DEADLINE
    start = time.time()
    DEADLINE = start + 85
    
    best_bitset = np.zeros(512, dtype=bool)
    best_size = 0
    
    # Phase 1: Generate diverse initial candidates using different orderings
    # 1. Standard Forward
    s1 = greedy_vectorized(list(range(1, 513)))
    if np.sum(s1) > best_size:
        best_size = np.sum(s1)
        best_bitset = s1.copy()
        
    # 2. Standard Backward
    s2 = greedy_vectorized(list(range(512, 0, -1)))
    if np.sum(s2) > best_size:
        best_size = np.sum(s2)
        best_bitset = s2.copy()

    # 3. Base-3 Construction (Known good lower bound, fast)
    s3 = np.zeros(512, dtype=bool)
    for x in range(1, 513):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 2:
                ok = False
                break
            t //= 3
        if ok:
            s3[x-1] = True
    if np.sum(s3) > best_size:
        best_size = np.sum(s3)
        best_bitset = s3.copy()

    # 4. Randomized Greedy (Multiple seeds)
    rng = np.random.RandomState(42)
    for seed in range(10):
        if time.time() > DEADLINE:
            break
        order = list(range(1, 513))
        rng.shuffle(order)
        sr = greedy_vectorized(order)
        if np.sum(sr) > best_size:
            best_size = np.sum(sr)
            best_bitset = sr.copy()
            
    # 5. Weighted Greedy (Prefer "central" or "dense" numbers? Or random weights?)
    # Let's try sorting by random weights to break ties differently
    for seed in range(10, 20):
        if time.time() > DEADLINE:
            break
        weights = rng.rand(512)
        order = [x for x, _ in sorted(zip(range(1, 513), weights), key=lambda z: -z[1])]
        sw = greedy_vectorized(order)
        if np.sum(sw) > best_size:
            best_size = np.sum(sw)
            best_bitset = sw.copy()

    # Phase 2: Local Search on the best candidate
    if time.time() < DEADLINE - 5:
        improved = local_search_bitset(best_bitset, max_iters=20000, seed=123)
        if np.sum(improved) > best_size:
            best_size = np.sum(improved)
            best_bitset = improved

    # Phase 3: Try to improve other top candidates via local search if time permits
    # We only kept the best one, but we could keep top 3. 
    # For simplicity and speed, we just do more local search on the current best.
    if time.time() < DEADLINE - 2:
        # Run local search again with different seed to explore different swaps
        improved2 = local_search_bitset(best_bitset, max_iters=20000, seed=999)
        if np.sum(improved2) > best_size:
            best_size = np.sum(improved2)
            best_bitset = improved2

    # Final Verification
    arr = np.where(best_bitset)[0] + 1
    arr = sorted(arr.tolist())
    
    if has_4ap(arr):
        # Fallback to base-3 if something went wrong (shouldn't happen)
        s3 = np.zeros(512, dtype=bool)
        for x in range(1, 513):
            t = x
            ok = True
            while t > 0:
                if t % 3 == 2:
                    ok = False
                    break
                t //= 3
            if ok:
                s3[x-1] = True
        arr = np.where(s3)[0] + 1
        arr = sorted(arr.tolist())
        
    return arr

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Size: {len(result)}, Time: {time.time()-t0:.2f}s")
    print(f"Valid: {not has_4ap(result)}")
