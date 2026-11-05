import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 512
    N = n + 1  # 1-indexed, indices 1..n
    
    # Precompute all 4-term APs as (a, d) pairs where a, a+d, a+2d, a+3d are all in [1, n]
    # For each element x, precompute a bitmask of elements that conflict with x
    # i.e., adding x to a set S creates a 4-AP iff S intersects conflict_mask[x]
    
    # conflict_mask[x] = set of elements y such that {x, y} is part of some 4-AP
    # Actually, we need: adding x to S creates a 4-AP iff there exist y1, y2, y3 in S 
    # such that {x, y1, y2, y3} is a 4-AP.
    
    # For a given x, the possible 4-APs containing x:
    # Case 1: x is first term: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d in S
    # Case 2: x is second term: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in S
    # Case 3: x is third term: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in S
    # Case 4: x is fourth term: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in S
    
    # For fast checking, for each x, precompute a list of "triple patterns":
    # Each pattern is a tuple of 3 elements that must all be in S for x to create a 4-AP.
    # But that's still slow.
    
    # Better approach: use bitmasks. Represent S as a bitmask (Python int or numpy array).
    # For each x, precompute a bitmask B[x] such that adding x is forbidden iff 
    # (S_mask & some_condition) is non-empty.
    
    # Actually the cleanest fast approach: for each x, precompute the set of "witness triples".
    # But let's use a different strategy: precompute for each x a list of (d, offset_type) 
    # and check using a boolean array.
    
    # Let me use numpy boolean arrays for the set, which gives fast membership.
    
    def check_adds_4ap_np(s_arr, x):
        """Check if adding x to set represented by boolean array s_arr creates a 4-AP.
        s_arr is a numpy boolean array of size N (index 0 unused, 1..n used)."""
        # Case 1: x is first term
        for d in range(1, n - x + 1):
            if s_arr[x + d] and s_arr[x + 2*d] and s_arr[x + 3*d]:
                return True
        # Case 2: x is second term
        for d in range(1, n - x + 1):
            if x - d >= 1 and s_arr[x - d] and s_arr[x + d] and s_arr[x + 2*d]:
                return True
        # Case 3: x is third term
        for d in range(1, x // 2 + 1):
            if x - 2*d >= 1 and s_arr[x - 2*d] and s_arr[x - d] and s_arr[x + d]:
                return True
        # Case 4: x is fourth term
        for d in range(1, x // 3 + 1):
            if x - 3*d >= 1 and s_arr[x - 3*d] and s_arr[x - 2*d] and s_arr[x - d]:
                return True
        return False
    
    def greedy_forward():
        s = set()
        s_arr = np.zeros(N, dtype=bool)
        for x in range(1, n + 1):
            if not check_adds_4ap_np(s_arr, x):
                s.add(x)
                s_arr[x] = True
        return s
    
    def greedy_backward():
        s = set()
        s_arr = np.zeros(N, dtype=bool)
        for x in range(n, 0, -1):
            if not check_adds_4ap_np(s_arr, x):
                s.add(x)
                s_arr[x] = True
        return s
    
    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        s = set()
        s_arr = np.zeros(N, dtype=bool)
        for x in order:
            if not check_adds_4ap_np(s_arr, x):
                s.add(x)
                s_arr[x] = True
        return s
    
    def local_search(s, max_iter=5000):
        """Try to improve set s by swapping elements.
        For each element in s, try removing it and adding a new element not in s."""
        s_arr = np.zeros(N, dtype=bool)
        for x in s:
            s_arr[x] = True
        
        s_list = sorted(s)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i, x in enumerate(s_list):
                if time.time() > deadline:
                    return s
                # Try removing x
                s_arr[x] = False
                s_list.remove(x)
                # Try adding each element not in s
                best_add = None
                for y in range(1, n + 1):
                    if s_arr[y]:
                        continue
                    if not check_adds_4ap_np(s_arr, y):
                        best_add = y
                        break
                if best_add is not None:
                    s_arr[best_add] = True
                    s_list.append(best_add)
                    s_list.sort()
                    improved = True
                    # Restart the scan since the set changed
                    break
                else:
                    # Can't add anything, put x back
                    s_arr[x] = True
                    s_list.append(x)
                    s_list.sort()
        return set(s_list)
    
    # Start with forward greedy
    best_set = greedy_forward()
    best_size = len(best_set)
    
    # Try backward greedy
    if time.time() < deadline - 10:
        s = greedy_backward()
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Try several random greedy restarts
    seed = 42
    while time.time() < deadline - 15:
        s = greedy_random(seed)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        seed += 1
    
    # Local search on best set
    if time.time() < deadline - 5:
        best_set = local_search(best_set, max_iter=2000)
        best_size = len(best_set)
    
    return sorted(best_set)
