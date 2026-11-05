import time
import random
import numpy as np

def set_sizes(A):
    A = np.array(A, dtype=np.int64)
    n = len(A)
    idx = np.triu_indices(n)
    sums = A[idx[0]] + A[idx[1]]
    prods = A[idx[0]] * A[idx[1]]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def generate_smooth_numbers(limit=10000, primes=[2, 3, 5, 7]):
    """Generate all 7-smooth numbers up to limit."""
    smooths = set([1])
    for p in primes:
        for i in range(1, 60):
            p_power = p ** i
            if p_power > limit:
                break
            for s in list(smooths):
                val = s * p_power
                if val <= limit:
                    smooths.add(val)
    return sorted(list(smooths))

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s
    random.seed(12345)
    
    # Precompute 7-smooth numbers
    smooths = generate_smooth_numbers(10000)
    smooths_set = set(smooths)
    
    best_set = list(range(1, 21))
    best_obj = objective(best_set)
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        # validate: 20 distinct positive ints <= 10000
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        A = list(A)
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]
    
    # Candidate 1: Powers of 2 (baseline)
    try_candidate([2**i for i in range(20)])
    
    # Candidate 2: Small 7-smooth numbers
    if len(smooths) >= 20:
        try_candidate(smooths[:20])
        
    # Candidate 3: Random 20 7-smooth numbers
    if len(smooths) >= 20:
        try_candidate(random.sample(smooths, 20))
        
    # Candidate 4: Mixed small integers and smooths
    mixed = list(range(1, 11)) + random.sample([x for x in smooths if x > 10], 10)
    try_candidate(mixed)
    
    # Local search from several starting points
    def local_search(start_set, max_iters=100000):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        # global best tracking
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        it = 0
        A_arr = np.array(A, dtype=np.int64)
        idx = np.triu_indices(20)
        
        # Precompute neighbors in smooths for faster candidate generation
        # Map value to index in smooths list for O(1) neighbor lookup
        smooth_index = {v: i for i, v in enumerate(smooths)}
        
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            i = random.randrange(20)
            # propose a new value near current or geometric neighborhood
            x = A[i]
            
            cands = set()
            
            # 1. Smooth neighbors
            if x in smooth_index:
                idx_s = smooth_index[x]
                for offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    ni = idx_s + offset
                    if 0 <= ni < len(smooths):
                        cands.add(smooths[ni])
            
            # 2. Log-space neighbors (geometric)
            for factor in [2, 3, 4, 0.5, 1/3, 1.5, 0.75, 1.25]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            
            # 3. Random
            cands.add(random.randrange(1, 10001))
            cands.add(x + random.randint(-20, 20))
            
            # Filter valid candidates
            A_set = set(A)
            valid_cands = [c for c in cands if 1 <= c <= 10000 and c not in A_set]
            
            if not valid_cands:
                continue
            
            nv = random.choice(valid_cands)
            old = A[i]
            A[i] = nv
            A_arr[i] = nv
            sums = A_arr[idx[0]] + A_arr[idx[1]]
            prods = A_arr[idx[0]] * A_arr[idx[1]]
            new_obj = max(len(np.unique(sums)), len(np.unique(prods)))
            if new_obj <= cur_obj:
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
                A_arr[i] = old
        return
    
    # Starting points for local search
    start_points = [
        list(range(1, 21)),
        [2**i for i in range(20)],
        smooths[:20],
        random.sample(smooths, 20) if len(smooths) >= 20 else smooths,
        # Random mix of small and smooth
        list(range(1, 10)) + random.sample([x for x in smooths if x > 10], 12) if len([x for x in smooths if x > 10]) >= 12 else smooths[:20],
    ]
    
    # Add a few random geometric starts
    for trial in range(10):
        r = random.uniform(1.2, 1.8)
        start_val = random.randint(1, 5)
        A = []
        v = start_val
        while len(A) < 20:
            A.append(int(v))
            v = v * r
            if v > 10000:
                break
        while len(A) < 20:
            A.append(random.randrange(1, 10001))
        # dedupe
        seen = set()
        uniq = []
        for x in A:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            x = random.randrange(1, 10001)
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        start_points.append(uniq[:20])
    
    for sp in start_points:
        if time.time() > deadline - 2:
            break
        local_search(sp, max_iters=50000)
    
    # Final polish: try to replace elements with nearby smooth numbers
    if time.time() < deadline - 5:
        A = list(best_set)
        cur_obj = best_obj
        smooth_index = {v: i for i, v in enumerate(smooths)}
        
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            i = random.randrange(20)
            x = A[i]
            cands = set()
            
            # Try nearby smooth numbers
            if x in smooth_index:
                idx_s = smooth_index[x]
                for offset in [-10, -5, -2, -1, 1, 2, 5, 10]:
                    ni = idx_s + offset
                    if 0 <= ni < len(smooths):
                        cands.add(smooths[ni])
            
            # Also try small additive changes
            for delta in range(-5, 6):
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            
            A_set = set(A)
            valid_cands = [c for c in cands if 1 <= c <= 10000 and c not in A_set]
            
            if not valid_cands:
                continue
            
            nv = random.choice(valid_cands)
            old = A[i]
            A[i] = nv
            new_obj = objective(A)
            if new_obj <= cur_obj:
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
    
    return best_set
