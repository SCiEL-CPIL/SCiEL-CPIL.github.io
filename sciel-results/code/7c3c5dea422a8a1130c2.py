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

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s
    random.seed(12345)
    
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
    
    # Candidate 1: powers of 2 (baseline)
    try_candidate([2**i for i in range(20)])
    
    # Candidate 2: small geometric-ish bases
    for base in [2, 3, 4, 5, 6]:
        A = [base**i for i in range(20)]
        if max(A) <= 10000:
            try_candidate(A)
        else:
            # Truncate and fill with nearby values
            valid = [x for x in A if x <= 10000]
            if len(valid) >= 20:
                try_candidate(valid[:20])
            else:
                last = valid[-1] if valid else 1
                while len(valid) < 20:
                    last += 1
                    if last <= 10000:
                        valid.append(last)
                try_candidate(valid[:20])
    
    # Local search from several starting points
    def local_search(start_set, max_iters=200000):
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
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            i = random.randrange(20)
            # propose a new value near current or geometric neighborhood
            x = A[i]
            # candidate replacements: neighbors in log-space
            cands = set()
            for factor in [2, 3, 4, 5, 0.5, 1/3, 1/4, 1/5, 1.5, 0.75, 1.25, 2.0, 0.5]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # also random
            cands.add(random.randrange(1, 10001))
            cands.add(x + random.randint(-20, 20))
            cands.add(x + random.randint(-500, 500))
            cands = [c for c in cands if 1 <= c <= 10000 and c not in A]
            if not cands:
                continue
            nv = random.choice(cands)
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
        [3**i for i in range(20) if 3**i <= 10000],
        [2**i for i in range(15)] + [2**15 + k for k in range(1, 6)],
        # Add more diverse starting points
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 6144, 7168, 8192, 8704, 9216, 9728],
        [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 7290, 8019, 8748, 9477, 9700, 9800, 9900, 9950, 9975, 9990, 9999],
    ]
    # Ensure all start points are valid
    valid_start_points = []
    for sp in start_points:
        sp = [x for x in sp if 1 <= x <= 10000]
        seen = set()
        uniq = []
        for x in sp:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            x = random.randrange(1, 10001)
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        valid_start_points.append(uniq[:20])
    start_points = valid_start_points
    
    # Also random multiplicative starts
    for trial in range(50):
        # random increasing geometric-ish
        r = random.uniform(1.05, 1.6)
        start_val = random.randint(1, 5)
        A = []
        v = start_val
        while len(A) < 20:
            A.append(int(v))
            v = v * r * random.uniform(0.9, 1.1)
            if v > 10000:
                # fill rest with random
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
        local_search(sp, max_iters=100000)
    
    # Final polish: try small perturbations of the best set found
    if time.time() < deadline - 5:
        A = list(best_set)
        cur_obj = best_obj
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            i = random.randrange(20)
            x = A[i]
            cands = set()
            for delta in range(-5, 6):
                nv = x + delta
                if 1 <= nv <= 10000 and nv not in A:
                    cands.add(nv)
            for factor in [1.1, 0.9, 1.2, 0.8, 1.5, 0.7]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000 and nv not in A:
                    cands.add(nv)
            if not cands:
                continue
            nv = random.choice(list(cands))
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
