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
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    random.seed(42)

    best_set = list(range(1, 21))
    best_obj = objective(best_set)

    def update_best(A, obj):
        nonlocal best_set, best_obj
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)

    # --- Phase 1: Systematic search over structured families ---

    # Family 1: Pure geometric progression a * r^i for i=0..19
    for a in range(1, 30):
        for r_num in range(1, 12):
            for r_den in range(1, 12):
                if r_num == r_den:
                    continue
                r = r_num / r_den
                if r <= 1.0:
                    continue
                A = []
                v = float(a)
                for i in range(20):
                    x = int(round(v))
                    if x < 1:
                        x = 1
                    if x > 10000:
                        break
                    if x in A:
                        break
                    A.append(x)
                    v *= r
                if len(A) == 20:
                    obj = objective(A)
                    update_best(A, obj)

    # Family 2: Two-segment geometric (small step then large step)
    for a in range(1, 15):
        for r1 in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
            for r2 in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
                for split in range(5, 16):
                    A = []
                    v = float(a)
                    for i in range(20):
                        if i < split:
                            x = int(round(v))
                            v *= r1
                        else:
                            x = int(round(v))
                            v *= r2
                        if x < 1:
                            x = 1
                        if x > 10000:
                            break
                        if x in A:
                            break
                        A.append(x)
                    if len(A) == 20:
                        obj = objective(A)
                        update_best(A, obj)

    # Family 3: Mixed bases - alternate between two growth rates
    for a in range(1, 10):
        for r1 in [1.2, 1.5, 2.0, 2.5]:
            for r2 in [1.3, 1.6, 2.0, 2.5, 3.0]:
                A = []
                v = float(a)
                for i in range(20):
                    x = int(round(v))
                    if x < 1:
                        x = 1
                    if x > 10000:
                        break
                    if x in A:
                        break
                    A.append(x)
                    v *= r1 if i % 2 == 0 else r2
                if len(A) == 20:
                    obj = objective(A)
                    update_best(A, obj)

    # Family 4: Near-arithmetic with geometric tail
    for a in range(1, 8):
        for d in range(1, 5):
            for geom_start in range(10, 18):
                for gr in [1.5, 2.0, 2.5, 3.0]:
                    A = []
                    for i in range(geom_start):
                        x = a + i * d
                        if x > 10000 or x in A:
                            break
                        A.append(x)
                    v = float(A[-1] if A else a) * 1.01
                    while len(A) < 20:
                        x = int(round(v))
                        if x > 10000 or x in A:
                            x += 1
                            if x > 10000:
                                break
                        A.append(x)
                        v *= gr
                    if len(A) == 20:
                        obj = objective(A)
                        update_best(A, obj)

    # Family 5: Log-uniform spaced
    for lo in [1, 2, 3, 5]:
        for hi in [500, 1000, 2000, 5000, 9000, 10000]:
            if lo >= hi:
                continue
            A = []
            log_lo = np.log(lo)
            log_hi = np.log(hi)
            for i in range(20):
                x = int(round(np.exp(log_lo + (log_hi - log_lo) * i / 19)))
                x = max(1, min(10000, x))
                if x in A:
                    x += 1
                    if x > 10000:
                        x = 10000
                    while x in A:
                        x += 1
                        if x > 10000:
                            x = 1
                            while x in A:
                                x += 1
                A.append(x)
            if len(set(A)) == 20:
                obj = objective(A)
                update_best(A, obj)

    if time.time() > deadline - 20:
        return best_set

    # --- Phase 2: Local search with incremental objective ---

    def local_search(start_set, max_iters=200000):
        nonlocal best_set, best_obj
        A = list(start_set)
        A.sort()
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]

        A_arr = np.array(A, dtype=np.int64)
        idx = np.triu_indices(20)
        sums = A_arr[idx[0]] + A_arr[idx[1]]
        prods = A_arr[idx[0]] * A_arr[idx[1]]
        cur_s = len(np.unique(sums))
        cur_p = len(np.unique(prods))
        cur_obj = max(cur_s, cur_p)

        it = 0
        while it < max_iters:
            if time.time() > deadline - 5:
                return
            it += 1

            i = random.randrange(20)
            x = A[i]

            # Generate candidates
            cands = set()
            # Multiplicative neighbors
            for factor in [0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # Additive neighbors
            for delta in [-3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # Random
            cands.add(random.randrange(1, 10001))
            # Log-space random near x
            if x > 1:
                lx = random.uniform(0.3, 3.0)
                nv = int(round(x * lx))
                if 1 <= nv <= 10000:
                    cands.add(nv)

            cands = [c for c in cands if 1 <= c <= 10000 and c not in A]
            if not cands:
                continue

            nv = random.choice(cands)
            old = A[i]

            A[i] = nv
            A_arr[i] = nv
            sums = A_arr[idx[0]] + A_arr[idx[1]]
            prods = A_arr[idx[0]] * A_arr[idx[1]]
            new_s = len(np.unique(sums))
            new_p = len(np.unique(prods))
            new_obj = max(new_s, new_p)

            if new_obj <= cur_obj:
                cur_obj = new_obj
                cur_s = new_s
                cur_p = new_p
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
                A_arr[i] = old

    # Starting points for local search
    start_points = [
        list(range(1, 21)),
        [2**i for i in range(15)] + [2**15 + k for k in range(1, 6)],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 6144, 7168, 8192, 8704, 9216, 9728],
        [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 7290, 8019, 8748, 9477, 9700, 9800, 9900, 9950, 9975, 9990, 9999],
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 9350],
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 196, 256, 324, 400, 500, 625, 784, 961, 1200],
    ]

    # Add best from structured families
    start_points.append(list(best_set))

    # Random geometric starts
    for trial in range(30):
        r = random.uniform(1.05, 2.5)
        start_val = random.randint(1, 5)
        A = []
        v = float(start_val)
        for i in range(20):
            x = int(round(v))
            if x < 1:
                x = 1
            if x > 10000:
                break
            if x in A:
                x += 1
                if x > 10000:
                    break
            A.append(x)
            v *= r * random.uniform(0.9, 1.1)
        while len(A) < 20:
            x = random.randrange(1, 10001)
            if x not in A:
                A.append(x)
        A = A[:20]
        if len(set(A)) == 20:
            start_points.append(A)

    # Run local search from each start point
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        local_search(sp, max_iters=150000)

    # --- Phase 3: Fine polish on best solution ---
    if time.time() < deadline - 3:
        A = list(best_set)
        cur_obj = best_obj
        A_arr = np.array(A, dtype=np.int64)
        idx = np.triu_indices(20)

        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            i = random.randrange(20)
            x = A[i]
            cands = set()
            for delta in range(-3, 4):
                nv = x + delta
                if 1 <= nv <= 10000 and nv not in A:
                    cands.add(nv)
            for factor in [0.9, 0.95, 1.05, 1.1, 1.2, 0.8]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000 and nv not in A:
                    cands.add(nv)
            if not cands:
                continue
            nv = random.choice(list(cands))
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

    return best_set
