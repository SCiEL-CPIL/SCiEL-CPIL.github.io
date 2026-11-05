import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 52.0  # safety margin under 60s

    best_set = None
    best_obj = float('inf')

    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]

    # --- Family 1: Lattice-based seeds (2^i * 3^j) ---
    # Products stay in multiplicative subgroup, bounding product set size
    lattice_patterns = [
        [(i, 0) for i in range(20)],
        [(0, i) for i in range(20)],
        [(i, i) for i in range(20)],
        [(i, 20-i) for i in range(20)],
        [(i, i//2) for i in range(20)],
        [(i//2, i) for i in range(20)],
        [(i, (i*3)%20) for i in range(20)],
        [(i%5, i//5) for i in range(20)],
        [(i//5, i%5) for i in range(20)],
        [(i%4, i//4) for i in range(20)],
        [(i//4, i%4) for i in range(20)],
        [(i%3, i//3) for i in range(20)],
        [(i//3, i%3) for i in range(20)],
        [(i%6, i//6) for i in range(20)],
        [(i//6, i%6) for i in range(20)],
        [(i%7, i//7) for i in range(20)],
        [(i//7, i%7) for i in range(20)],
        [(i%10, i//10) for i in range(20)],
        [(i//10, i%10) for i in range(20)],
    ]

    for pattern in lattice_patterns:
        A = []
        for ii, jj in pattern:
            if 0 <= ii < 20 and 0 <= jj < 20:
                val = (2 ** ii) * (3 ** jj)
                if 1 <= val <= 10000:
                    A.append(val)
        seen = set()
        uniq = []
        for x in A:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        if len(uniq) == 20:
            try_candidate(uniq)

    # Also scan rectangular grids in (i,j) space
    for i_start in range(0, 12):
        for j_start in range(0, 12):
            for i_w in range(2, 10):
                for j_w in range(2, 10):
                    if i_start + i_w > 20 or j_start + j_w > 20:
                        continue
                    A = []
                    for ii in range(i_start, min(i_start + i_w, 20)):
                        for jj in range(j_start, min(j_start + j_w, 20)):
                            val = (2 ** ii) * (3 ** jj)
                            if 1 <= val <= 10000:
                                A.append(val)
                    seen = set()
                    uniq = []
                    for x in A:
                        if x not in seen:
                            seen.add(x)
                            uniq.append(x)
                    if len(uniq) == 20:
                        try_candidate(uniq)
                        break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # --- Family 2: Geometric sequences ---
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        for start_val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            A = []
            v = start_val
            for i in range(20):
                A.append(v)
                v *= base
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)

    # --- Family 3: Geometric with ratio close to 1 ---
    for ratio in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
        for start_val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
            A = []
            v = float(start_val)
            for i in range(20):
                val = int(round(v))
                if val < 1:
                    val = 1
                A.append(val)
                v *= ratio
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)

    # --- Family 4: Powers of 2 times small odd numbers ---
    for offset in [1, 2, 3, 5, 7]:
        for shift in [0, 1, 2, 3]:
            A = []
            for i in range(20):
                val = (2**i) * (1 + (i + shift) * offset)
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)

    # --- Family 5: Sparse geometric ---
    for base in [2, 3, 4, 5]:
        for step in [1, 2, 3]:
            A = [base**(i*step) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)

    # --- Family 6: Mixed - first half geometric, second half perturbed ---
    for base in [2, 3, 4]:
        A = [base**i for i in range(10)]
        last = base**9
        for i in range(10):
            A.append(last * (i + 2))
        seen = set()
        uniq = []
        for x in A:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        if len(uniq) == 20:
            try_candidate(uniq)

    # --- Family 7: Fibonacci-like sequences ---
    for a0 in [1, 2, 3, 4, 5]:
        for a1 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            A = [a0, a1]
            for i in range(18):
                nxt = A[-1] + A[-2]
                if nxt > 10000:
                    break
                A.append(nxt)
            if len(A) == 20 and len(set(A)) == 20 and all(1 <= x <= 10000 for x in A):
                try_candidate(A)

    # --- Family 8: Arithmetic-geometric hybrid ---
    for base in [2, 3, 4]:
        for offset in [1, 2, 3, 5, 10]:
            A = []
            for i in range(20):
                val = base**i + offset * (i + 1)
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)

    # --- Local search with simulated annealing style acceptance ---
    random.seed(12345)

    def local_search(start_set, max_iters=30000):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        it = 0
        no_improve = 0
        temp = 10.0
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            # Neighbors in log space
            for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # Additive neighbors
            for delta in [-10, -5, -2, -1, 1, 2, 5, 10, 50, 100]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # Random
            cands.add(random.randrange(1, 10001))
            cands.discard(x)
            cands = [c for c in cands if c not in A]
            if not cands:
                continue
            nv = random.choice(cands)
            old = A[i]
            A[i] = nv
            new_obj = objective(A)
            delta_obj = new_obj - cur_obj
            if new_obj <= cur_obj:
                cur_obj = new_obj
                no_improve = 0
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < np.exp(-delta_obj / temp):
                    cur_obj = new_obj
                    no_improve = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                else:
                    A[i] = old
                    no_improve += 1
            temp *= 0.9995
            if no_improve > 5000:
                break

    # Local search from best candidate
    local_search(best_set, max_iters=40000)

    # Also try local search from a few other structured starts
    other_starts = [
        list(range(1, 21)),
    ]
    for base in [2, 3]:
        A = []
        v = 1
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000 and len(set(A)) == 20:
            other_starts.append(A)

    # Add lattice-based starts for local search
    for pattern in lattice_patterns[:6]:
        A = []
        for ii, jj in pattern:
            if 0 <= ii < 20 and 0 <= jj < 20:
                val = (2 ** ii) * (3 ** jj)
                if 1 <= val <= 10000:
                    A.append(val)
        seen = set()
        uniq = []
        for x in A:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        if len(uniq) == 20:
            other_starts.append(uniq)

    for sp in other_starts:
        if time.time() > deadline - 5:
            break
        if sp == best_set:
            continue
        local_search(sp, max_iters=15000)

    # Final polish: try small perturbations of the best solution
    if time.time() < deadline - 5:
        A = list(best_set)
        cur_obj = objective(A)
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            i = random.randrange(20)
            x = A[i]
            for delta in [-3, -2, -1, 1, 2, 3]:
                nv = x + delta
                if 1 <= nv <= 10000 and nv not in A:
                    A[i] = nv
                    new_obj = objective(A)
                    if new_obj < cur_obj:
                        cur_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_set = A[:]
                    else:
                        A[i] = x
                    break

    return best_set
