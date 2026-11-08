import time
import math
import random
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    # Fast numpy-based sumset/difference-set sizes
    U_arr = np.array(U)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def local_optimize(U, deadline, max_iters=300):
    """Local search: add / remove / adjust elements to improve score."""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score

    for _ in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        n = len(U)
        maxU = U[-1] if U else 1

        # 1) Add a new element (sparse candidates to keep maxU growth favorable)
        if n < 18:
            candidates = set()
            for u in U:
                candidates.add(u + 1)
                candidates.add(max(1, u - 1))
                candidates.add(u * 2)
                candidates.add(u + 3)
                candidates.add(u + 7)
            for factor in [2, 3, 5, 8, 13]:
                candidates.add(maxU * factor)
                candidates.add(maxU * factor + 1)
                candidates.add(maxU * factor - 1)
            # A few random large jumps to escape local optima
            for _ in range(5):
                candidates.add(random.randint(1, maxU * 10 + 50))

            for cand in sorted(candidates):
                if cand < 0 or cand in U:
                    continue
                new_U = sorted(U + [cand])
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 2) Remove an element (keep at least 2, keep 0)
        if n > 2:
            for i in range(1, n):
                new_U = U[:i] + U[i+1:]
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 3) Adjust an element (keep 0 fixed)
        if n > 1:
            for i in range(1, n):
                if improved:
                    break
                for delta in [-3, -2, -1, 1, 2, 3, 5, 7]:
                    new_val = U[i] + delta
                    if new_val < 0 or new_val in U:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        if not improved:
            break

    return best_U, best_score

def construct():
    start = time.time()
    deadline = start + 135  # safety margin under 150s

    best_score = 0
    best_U = [0, 1, 3]
    s = compute_score([0, 1, 3])
    best_score = s
    best_U = [0, 1, 3]

    # ---- Phase 1: structured families ----
    families = []

    # Powers of 2
    for n in range(3, 18):
        families.append([0] + [2**i for i in range(1, n)])

    # Mersenne numbers (2^i - 1)
    for n in range(3, 18):
        families.append([0] + [2**i - 1 for i in range(1, n)])

    # Twin blocks {0..k-1, M..M+k-1}
    for k in [2, 3, 4, 5, 6]:
        for M in [k+1, 2*k, 5*k, 10*k, 20*k, 50*k, 100*k, 500*k]:
            families.append(list(range(k)) + list(range(M, M+k)))

    # Geometric progressions {0, 1, r, r^2, ...}
    for r in [2, 3, 4, 5, 7, 10]:
        for k in [3, 4, 5, 6, 7, 8]:
            U = [0, 1]
            val = r
            for i in range(k-2):
                U.append(val)
                val *= r
            families.append(U)

    # Sparse {0, 1, M, M+1} and {0, 1, 2, M, M+1}
    for M in [2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        families.append([0, 1, M, M+1])
        families.append([0, 1, 2, M, M+1])

    # {0, 1, M, M+2, 2M}
    for M in [3, 5, 10, 50, 100, 500]:
        families.append([0, 1, M, M+2, 2*M])

    # Arithmetic {0,1} + multiples
    for k in [2, 3, 4, 5]:
        for M in [2, 3, 5, 7, 10, 20, 50]:
            U = [0, 1] + [i*M for i in range(1, k+1)]
            families.append(sorted(set(U)))

    for U in families:
        if time.time() > deadline - 30:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # ---- Phase 2: exhaustive small search for a strong anchor ----
    if time.time() < deadline - 25:
        for a in range(1, 12):
            for b in range(a+1, 25):
                for c in range(b+1, 45):
                    if time.time() > deadline - 25:
                        break
                    U = [0, a, b, c]
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]

    # ---- Phase 3: random restarts (inspiration 1's breadth) ----
    random.seed(42)
    diverse = []
    if time.time() < deadline - 15:
        for _ in range(4000):
            if time.time() > deadline - 15:
                break
            n = random.randint(3, 14)
            # Mix small dense and large sparse ranges
            if random.random() < 0.5:
                max_val = random.randint(10, 200)
            else:
                max_val = random.randint(50, 2000)
            U = set([0])
            for _ in range(n - 1):
                U.add(random.randint(1, max_val))
            U = sorted(U)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            # Keep a few high-scoring diverse starts for local search
            if len(diverse) < 12 and s > best_score - 0.01:
                diverse.append(U[:])

    # ---- Phase 4: local search from best + diverse anchors ----
    if time.time() < deadline - 5:
        opt_U, opt_score = local_optimize(best_U, deadline, max_iters=400)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # Local search from a handful of other promising/diverse candidates
    seen = set()
    for cand in diverse:
        if time.time() > deadline - 3:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        if key == tuple(best_U):
            continue
        opt_U, opt_score = local_optimize(cand, deadline, max_iters=150)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    return best_U
