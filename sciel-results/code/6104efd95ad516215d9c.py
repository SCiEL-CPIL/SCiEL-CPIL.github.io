import numpy as np
import time
import math
import random


def _compute_score(U):
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sumset = np.unique(np.add.outer(U_arr, U_arr))
    sum_size = len(sumset)
    diffset = np.unique(np.subtract.outer(U_arr, U_arr))
    diff_size = len(diffset)
    if sum_size == 0:
        return 0.0
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1 + math.log(ratio) / denom


def _initial_greedy_set(target_size, max_val):
    random.seed(42)
    selected = [0]
    remaining = target_size - 1
    for i in range(1, min(5, remaining + 1)):
        selected.append(i)
    remaining -= min(4, remaining)
    if remaining > 0:
        step = max_val // (remaining + 1)
        for i in range(remaining):
            val = (i + 1) * step + random.randint(0, max(1, step // 10))
            if val < max_val and val not in selected:
                selected.append(val)
    return sorted(set(selected))


def _local_search(U, deadline, max_iters=50000):
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    best_score = _compute_score(U)
    best_U = U[:]
    n = len(U)
    maxU = U[-1]
    search_max = int(maxU * 1.5) + 100
    if search_max < 100:
        search_max = 100
    rng = random.Random(123)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        op = rng.random()
        if op < 0.4 and n < 8000:
            new_val = rng.randint(1, search_max)
            if new_val in U:
                continue
            new_U = U + [new_val]
            new_U = sorted(new_U)
            new_score = _compute_score(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                U = new_U
                n = len(U)
                maxU = U[-1]
                search_max = int(maxU * 1.5) + 100
        elif op < 0.7 and n > 2:
            idx = rng.randint(0, n - 1)
            if U[idx] == 0:
                continue
            new_U = U[:idx] + U[idx + 1:]
            new_score = _compute_score(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                U = new_U
                n = len(U)
                maxU = U[-1]
                search_max = int(maxU * 1.5) + 100
        else:
            if n < 2:
                continue
            idx = rng.randint(0, n - 1)
            if U[idx] == 0:
                continue
            new_val = rng.randint(1, search_max)
            if new_val in U:
                continue
            new_U = U[:idx] + U[idx + 1:] + [new_val]
            new_U = sorted(new_U)
            new_score = _compute_score(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U
                U = new_U
                n = len(U)
                maxU = U[-1]
                search_max = int(maxU * 1.5) + 100
    return best_U, best_score


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    target_size = 2000
    max_val = 50000
    U = _initial_greedy_set(target_size, max_val)

    if time.time() < deadline - 10:
        U, score = _local_search(U, deadline, max_iters=20000)

    if time.time() < deadline - 30:
        m = 45
        U2 = set()
        for a in range(m):
            for b in range(m):
                U2.add(a * m + b)
        U2 = sorted(U2)
        if 0 not in U2:
            U2 = sorted(set(U2) | {0})
        score2 = _compute_score(U2)
        if score2 > score:
            if time.time() < deadline - 10:
                U2, score2 = _local_search(U2, deadline, max_iters=10000)
            if score2 > score:
                U = U2

    if 0 not in U:
        U = sorted(set(U) | {0})
    return U
# EVOLVE-BLOCK-END
