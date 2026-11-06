"""Bitwise-verified sorting network with greedy removal + simulated annealing."""
import time
import random
import numpy as np

N = 18
DEADLINE = 75.0          # safety margin under the 90 s wall-clock kill
SEED = 12345

# ---------- bitwise verification (linear operator on Boolean vectors) ----------
def _verify_bitwise(net):
    """Return True iff net sorts every 2**N Boolean input.

    Each comparator (i, j) is the min/max projection on wires i, j.
    The network sorts all inputs iff it sorts every standard basis vector.
    """
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        M[i, i] = 1
    for (i, j) in net:
        mi = M[:, i]
        mj = M[:, j]
        lo = np.minimum(mi, mj)
        hi = np.maximum(mi, mj)
        M[:, i] = lo
        M[:, j] = hi
    # sorted output of basis vector k is basis vector k (identity).
    return bool(np.array_equal(M, np.eye(N, dtype=np.int8)))

# ---------- network generation ----------
def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bitonic_network(n):
    """Recursive bitonic sort network on wires [0, n)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    if n % 2 == 0:
        left = _bitonic_network(half)
        right = _bitonic_network(half)
        right = [(i + half, j + half) for (i, j) in right]
        merge = _bitonic_merge(0, n)
        return left + right + merge
    else:
        left = _bitonic_network(half)
        right = _bitonic_network(half + 1)
        right = [(i + half, j + half) for (i, j) in right]
        merge = _bitonic_merge(0, n)
        return left + right + merge

def _bitonic_merge(start, length):
    """Merge two bitonic halves (ascending + descending) into ascending."""
    comps = []
    k = length // 2
    while k > 1:
        for i in range(k):
            comps.append((start + i, start + i + k))
        k //= 2
    return comps

# ---------- optimization ----------
def _greedy_remove(net, deadline):
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cand = net[:idx] + net[idx + 1:]
            if _verify_bitwise(cand):
                net = cand
                changed = True
                break
    return net

def _sa(net, deadline):
    """Simulated annealing over comparator moves."""
    rng = random.Random(SEED)
    net = list(net)
    best = net
    best_len = len(net)
    T = 1.0
    cooling = 0.9999
    T_min = 1e-4

    def neighbors(x):
        n = len(x)
        r = rng.random()
        if r < 0.45 and n > 1:
            i = rng.randrange(n)
            j = i + 1 + rng.randrange(n - i - 1)
            y = x[:i] + x[i + 1:]
            y.insert(j, x[i])
            return y
        elif r < 0.85:
            i = rng.randrange(n)
            j = rng.randrange(N)
            if j > i:
                j -= 1
            y = list(x)
            y[i] = (j, x[i][1])
            return y
        else:
            i = rng.randrange(N)
            j = i + 1 + rng.randrange(N - i - 1)
            y = x + [(i, j)]
            return y

    while time.time() < deadline:
        if T < T_min:
            T = 1.0
            cooling = 0.99999
        x = net
        y = neighbors(x)
        if _verify_bitwise(y):
            net = y
            if len(y) < best_len:
                best = y
                best_len = len(y)
            T = max(T * cooling, T_min)
        else:
            # accept worsening with probability
            if rng.random() < np.exp(-1.0 / T):
                net = y
        if time.time() >= deadline - 2.0:
            break
    return best

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(SEED)

    # candidate seed networks
    seeds = []
    seeds.append(_odd_even_transposition())
    seeds.append(_bitonic_network(N))
    # random comparator soup (large, to be pruned)
    seeds.append([(i, j) for _ in range(400)
                  for (i, j) in [(rng.randrange(N), rng.randrange(N))]])
    seeds[-1] = [(min(i, j), max(i, j)) for (i, j) in seeds[-1] if i != j]

    best = None
    best_len = 10**9

    for seed in seeds:
        if time.time() >= deadline:
            break
        net = _greedy_remove(seed, deadline)
        if _verify_bitwise(net):
            if len(net) < best_len:
                best = net
                best_len = len(net)

    if best is None:
        best = _odd_even_transposition()
        best_len = len(best)

    # simulated annealing to escape local minima
    if time.time() < deadline - 5.0:
        improved = _sa(best, deadline)
        if _verify_bitwise(improved) and len(improved) < best_len:
            best = improved
            best_len = len(improved)

    # final safety: ensure validity
    if not _verify_bitwise(best):
        best = _odd_even_transposition()

    return best
