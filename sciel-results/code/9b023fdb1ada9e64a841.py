import numpy as np
import time
import random

N = 18

def _make_quick_inputs(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(n, N), dtype=np.uint8)

def _verify_full(network):
    n_inputs = 1 << N
    idx = np.arange(n_inputs, dtype=np.uint32)
    bits = np.array([(idx >> (N-1-p)) & 1 for p in range(N)], dtype=np.uint8)
    state = bits.T.copy()
    for (i, j) in network:
        si = state[:, i].copy()
        sj = state[:, j].copy()
        state[:, i] = si & sj
        state[:, j] = si | sj
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def _verify_quick(network, inputs):
    state = inputs.copy()
    for (i, j) in network:
        si = state[:, i].copy()
        sj = state[:, j].copy()
        state[:, i] = si & sj
        state[:, j] = si | sj
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def _odd_even_transposition():
    return [(i, i+1) for phase in range(N) for i in range(phase % 2, N-1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 80

    network = _odd_even_transposition()
    quick_inputs = _make_quick_inputs(2000, seed=42)

    # Greedy removal of redundant comparators
    for pass_num in range(20):
        if time.time() > deadline - 25:
            break
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 25:
                break
            candidate = network[:idx] + network[idx + 1:]
            if _verify_quick(candidate, quick_inputs):
                if _verify_full(candidate):
                    network = candidate
                    changed = True
        if not changed:
            break

    # Local search
    rng = random.Random(123)
    best = network
    best_len = len(network)

    while time.time() < deadline - 5:
        candidate = best[:]
        op = rng.choice(['swap', 'remove', 'add'])
        if op == 'swap' and len(candidate) >= 2:
            i = rng.randint(0, len(candidate) - 2)
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
        elif op == 'remove' and len(candidate) > 1:
            i = rng.randint(0, len(candidate) - 1)
            candidate = candidate[:i] + candidate[i + 1:]
        elif op == 'add':
            i = rng.randint(0, N - 1)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(candidate))
            candidate = candidate[:pos] + [(i, j)] + candidate[pos:]

        if _verify_quick(candidate, quick_inputs):
            if _verify_full(candidate):
                if len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)

    return best
