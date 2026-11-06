"""Size-efficient sorting network on 17 wires.
Strategy: build a candidate network (odd-even transposition),
then minimize via systematic comparator deletion, certified over all 131,072
Boolean inputs (zero-one principle). Explicit deadline guard returns best found.
"""
import time
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = time.time() + 82.0  # safety margin before 90s kill


def _build_inputs():
    bits = np.arange(N, dtype=np.uint8)
    rows = np.arange(NUM_INPUTS, dtype=np.uint32)[:, None]
    return ((rows >> bits) & 1).astype(np.uint8)


_INPUTS = _build_inputs()


def _apply(comps, inp):
    for i, j in comps:
        a = inp[:, i]
        b = inp[:, j]
        m = np.minimum(a, b)
        M = np.maximum(a, b)
        inp[:, i] = m
        inp[:, j] = M
    return inp


def _is_sorted(comps):
    if len(comps) > 4096:
        return False
    inp = _INPUTS.copy()
    _apply(comps, inp)
    d = np.diff(inp, axis=1)
    return not np.any(d < 0)


def _make_transposition_network():
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _minimize(comps, deadline):
    best = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        for idx in range(n):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx + 1:]
            if _is_sorted(candidate):
                best = candidate
                improved = True
                n = len(best)
                break
    return best


def construct():
    start = time.time()
    comps = _make_transposition_network()
    if not _is_sorted(comps):
        return comps
    best = _minimize(comps, DEADLINE)
    if not _is_sorted(best):
        return comps
    return best
