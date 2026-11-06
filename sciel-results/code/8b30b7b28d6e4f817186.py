"""Size-efficient sorting network on 18 wires.

Strategy: start from a small verified network, verify with a fast NumPy
verifier over all 2^18 Boolean inputs, then hill-climb (add / swap / delete
comparators) toward the known 77-comparator optimum. A hard deadline guard
returns the best valid result found.
"""
import time
import random
import numpy as np

N = 18
TOTAL = 1 << N
DEADLINE = time.time() + 78.0  # safety margin vs 90s wall-clock

# ---------------------------------------------------------------------------
# Fast verification (NumPy, vectorised over all 2^18 Boolean inputs)
# ---------------------------------------------------------------------------
def _apply(net, W):
    for i, j in net:
        a = W[:, i]
        b = W[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        W[:, i] = mn
        W[:, j] = mx
    return W

def _is_sorted(W):
    return np.all(W[:, 1:] >= W[:, :-1])

def _verify(net):
    W = (np.arange(TOTAL, dtype=np.uint8)[:, None]
         >> np.arange(N, dtype=np.uint8)[None, :]) & 1
    return _is_sorted(_apply(net, W))

# ---------------------------------------------------------------------------
# Initial small verified network (odd-even transposition, 153 comparators)
# ---------------------------------------------------------------------------
def _seed_network():
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    return net

# ---------------------------------------------------------------------------
# Hill-climbing search
# ---------------------------------------------------------------------------
def _best_valid():
    net = _seed_network()
    if not _verify(net):
        raise RuntimeError("seed network is not a valid sorting network")
    return net

def _add_move(rng, net):
    i = rng.randrange(N)
    j = rng.randrange(i + 1, N)
    pos = rng.randrange(len(net) + 1)
    cand = net[:pos] + [(i, j)] + net[pos:]
    return cand

def _swap_move(rng, net):
    if len(net) < 2:
        return None
    a = rng.randrange(len(net))
    b = rng.randrange(len(net))
    if a == b:
        return None
    cand = net[:]
    cand[a], cand[b] = net[b], net[a]
    return cand

def _del_move(rng, net):
    if len(net) == 0:
        return None
    k = rng.randrange(len(net))
    cand = net[:k] + net[k + 1:]
    return cand

def construct():
    rng = random.Random(12345)
    best = _best_valid()
    best_len = len(best)
    cur = best
    cur_len = best_len

    while time.time() < DEADLINE:
        r = rng.random()
        if r < 0.55:
            cand = _add_move(rng, cur)
        elif r < 0.80:
            cand = _swap_move(rng, cur)
            if cand is None:
                continue
        else:
            cand = _del_move(rng, cur)
            if cand is None:
                continue

        # Prefer shrinking; allow occasional growth to escape local minima.
        if len(cand) <= cur_len + 2 and _verify(cand):
            cur = cand
            cur_len = len(cand)
            if cur_len < best_len:
                best = cur
                best_len = cur_len
                if best_len <= 77:
                    return best
        else:
            # rejected: keep current
            pass
        if time.time() > DEADLINE - 1.0:
            break

    return best
