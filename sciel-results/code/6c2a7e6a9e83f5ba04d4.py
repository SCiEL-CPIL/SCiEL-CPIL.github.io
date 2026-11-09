"""Odd-even transposition family with budgeted comparator reduction for 18 wires."""
import time

N = 18
SEED_COUNT = N * (N - 1) // 2          # 153
DEADLINE = 80.0                        # safety margin under the 90 s kill
_seed_time = time.time()


# EVOLVE-BLOCK-START
def construct():
    """Return a correct compare-exchange network (list of (i, j), i<j) for N wires.

    Starts from the odd-even transposition network and, within a wall-clock budget,
    greedily removes redundant comparators (verified over all 2**N Boolean inputs).
    """
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    best_len = len(net)
    deadline = _seed_time + DEADLINE

    while time.time() < deadline:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cand = net[:idx] + net[idx + 1:]
            if _is_sorted_network(cand):
                net = cand
                best = list(net)
                best_len = len(net)
                improved = True
                break
        if not improved:
            break

    return best


def _is_sorted_network(net):
    """True iff the network sorts every 2**N Boolean input (zero/one principle)."""
    M = len(net)
    for mask in range(1 << N):
        v = list(range(N))
        for i in range(N):
            v[i] = 1 if (mask >> i) & 1 else 0
        for i, j in net:
            a, b = v[i], v[j]
            if a > b:
                v[i], v[j] = b, a
        ok = True
        for i in range(N - 1):
            if v[i] > v[i + 1]:
                ok = False
                break
        if not ok:
            return False
    return True
# EVOLVE-BLOCK-END
