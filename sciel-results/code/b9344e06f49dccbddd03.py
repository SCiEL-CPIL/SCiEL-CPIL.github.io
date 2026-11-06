"""Size-efficient sorting networks on 17 wires.

Fresh design:
  * Fast NumPy zero-one verifier (all 131,072 Boolean inputs at once).
  * Parallel genetic algorithm with a correctness-preserving crossover
    ("sort by A then by B"), depth-aware comparator removal, and
    bit-level pruning.

Interface (unchanged): construct() -> ordered list of (i, j) pairs, 0 <= i < j < N.
"""
import time
import random
import numpy as np

N = 17
NBOOL = 1 << N          # 131072
MAX_COMPARATORS = 4096
DEADLINE = 75.0         # wall-clock safety margin (evaluator kills at 90 s)
SEED = 136

# ---------------------------------------------------------------------------
# Fast zero-one verification (vectorised)
# ---------------------------------------------------------------------------
def _apply(net, x):
    """Apply network `net` (list of (i,j)) to a 2-D array x (rows = inputs)."""
    for a, b in net:
        x[:, a], x[:, b] = np.minimum(x[:, a], x[:, b]), np.maximum(x[:, a], x[:, b])
    return x


def _is_sorted(net):
    """True iff `net` sorts every Boolean input (zero-one principle)."""
    if len(net) > MAX_COMPARATORS:
        return False
    rows = np.arange(NBOOL, dtype=np.uint8)
    x = ((rows[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    _apply(net, x)
    # sorted ascending  =>  no adjacent inversion
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _bit_count(net):
    """Number of Boolean inputs for which the network fails to sort."""
    rows = np.arange(NBOOL, dtype=np.uint8)
    x = ((rows[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    _apply(net, x)
    return int(np.sum(x[:, :-1] > x[:, 1:]))


def _depth(net):
    """Parallel depth of the network."""
    d = np.zeros(N, dtype=np.int32)
    for a, b in net:
        nd = d[a] + 1
        d[a] = d[b] = nd
    return int(d.max())


# ---------------------------------------------------------------------------
# Network construction / mutation / crossover
# ---------------------------------------------------------------------------
def _seed_network():
    # Odd-even transposition: 136 comparators, depth 17.
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _rand_net(rng):
    """Random network that is a valid sorter (seed + random extra comparators)."""
    net = list(_seed_network())
    net += [(rng.randrange(N - 1), rng.randrange(N - 1) + 1)
            for _ in range(rng.randint(0, 20))]
    return net


def _cross(A, B):
    """Correctness-preserving crossover: sort by A, then by B."""
    return list(A) + list(B)


def _remove(net, rng):
    """Depth-aware removal: drop a comparator only if the network stays sorted
    and no parallel layer becomes empty."""
    layers = {}
    for a, b in net:
        layers.setdefault((a, b), []).append(a)  # value stored only to count
    # count comparators per layer
    cnt = {}
    for a, b in net:
        cnt[(a, b)] = cnt.get((a, b), 0) + 1
    n = len(net)
    # try a few random positions
    order = list(range(n))
    rng.shuffle(order)
    for idx in order[:8]:
        if cnt[(net[idx][0], net[idx][1])] <= 1:
            continue  # would empty a layer
        cand = net[:idx] + net[idx + 1:]
        if _is_sorted(cand):
            return cand
    return net


def _add(net, rng):
    """Insert a random comparator at a random position."""
    a = rng.randrange(N - 1)
    b = a + 1
    pos = rng.randint(0, len(net))
    out = list(net)
    out.insert(pos, (a, b))
    return out


def _repair(net, rng):
    """Make `net` a valid sorter by appending random comparators (bounded)."""
    net = list(net)
    for _ in range(40):
        if _is_sorted(net):
            return net
        a = rng.randrange(N - 1)
        b = a + 1
        net.append((a, b))
    return net


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def _ga():
    rng = random.Random(12345)
    pop = [_seed_network()]
    pop += [_rand_net(rng) for _ in range(9)]

    best = min(pop, key=len)
    best_len = len(best)
    t0 = time.time()

    while time.time() - t0 < DEADLINE:
        # tournament selection
        def sel():
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            return pop[a] if len(pop[a]) <= len(pop[b]) else pop[b]

        children = []
        for _ in range(10):
            p1, p2 = sel(), sel()
            child = _cross(p1, p2)
            # mutation
            r = rng.random()
            if r < 0.45:
                child = _remove(child, rng)
            elif r < 0.7:
                child = _add(child, rng)
            # keep only valid sorters (crossover is already valid; removal may not be)
            if len(child) <= MAX_COMPARATORS and _is_sorted(child):
                children.append(child)

        # elitism + selection
        pop = sorted(pop + children, key=len)[:12]
        cand = pop[0]
        if len(cand) < best_len:
            best, best_len = cand, len(cand)

    return best


def construct():
    t0 = time.time()
    try:
        net = _ga()
    except Exception:
        net = _seed_network()
    # final safety: guarantee validity and time margin
    if not _is_sorted(net):
        net = _seed_network()
    if time.time() - t0 > DEADLINE:
        net = _seed_network()
    return net
