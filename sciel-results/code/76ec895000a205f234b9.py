"""Size-efficient sorting network on 17 wires via time-boxed local search."""
import time
import random

N = 17
MASKS = 1 << N  # 131072


def _apply(network, m):
    a = m
    for i, j in network:
        bi = (a >> i) & 1
        bj = (a >> j) & 1
        if bi > bj:
            a ^= (1 << i) | (1 << j)
    return a


def _is_valid(network):
    for m in range(MASKS):
        r = _apply(network, m)
        if r != (r & -r) - 1:
            return False
    return True


def _count(network):
    return len(network)


def _seed_network():
    # Odd-even transposition (valid baseline)
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return net


def _remove_pass(net):
    """Try removing each comparator; keep removals that preserve validity."""
    improved = True
    while improved:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            cand = net[:idx] + net[idx + 1:]
            if _is_valid(cand):
                net = cand
                improved = True
    return net


def _swap_pass(net, rng):
    """Try swapping adjacent comparators; keep swaps that preserve validity."""
    improved = True
    while improved:
        improved = False
        for idx in range(len(net) - 1):
            cand = net[:idx] + [net[idx + 1], net[idx]] + net[idx + 2:]
            if _is_valid(cand):
                net = cand
                improved = True
    return net


def _random_insert_pass(net, rng, deadline):
    """Insert random comparators then clean; accept if smaller."""
    best = net
    while time.time() < deadline:
        trial = list(best)
        # insert 1-3 random comparators
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pos = rng.randrange(len(trial) + 1)
            trial.insert(pos, (i, j))
        # clean by removal
        cleaned = _remove_pass(trial)
        if len(cleaned) < len(best):
            best = cleaned
    return best


def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s kill
    rng = random.Random(12345)

    net = _seed_network()
    # Initial reduction pass (deterministic, cheap enough)
    net = _remove_pass(net)

    # Greedy structural improvement within budget
    while time.time() < deadline - 5.0:
        before = len(net)
        net = _swap_pass(net, rng)
        net = _remove_pass(net)
        if len(net) < before:
            before = len(net)
        # occasional insert-explore
        if time.time() < deadline - 5.0:
            net = _random_insert_pass(net, rng, deadline - 5.0)
            net = _remove_pass(net)

    # Final safety: ensure validity; if somehow invalid, fall back to seed
    if time.time() < deadline:
        if not _is_valid(net):
            net = _seed_network()

    return net
