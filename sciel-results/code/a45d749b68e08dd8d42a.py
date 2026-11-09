import time
import random

N = 17
TOTAL_MASKS = 1 << N  # 131072

# Precompute sorted mask values: 0, 1, 3, 7, ..., 131071
_SORTED_VALS = frozenset((1 << k) - 1 for k in range(N + 1))


def _verify_bigint(net):
    """Verify network using Python big-int bitwise ops. Returns True if valid."""
    if len(net) == 0:
        return False  # Only valid if N<=1, but N=17 so empty net is invalid
    x = (1 << N) - 1  # all-ones mask: bit i = 1 for wire i
    for i, j in net:
        # Extract bits i and j from x
        ai = (x >> i) & 1
        aj = (x >> j) & 1
        if ai == 1 and aj == 0:
            # Swap: clear bit i, set bit j
            x = (x & ~(1 << i)) | (1 << j)
    # Check if x is a sorted mask (all 1s in lowest positions)
    return x in _SORTED_VALS


def _verify_count_unsorted(net):
    """Return count of unsorted masks. Uses big-int for each mask."""
    if len(net) == 0:
        unsorted = 0
        for m in range(TOTAL_MASKS):
            if m not in _SORTED_VALS:
                unsorted += 1
        return unsorted
    unsorted = 0
    for m in range(TOTAL_MASKS):
        x = m
        for i, j in net:
            ai = (x >> i) & 1
            aj = (x >> j) & 1
            if ai == 1 and aj == 0:
                x = (x & ~(1 << i)) | (1 << j)
        if x != m:
            unsorted += 1
    return unsorted


def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def _greedy_remove(net, deadline):
    """Aggressively remove redundant comparators. Returns optimized net."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from the end (often redundant last)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            trial = net[:idx] + net[idx + 1:]
            if _verify_bigint(trial):
                net = trial
                improved = True
                break
        # Also try from the beginning
        if not improved:
            for idx in range(len(net)):
                if time.time() > deadline:
                    return net
                trial = net[:idx] + net[idx + 1:]
                if _verify_bigint(trial):
                    net = trial
                    improved = True
                    break
    return net


def _randomized_removal(net, deadline, n_restarts=5):
    """Try randomized removal orders to escape local minima."""
    best = net[:]
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        candidate = best[:]
        random.shuffle(candidate)
        # After shuffling, re-greedy-remove
        candidate = _greedy_remove(candidate, deadline)
        if len(candidate) < len(best):
            best = candidate
    return best


def _two_for_one_exchange(net, deadline):
    """Try replacing one comparator with two others to enable further removal."""
    best = net[:]
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # For each comparator in the network, try removing it and adding one or two others
    for idx in range(len(best)):
        if time.time() > deadline:
            break
        without = best[:idx] + best[idx + 1:]
        # Try adding one new comparator
        for comp in all_comps:
            if time.time() > deadline:
                break
            if comp in without:
                continue
            trial = without + [comp]
            if _verify_bigint(trial) and len(trial) < len(best):
                best = trial
                break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Start with seed
    net = _build_seed()
    
    # Verify seed is valid (sanity check)
    if not _verify_bigint(net):
        # Shouldn't happen, but fallback
        return net
    
    best_net = net[:]
    best_len = len(net)
    
    # Phase 1: Greedy removal from seed
    net = _greedy_remove(net, deadline)
    if len(net) < best_len:
        best_net = net[:]
        best_len = len(net)
    
    # Phase 2: Randomized restarts with re-removal
    if time.time() < deadline:
        net = _randomized_removal(best_net, deadline, n_restarts=10)
        if len(net) < best_len:
            best_net = net[:]
            best_len = len(net)
    
    # Phase 3: Try 2-for-1 exchanges if time permits
    if time.time() < deadline - 5:
        net = _two_for_one_exchange(best_net, deadline)
        if len(net) < best_len:
            best_net = net[:]
            best_len = len(net)
    
    # Phase 4: Final greedy pass on the best found
    if time.time() < deadline:
        net = _greedy_remove(best_net, deadline)
        if len(net) < best_len:
            best_net = net[:]
            best_len = len(net)
    
    return best_net
