import time
import numpy as np
import random

N = 18
ALL_COUNT = 1 << N
KILL = 90.0
DEADLINE = KILL - 8.0  # safety margin

def _init_test_vectors():
    idx = np.arange(ALL_COUNT, dtype=np.uint32)
    bits = (idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    return bits.astype(np.uint8)

def _check_sorted_fast(vals):
    return np.all(vals[:, :-1] <= vals[:, 1:])

def _simulate_net(net, test_vectors):
    vals = test_vectors.copy()
    for a, b in net:
        va = vals[:, a]
        vb = vals[:, b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[:, a] = lo
        vals[:, b] = hi
    return vals

def _is_sorted(net, test_vectors):
    if not net:
        return N <= 1
    vals = _simulate_net(net, test_vectors)
    return _check_sorted_fast(vals)

def _needed_mask(net, test_vectors):
    """
    Simulate once; for each comparator return a boolean: True if that
    comparator actually changed at least one wire value on some input.
    A comparator with mask False is redundant given the rest of the network
    (it never affects the final output), so it is safe to remove.
    """
    vals = test_vectors.copy()
    needed = np.zeros(len(net), dtype=bool)
    for k, (a, b) in enumerate(net):
        va = vals[:, a]
        vb = vals[:, b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        # changed if va != lo OR vb != hi  (equivalently va > vb)
        if np.any(va > vb):
            needed[k] = True
        vals[:, a] = lo
        vals[:, b] = hi
    return needed

def _prune_pass(net, test_vectors):
    """Remove all provably-redundant comparators in one pass. Returns (new_net, removed_count)."""
    if not net:
        return net, 0
    needed = _needed_mask(net, test_vectors)
    if needed.all():
        return net, 0
    keep = [i for i in range(len(net)) if needed[i]]
    new_net = [net[i] for i in keep]
    return new_net, len(net) - len(keep)

def _greedy_prune(net, test_vectors, deadline):
    """
    Loop: batch-remove redundant comparators (fast), then try single removals
    in random order to escape plateaus. Stop when no progress or deadline.
    """
    current = net[:]
    while time.time() < deadline:
        # Fast batch removal
        current, removed = _prune_pass(current, test_vectors)
        if removed == 0:
            # Try single random removals to break the plateau
            if len(current) == 0:
                break
            indices = list(range(len(current)))
            random.shuffle(indices)
            progress = False
            for idx in indices:
                if time.time() > deadline:
                    break
                if idx >= len(current):
                    continue
                candidate = current[:idx] + current[idx+1:]
                if _is_sorted(candidate, test_vectors):
                    current = candidate
                    progress = True
                    break
            if not progress:
                break
    return current

def _bitonic_merge(n, direction):
    """Batcher bitonic merge on wires 0..n-1 (input assumed bitonic)."""
    if n <= 1:
        return []
    half = n // 2
    comps = []
    comps += _bitonic_merge(half, direction)
    comps += _bitonic_merge(half, 1 - direction)
    for i in range(half):
        comps.append((i, i + half))
    return comps

def _bitonic_sort(n, direction=0):
    if n <= 1:
        return []
    half = n // 2
    left = _bitonic_sort(half, direction)
    right = _bitonic_sort(n - half, 1 - direction)
    # shift right half comparators by 'half'
    right_shifted = [(i + half, j + half) for (i, j) in right]
    return left + right_shifted + _bitonic_merge(n, direction)

def _generate_bitonic(n):
    return _bitonic_sort(n, 0)

def _generate_odd_even(n):
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net

def _generate_akser(n):
    """
    Simple AKS-style: use bitonic as a strong base (AKS is complex to build
    by hand; bitonic gives a valid, compact-ish starting point).
    """
    return _generate_bitonic(n)

def _random_perturb(net, k):
    out = net[:]
    for _ in range(k):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        out.append((i, j))
    return out

def construct():
    start_time = time.time()
    test_vectors = _init_test_vectors()

    # Build candidate starting networks (all valid sorters for N=18).
    base_candidates = []
    b = _generate_bitonic(N)
    if _is_sorted(b, test_vectors):
        base_candidates.append(b)
    o = _generate_odd_even(N)
    if _is_sorted(o, test_vectors):
        base_candidates.append(o)

    # Pick the smallest valid base to start from.
    base_candidates = [x for x in base_candidates if x]
    if not base_candidates:
        # Fallback: brute force adjacent (always valid)
        return _generate_odd_even(N)
    base_candidates.sort(key=len)
    best_net = base_candidates[0][:]
    best_len = len(best_net)

    # Prune the best base first.
    dl = start_time + 20
    pruned = _greedy_prune(best_net, test_vectors, dl)
    if len(pruned) < best_len:
        best_net, best_len = pruned, len(pruned)

    # Also prune the other base(s) and keep the best.
    for cand in base_candidates[1:]:
        if time.time() > DEADLINE - 5:
            break
        dl = min(start_time + 35, DEADLINE - 5)
        pruned = _greedy_prune(cand, test_vectors, dl)
        if len(pruned) < best_len:
            best_net, best_len = pruned, len(pruned)

    # Multi-start with random perturbations + pruning.
    deadline2 = start_time + (KILL - 12)
    while time.time() < deadline2:
        # perturb best net with a few random comparators, then prune hard
        k = random.randint(1, 4)
        pert = _random_perturb(best_net, k)
        dl = min(time.time() + 8, deadline2)
        pruned = _greedy_prune(pert, test_vectors, dl)
        if len(pruned) < best_len:
            best_net, best_len = pruned, len(pruned)

    # Final careful single-removal polish in deterministic order.
    deadline3 = DEADLINE
    current = best_net[:]
    changed = True
    while changed and time.time() < deadline3:
        changed = False
        # batch redundant removal first
        current, _ = _prune_pass(current, test_vectors)
        for idx in range(len(current)):
            if time.time() > deadline3:
                break
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted(candidate, test_vectors):
                current = candidate
                changed = True
                break
    if len(current) < best_len:
        best_net, best_len = current, len(current)

    return best_net
