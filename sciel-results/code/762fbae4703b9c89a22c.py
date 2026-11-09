"""Size-efficient sorting network on 18 wires.
Uses a known-good network with vectorized verification and local-search pruning.
Deadline-guarded so it always returns within the wall-clock budget.
"""
import time
import random
import numpy as np

N = 18

_START_TIME = None
_DEADLINE_MARGIN = 5.0
_DEADLINE = 85.0


def _time_left():
    return _DEADLINE - time.time()


def _apply_network_vec(net, bits_array):
    """Apply a list of comparators to a 2D array of bit vectors (num_patterns x N).
    bits_array: shape (num_patterns, N) with values in {0,1}
    Returns the sorted result array.
    """
    b = bits_array.copy()
    for i, j in net:
        # For each pattern, compare wire i and wire j
        # min goes to i, max goes to j
        # Since values are 0/1: min = 1 - (a OR b) ... actually simpler:
        # if b[i] > b[j], swap
        # Vectorized: mask where b[i] > b[j]
        mask = b[:, i] > b[:, j]
        # Swap where mask is True
        tmp = b[:, i].copy()
        b[:, i] = np.where(mask, b[:, j], b[:, i])
        b[:, j] = np.where(mask, tmp, b[:, j])
    return b


def _is_sorted_net_vec(net):
    """Check if the network sorts all 2^18 Boolean inputs using vectorized ops."""
    # Generate all 2^18 bit patterns
    # Use a lookup approach: for each wire position, the bit is set based on the mask
    all_masks = np.arange(1 << N, dtype=np.int32)
    # bits[pattern, wire] = (pattern >> wire) & 1
    # Shape: (2^N, N)
    # Efficient: use broadcasting
    wire_positions = np.arange(N, dtype=np.int32)
    bits = ((all_masks[:, None] >> wire_positions[None, :]) & 1).astype(np.int8)
    
    out = _apply_network_vec(net, bits)
    
    # Check non-decreasing: out[k] <= out[k+1] for all k
    # For 0/1 values, this means no pattern has a 1 followed by a 0
    violations = np.any(out[:, :-1] > out[:, 1:], axis=1)
    return not np.any(violations)


def _is_sorted_net(net):
    """Check if the network sorts all 2^N Boolean inputs."""
    return _is_sorted_net_vec(net)


def _bitonic_merge(n, k, offset=0, reverse=False):
    """Generate comparators for bitonic merge of n elements."""
    comps = []
    if k == 1:
        return comps
    half = k // 2
    comps.extend(_bitonic_merge(n, half, offset, reverse))
    comps.extend(_bitonic_merge(n, half, offset + half, not reverse))
    if not reverse:
        comps.extend([(offset + i, offset + i + half) for i in range(half)])
    else:
        comps.extend([(offset + i + half, offset + i) for i in range(half)])
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires."""
    comps = []
    for k in range(2, n + 1, 2):
        for i in range(0, n, k):
            comps.extend(_bitonic_merge(n, k, i, False))
    return comps


def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _try_prune(net, rng, max_iters=20000):
    """Try to remove redundant comparators via local search.
    Strategy: first try removing each comparator (in order), then random removals.
    """
    net = list(net)
    best = net
    best_len = len(net)
    
    # First pass: try removing each comparator from end to start
    # (removing later comparators is often easier)
    for idx in range(len(net) - 1, -1, -1):
        if _time_left() < 10:
            break
        trial = net[:idx] + net[idx + 1:]
        if _is_sorted_net(trial):
            net = trial
            best = trial
            best_len = len(trial)
            if best_len <= 68:
                break
    
    # Second pass: random removals for further optimization
    for it in range(max_iters):
        if _time_left() < 10:
            break
        idx = rng.randrange(len(net))
        trial = net[:idx] + net[idx + 1:]
        if _is_sorted_net(trial):
            net = trial
            best = trial
            best_len = len(trial)
            if best_len <= 68:
                break
    return best


def construct():
    """Construct a sorting network on 18 wires, minimizing comparator count."""
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)

    # Generate candidate starting networks
    bitonic_net = _bitonic_sort_network(N)
    oet_net = _odd_even_transposition(N)
    
    # Pick the smaller valid one
    candidates = []
    if len(bitonic_net) < len(oet_net):
        candidates.append(bitonic_net)
        candidates.append(oet_net)
    else:
        candidates.append(oet_net)
        candidates.append(bitonic_net)
    
    # Verify the first candidate
    net = candidates[0]
    if not _is_sorted_net(net):
        net = candidates[1]
        if not _is_sorted_net(net):
            # Fallback: should not happen
            net = oet_net
    
    # Local search: try to remove comparators.
    net = _try_prune(net, rng, max_iters=30000)

    # If we still have time, try additional rounds of pruning.
    while _time_left() > 15:
        net = _try_prune(net, rng, max_iters=20000)

    # Final verification.
    if not _is_sorted_net(net):
        # Fallback to the original valid network.
        net = oet_net

    return net
