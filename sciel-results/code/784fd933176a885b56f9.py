"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _apply_network(net, bits):
    """Apply a sorting network to a list of bits. Returns sorted list."""
    vals = list(bits)
    for i, j in net:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _is_sorted(vals):
    for k in range(len(vals) - 1):
        if vals[k] > vals[k+1]:
            return False
    return True

def _verify_network(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-vector simulation."""
    # For each input mask, we track the value on each wire.
    # Since we only care about 0/1, we can simulate all 2^N inputs at once using bitmasks.
    # Let state[w] be a bitmask of length 2^N where bit k is 1 if wire w has value 1 for input k.
    # Initially, for input k (0..2^N-1), wire w has value (k >> w) & 1.
    # So state[w] = bitmask where bit k is (k >> w) & 1.
    
    total_inputs = 1 << N
    # Precompute the initial state for each wire
    # state[w] has bit k set if (k >> w) & 1 == 1
    # This is a pattern of blocks of 2^w ones and 2^w zeros.
    state = []
    for w in range(N):
        mask = 0
        # For each k in 0..total_inputs-1, bit w of k
        # We can compute this efficiently.
        # The pattern is: 2^w zeros, 2^w ones, 2^w zeros, 2^w ones, ...
        # Since total_inputs is a power of 2, and w < N, the number of blocks is 2^(N-w).
        # Let's just build it.
        for k in range(total_inputs):
            if (k >> w) & 1:
                mask |= (1 << k)
        state.append(mask)
    
    # Apply comparators
    for i, j in net:
        # compare-exchange: min to i, max to j
        # new_i = state[i] & state[j]  (both 1)
        # new_j = state[i] | state[j]  (at least one 1)
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
    
    # Check if sorted: for all k, wire 0 <= wire 1 <= ... <= wire N-1
    # This means: state[0] should be subset of state[1] subset of ... subset of state[N-1]
    # i.e., state[w] & ~state[w+1] should be 0 for all w
    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False
    return True

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _bitonic_sort_network(n):
    """Bitonic sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Sort first half ascending, second half descending
    left = _bitonic_sort_network(half)
    right = _bitonic_sort_network(n - half)
    # Shift right and reverse for descending
    right_reversed = [(j, i) for i, j in right]
    right_reversed_shifted = [(i + half, j + half) for i, j in right_reversed]
    # Merge
    merge = _bitonic_merge(n)
    return left + right_reversed_shifted + merge

def _bitonic_merge(n):
    """Bitonic merge network."""
    if n <= 1:
        return []
    half = n // 2
    result = [(i, i + half) for i in range(half)]
    left = _bitonic_merge(half)
    right = _bitonic_merge(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    return result + left + right_shifted

def _build_bitonic():
    return _bitonic_sort_network(N)

def _build_odd_even():
    """Odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _local_search_remove(net, deadline):
    """Try to remove comparators one by one, verifying each time."""
    best = list(net)
    best_score = len(best)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(best)):
            if time.time() >= deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network(candidate):
                best = candidate
                best_score = len(best)
                changed = True
                break
    return best

def _local_search_replace(net, deadline):
    """Try to replace comparators to enable further removals."""
    best = list(net)
    best_score = len(best)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(best)):
            if time.time() >= deadline:
                break
            i, j = best[idx]
            # Try replacing with other comparators
            for a in range(N):
                for b in range(a + 1, N):
                    if (a, b) == (i, j):
                        continue
                    candidate = list(best)
                    candidate[idx] = (a, b)
                    if _verify_network(candidate):
                        best = candidate
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return best

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # Start with bitonic sort
    net = _build_bitonic()
    
    # Verify
    if not _verify_network(net):
        net = _build_odd_even()
    
    # Phase 1: Greedy removal
    if time.time() < deadline - 5.0:
        net = _local_search_remove(net, deadline - 5.0)
    
    # Phase 2: Replace to enable more removals
    if time.time() < deadline - 2.0:
        net = _local_search_replace(net, deadline - 2.0)
    
    # Phase 3: More removal
    if time.time() < deadline:
        net = _local_search_remove(net, deadline)
    
    return net

# EVOLVE-BLOCK-END
