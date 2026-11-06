"""Size-efficient sorting network synthesis for 18 wires.

Strategy (structural change from local search to recursive synthesis):
  - Fast bit-parallel 0-1 verification.
  - Recursive divide-and-conquer synthesis: sort halves, then merge.
  - Recursive merge network with greedy pruning.
  - Layer-wise independent optimization.
  - Final global greedy removal.
  - Deadline guard with safety margin.
"""
import time
import random
import math

N = 18
_M = 1 << N  # 262144


def _build_mask(w, n_bits):
    """Build bit mask where bit k is set iff (k >> w) & 1."""
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m


_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, _M)


def _verify(comparators):
    """Verify comparator list sorts all 2**N boolean inputs via bitmask method."""
    state = _INITIAL_MASKS[:]
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    for i in range(N - 1):
        if (state[i] & ~state[i + 1]) != 0:
            return False
    return True


def _verify_partial(comparators, n_wires):
    """Verify that comparators sort all 2**n_wires boolean inputs on first n_wires wires.
    Returns True if the network on wires 0..n_wires-1 is valid."""
    state = _INITIAL_MASKS[:n_wires]
    for i, j in comparators:
        if i < n_wires and j < n_wires:
            a = state[i]
            b = state[j]
            state[i] = a & b
            state[j] = a | b
    for i in range(n_wires - 1):
        if (state[i] & ~state[i + 1]) != 0:
            return False
    return True


def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _recursive_merge(k, offset):
    """Recursive merge of k already-sorted halves. Returns (i,j) list.
    Standard odd-even merge network."""
    if k <= 1:
        return []
    result = []
    half_k = k // 2

    # Recursively merge even-indexed positions
    for i in range(offset, offset + k, 2):
        result.extend(_recursive_merge(half_k, i))

    # Recursively merge odd-indexed positions
    for i in range(offset + 1, offset + k, 2):
        result.extend(_recursive_merge(half_k, i))

    # Final adjacent comparisons
    for i in range(offset, offset + k - 1):
        result.append((i, i + 1))

    return result


def _recursive_sort(n, offset=0):
    """Recursive odd-even mergesort for n wires (n should be power of 2 or handled specially).
    Returns list of (i,j) comparators with offset applied."""
    if n <= 1:
        return []

    if n == 2:
        return [(offset, offset + 1)]

    # For non-power-of-2, split as evenly as possible
    half = n // 2
    left_n = half
    right_n = n - half

    left = _recursive_sort(left_n, offset)
    right = _recursive_sort(right_n, offset + left_n)

    # Merge the two sorted halves
    merge = _recursive_merge(n, offset)

    return left + right + merge


def _build_recursive_network(n, offset=0):
    """Build a sorting network for n wires using recursive divide-and-conquer.
    For n not a power of 2, handles the asymmetry."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]

    # Split into two halves as evenly as possible
    left_n = n // 2
    right_n = n - left_n

    left = _build_recursive_network(left_n, offset)
    right = _build_recursive_network(right_n, offset + left_n)

    # Merge: use odd-even merge on the full n wires
    merge = _recursive_merge(n, offset)

    return left + right + merge


def _greedy_remove_pass(current, deadline, n_wires=N):
    """One greedy pass: scan from end, remove if still valid. Returns (new_list, removed_count)."""
    removed = 0
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate = current[:i] + current[i + 1:]
        if n_wires == N:
            if _verify(candidate):
                current = candidate
                removed += 1
        else:
            if _verify_partial(candidate, n_wires):
                current = candidate
                removed += 1
    return current, removed


def _layer_decompose(network):
    """Decompose network into layers (sets of non-conflicting comparators)."""
    if not network:
        return []
    layers = []
    current_layer = []
    used_wires = set()
    for comp in network:
        i, j = comp
        if i not in used_wires and j not in used_wires:
            current_layer.append(comp)
            used_wires.add(i)
            used_wires.add(j)
        else:
            if current_layer:
                layers.append(current_layer)
            current_layer = [comp]
            used_wires = {i, j}
    if current_layer:
        layers.append(current_layer)
    return layers


def _optimize_layer(layer, other_network, deadline, n_wires=N):
    """Try to reduce a single layer by removing comparators that aren't needed.
    other_network contains all comparators outside this layer."""
    if not layer:
        return layer
    # Try removing each comparator from the layer
    optimized = layer[:]
    for i in range(len(optimized) - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate_layer = optimized[:i] + optimized[i + 1:]
        candidate_network = other_network + candidate_layer
        if n_wires == N:
            if _verify(candidate_network):
                optimized = candidate_layer
        else:
            if _verify_partial(candidate_network, n_wires):
                optimized = candidate_layer
    return optimized


def _recursive_synthesize(n, offset, deadline):
    """Recursively synthesize a sorting network for n wires starting at offset.
    Uses divide-and-conquer with greedy pruning at each level."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]

    # Split into two halves
    left_n = n // 2
    right_n = n - left_n

    # Recursively build left and right
    left = _recursive_synthesize(left_n, offset, deadline)
    right = _recursive_synthesize(right_n, offset + left_n, deadline)

    # Build merge network
    merge = _recursive_merge(n, offset)

    # Combine and greedily prune
    network = left + right + merge

    # Greedy removal on the full network
    network, _ = _greedy_remove_pass(network, deadline, n)

    return network


def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences."""
    start_time = time.time()
    # The evaluator kills after 90s wall-clock. Use a conservative deadline.
    deadline = start_time + 78

    # Phase 1: Build recursive divide-and-conquer network
    current = _recursive_synthesize(N, 0, deadline)

    # Verify and fallback if needed
    if not _verify(current):
        # Fallback to Batcher-style
        p = 1
        while p < N:
            p *= 2
        network = _recursive_sort(p, 0)
        network = [(i, j) for i, j in network if i < N and j < N]
        # Deduplicate
        seen = set()
        unique = []
        for comp in network:
            if comp not in seen:
                seen.add(comp)
                unique.append(comp)
        current = unique
        if not _verify(current):
            current = _odd_even_transposition()

    best = current[:]
    best_len = len(best)

    # Phase 2: Layer-wise optimization
    if time.time() < deadline - 20:
        layers = _layer_decompose(best)
        new_network = []
        for idx, layer in enumerate(layers):
            if time.time() > deadline - 20:
                break
            other = new_network + [c for li in layers[idx+1:] for c in li]
            optimized_layer = _optimize_layer(layer, other, deadline, N)
            new_network.extend(optimized_layer)

        if new_network and _verify(new_network) and len(new_network) < best_len:
            best = new_network
            best_len = len(new_network)

    # Phase 3: Multiple greedy removal passes
    while time.time() < deadline - 10:
        current = best[:]
        current, removed = _greedy_remove_pass(current, deadline)
        if removed == 0:
            break
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)

    # Phase 4: Try alternative split points for recursive synthesis
    if time.time() < deadline - 8:
        # Try different split ratios
        for split_frac in [0.4, 0.55, 0.45, 0.6]:
            if time.time() > deadline - 8:
                break
            left_n = int(N * split_frac)
            right_n = N - left_n
            if left_n < 2 or right_n < 2:
                continue

            left = _recursive_synthesize(left_n, 0, deadline)
            right = _recursive_synthesize(right_n, left_n, deadline)
            merge = _recursive_merge(N, 0)
            candidate = left + right + merge
            candidate, _ = _greedy_remove_pass(candidate, deadline)

            if _verify(candidate) and len(candidate) < best_len:
                best = candidate[:]
                best_len = len(candidate)

    # Phase 5: Final greedy cleanup
    if time.time() < deadline - 2:
        best, _ = _greedy_remove_pass(best, deadline)

    # Final safety check
    if not _verify(best):
        best = _odd_even_transposition()

    return best
