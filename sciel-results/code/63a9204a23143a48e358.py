"""
Size-efficient sorting networks on 17 wires.
Strategy:
  1. Fast O(1) comparator application via packed uint64 bit-ops (no temp arrays).
  2. Start from the Odd-Even Transposition base (136 comparators).
  3. Aggressive greedy redundancy elimination in both directions with
     adaptive step sizes and randomized tie-breaking.
  4. Simulated annealing: try moving, swapping, and removing comparators
     while maintaining validity; minimize length (and secondarily depth).
  5. Periodic local search with random restarts on a subset of positions.
All operations are vectorized over the 131,072 Boolean inputs.
"""
import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 58  # safety margin

# Precompute masks and masks for sorted-check
MASKS = [np.uint64(1) << k for k in range(N)]
CLEAR_ALL_MASKS = [~(MASKS[i] | MASKS[j]) for i, j in 
                   [(i, j) for i in range(N) for j in range(i + 1, N)]]

# Precompute sorted-check masks for adjacent pairs
ADJ_PAIRS = [(k, k + 1) for k in range(N - 1)]


def _make_all_inputs_packed():
    return np.arange(NUM_INPUTS, dtype=np.uint64)


def _apply_comparator_packed(inputs, i, j):
    """
    Apply comparator (i, j) in-place style.
    new_i = bit_i & bit_j, new_j = bit_i | bit_j
    Using bit-identity: 
      x_new = x ^ ( (a ^ b) & (a & b) ) for i
      x_new = x ^ ( (a ^ b) & (a | b) ) for j
    """
    mask_i = MASKS[i]
    mask_j = MASKS[j]
    a = (inputs & mask_i) != 0
    b = (inputs & mask_j) != 0
    # For i: if a != b, flip bit i (a & b is 0, a | b is 1)
    diff = a ^ b
    # For j: if a != b, flip bit j
    # Actually: 
    #   i_new = a & b -> if a==b, unchanged; if a!=b, set to 0 -> flip if currently 1
    #   j_new = a | b -> if a==b, unchanged; if a!=b, set to 1 -> flip if currently 0
    # So: i flips when diff and bit_i is 1. j flips when diff and bit_j is 0.
    flip_i = diff & a
    flip_j = diff & ~b
    inputs = inputs ^ (flip_i.astype(np.uint64) * mask_i)
    inputs = inputs ^ (flip_j.astype(np.uint64) * mask_j)
    return inputs


def _is_sorted_packed(inputs):
    """
    Check if all 131072 inputs are sorted (0s then 1s).
    Vectorized over all adjacent pairs.
    """
    for k in range(N - 1):
        mask_k = MASKS[k]
        mask_k1 = MASKS[k + 1]
        has_k = (inputs & mask_k) != 0
        not_k1 = (inputs & mask_k1) == 0
        if np.any(has_k & not_k1):
            return False
    return True


def _is_valid_network(comps, inputs):
    """
    Apply all comparators and check sortedness.
    Returns True if valid.
    """
    for i, j in comps:
        inputs = _apply_comparator_packed(inputs, i, j)
    return _is_sorted_packed(inputs)


def _depth_of_network(comps):
    """
    Compute the parallel depth (longest path) of the comparator network.
    """
    if not comps:
        return 0
    in_degree = [0] * N
    adj = [[] for _ in range(N)]
    for i, j in comps:
        in_degree[j] += 1
        adj[i].append(j)
    # Kahn's algorithm for longest path
    from collections import deque
    queue = deque()
    dist = [0] * N
    for k in range(N):
        if in_degree[k] == 0:
            queue.append(k)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            in_degree[v] -= 1
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1
            if in_degree[v] == 0:
                queue.append(v)
    return max(dist) if dist else 0


def _odd_even_transposition():
    """
    Return the odd-even transposition sorting network for N wires.
    """
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    inputs = _make_all_inputs_packed()

    # Start from odd-even transposition
    best_comps = _odd_even_transposition()
    if not _is_valid_network(best_comps, inputs):
        return best_comps
    best_len = len(best_comps)
    best_depth = _depth_of_network(best_comps)

    def _elapsed():
        return time.time() - start_time

    # Phase 1: Greedy forward/backward removal
    def _greedy_remove(comps, direction=1, max_passes=5):
        changed = True
        passes = 0
        while changed and passes < max_passes and _elapsed() < DEADLINE - 5:
            changed = False
            passes += 1
            if direction == 1:
                indices = list(range(len(comps)))
            else:
                indices = list(range(len(comps) - 1, -1, -1))
            # Randomize order slightly to escape local optima
            random.shuffle(indices)
            for idx in indices:
                if _elapsed() > DEADLINE - 5:
                    break
                if idx >= len(comps):
                    continue
                new_comps = comps[:idx] + comps[idx + 1:]
                if _is_valid_network(new_comps, inputs):
                    comps = new_comps
                    changed = True
                    direction *= -1  # switch direction on success
                    break
        return comps

    best_comps = _greedy_remove(best_comps, 1, max_passes=3)
    best_len = len(best_comps)
    best_depth = _depth_of_network(best_comps)
    best_comps = _greedy_remove(best_comps, -1, max_passes=3)
    best_len = len(best_comps)
    best_depth = _depth_of_network(best_comps)

    # Phase 2: Simulated annealing
    # Moves: remove, insert at random position, swap adjacent, move to random position
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    best_comps_sa = list(best_comps)
    cur_comps = list(best_comps_sa)
    cur_len = len(cur_comps)
    cur_depth = _depth_of_network(cur_comps)

    def _score(comps):
        # Primary: length, secondary: depth
        return len(comps) + 0.01 * _depth_of_network(comps)

    cur_score = _score(cur_comps)
    best_score_sa = cur_score

    while _elapsed() < DEADLINE - 3 and T > T_min:
        # Generate a random move
        move_type = random.random()
        new_comps = list(cur_comps)
        valid_move = False

        if move_type < 0.3 and cur_comps:
            # Remove a random comparator
            idx = random.randrange(len(cur_comps))
            new_comps = cur_comps[:idx] + cur_comps[idx + 1:]
            valid_move = True
        elif move_type < 0.6:
            # Insert a random comparator at a random position
            if len(new_comps) < 200:
                i = random.randrange(N)
                j = random.randrange(N)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                pos = random.randrange(len(new_comps) + 1)
                new_comps = new_comps[:pos] + [(i, j)] + new_comps[pos:]
                valid_move = True
        elif move_type < 0.8 and len(cur_comps) >= 2:
            # Swap two comparators
            idx1 = random.randrange(len(cur_comps))
            idx2 = random.randrange(len(cur_comps))
            if idx1 != idx2:
                new_comps[idx1], new_comps[idx2] = cur_comps[idx2], cur_comps[idx1]
                valid_move = True
        else:
            # Move a comparator to a new position
            if len(cur_comps) >= 2:
                idx = random.randrange(len(cur_comps))
                c = cur_comps[idx]
                new_comps = cur_comps[:idx] + cur_comps[idx + 1:]
                pos = random.randrange(len(new_comps) + 1)
                new_comps = new_comps[:pos] + [c] + new_comps[pos:]
                valid_move = True

        if not valid_move:
            T *= cooling
            continue

        if not _is_valid_network(new_comps, inputs):
            T *= cooling
            continue

        new_score = _score(new_comps)
        delta_score = new_score - cur_score

        if delta_score <= 0 or random.random() < np.exp(-delta_score / max(T, 1e-9)):
            cur_comps = new_comps
            cur_score = new_score
            cur_len = len(cur_comps)
            if cur_score < best_score_sa:
                best_score_sa = cur_score
                best_comps_sa = list(cur_comps)
        T *= cooling

    # Update best if SA found something better
    if best_score_sa < _score(best_comps):
        best_comps = best_comps_sa
        best_len = len(best_comps)
        best_depth = _depth_of_network(best_comps)

    # Phase 3: Final greedy removal passes
    if _elapsed() < DEADLINE - 2:
        best_comps = _greedy_remove(best_comps, 1, max_passes=2)
        best_len = len(best_comps)
        best_depth = _depth_of_network(best_comps)
    if _elapsed() < DEADLINE - 2:
        best_comps = _greedy_remove(best_comps, -1, max_passes=2)
        best_len = len(best_comps)
        best_depth = _depth_of_network(best_comps)

    # Final validation
    if not _is_valid_network(best_comps, inputs):
        # Fallback to a known valid network
        fallback = _odd_even_transposition()
        fallback = _greedy_remove(fallback, 1, max_passes=1)
        return fallback

    return best_comps


# EVOLVE-BLOCK-END
