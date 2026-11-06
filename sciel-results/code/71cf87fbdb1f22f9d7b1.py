"""
Size-efficient sorting networks on 17 wires.
Strategy:
  1. Fast O(1) comparator application via packed uint64 bit-ops (no temp arrays).
  2. Start from the Odd-Even Transposition base (136 comparators).
  3. Aggressive greedy redundancy elimination in both directions.
  4. Iterated Local Search (ILS): Perturb the current best network by adding
     random comparators, then apply aggressive greedy deletion to find a
     potentially smaller valid network. This escapes local minima better
     than standard Simulated Annealing for this problem.
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

# Precompute sorted-check masks for adjacent pairs
ADJ_PAIRS = [(k, k + 1) for k in range(N - 1)]


def _make_all_inputs_packed():
    return np.arange(NUM_INPUTS, dtype=np.uint64)


def _apply_comparator_packed(inputs, i, j):
    """
    Apply comparator (i, j) in-place style.
    """
    mask_i = MASKS[i]
    mask_j = MASKS[j]
    a = (inputs & mask_i) != 0
    b = (inputs & mask_j) != 0
    diff = a ^ b
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
    
    def _elapsed():
        return time.time() - start_time

    # Fast Greedy Removal
    # This function tries to remove comparators one by one.
    # It uses a randomized order to escape local optima.
    def _fast_greedy_remove(comps, max_passes=10):
        comps = list(comps)
        for _ in range(max_passes):
            if _elapsed() > DEADLINE - 2:
                break
            changed = False
            indices = list(range(len(comps)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(comps):
                    continue
                new_comps = comps[:idx] + comps[idx + 1:]
                if _is_valid_network(new_comps, inputs):
                    comps = new_comps
                    changed = True
                    # Restart pass immediately to take advantage of new state
                    break
            if not changed:
                break
        return comps

    # Initial aggressive pruning
    best_comps = _fast_greedy_remove(best_comps)
    best_len = len(best_comps)

    # Iterated Local Search
    # We perturb the current best network by adding random comparators,
    # then prune it back down. This allows us to jump to different
    # local minima.
    while _elapsed() < DEADLINE - 5:
        # Perturb: Add K random comparators
        k_perturb = random.randint(3, 8)
        perturbed = list(best_comps)
        for _ in range(k_perturb):
            i = random.randrange(N)
            j = random.randrange(N)
            if i != j:
                if i > j:
                    i, j = j, i
                pos = random.randrange(len(perturbed) + 1)
                perturbed.insert(pos, (i, j))
        
        # Prune the perturbed network
        candidate = _fast_greedy_remove(perturbed)
        
        # Accept if better
        if len(candidate) < best_len:
            best_comps = candidate
            best_len = len(best_comps)
            # Optionally, do a few more greedy passes on the new best
            best_comps = _fast_greedy_remove(best_comps, max_passes=5)
            best_len = len(best_comps)

    # Final cleanup
    best_comps = _fast_greedy_remove(best_comps, max_passes=10)
    
    # Final validation
    if not _is_valid_network(best_comps, inputs):
        # Fallback to a known valid network
        fallback = _odd_even_transposition()
        fallback = _fast_greedy_remove(fallback, max_passes=1)
        return fallback

    return best_comps


# EVOLVE-BLOCK-END
