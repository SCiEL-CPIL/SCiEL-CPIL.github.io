"""
Fast optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses vectorized NumPy delta computation and batched greedy local search.
"""
import time
import random
import numpy as np
import multiprocessing as mp
import os


def _precompute():
    """Precompute all AP data and padded neighbor structures for vectorized delta."""
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))

    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, store the indices of the other 3 elements in each AP it belongs to
    pos_to_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        for pos in (i0, i1, i2, i3):
            others = tuple(x for x in (i0, i1, i2, i3) if x != pos)
            pos_to_aps[pos].append(others)

    # Find max number of APs any position is in
    max_aps_per_pos = max(len(lst) for lst in pos_to_aps)
    
    # Pad to fixed size for vectorization: (256, max_aps_per_pos, 3)
    others_padded = np.zeros((256, max_aps_per_pos, 3), dtype=np.int32)
    # Mask to indicate valid entries
    valid_mask = np.zeros((256, max_aps_per_pos), dtype=bool)
    
    for pos in range(256):
        n_aps = len(pos_to_aps[pos])
        if n_aps > 0:
            others_padded[pos, :n_aps, :] = np.array(pos_to_aps[pos], dtype=np.int32)
            valid_mask[pos, :n_aps] = True

    return aps_arr, others_padded, valid_mask


def _count_mono(colors, aps_arr):
    c = colors[aps_arr]
    return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))


def _calc_deltas_vectorized(colors, others_padded, valid_mask):
    """
    Calculate the delta in objective function if we flip each bit individually.
    Delta < 0 means flipping improves (reduces mono count).
    """
    # colors: (256,)
    # others_padded: (256, K, 3)
    # valid_mask: (256, K)
    
    # Get colors of neighbors
    c_others = colors[others_padded] # (256, K, 3)
    
    # Check if the 3 neighbors are monochromatic
    is_mono = (c_others[:, :, 0] == c_others[:, :, 1]) & (c_others[:, :, 1] == c_others[:, :, 2])
    
    # If is_mono, the color of the triple is c_others[:,:,0]
    # If we flip position i:
    # If colors[i] == triple_color, we break a mono AP -> delta -1
    # If colors[i] != triple_color, we create a mono AP -> delta +1
    # If not is_mono, delta 0
    
    c_self = colors[:, np.newaxis, np.newaxis] # (256, 1, 1)
    c_triple = c_others[:, :, 0] # (256, K)
    
    matches = (c_self == c_triple) # (256, K)
    
    # Contribution: -1 if match, +1 if not match, 0 if not mono
    contrib = np.where(matches, -1.0, 1.0)
    contrib = np.where(is_mono, contrib, 0.0)
    
    # Mask out invalid entries
    contrib = np.where(valid_mask[:, :, np.newaxis], contrib, 0.0)
    
    deltas = contrib.sum(axis=1)
    return deltas


def _greedy_batched(colors, current_count, time_limit, aps_arr, others_padded, valid_mask):
    """
    Batched greedy descent: flip all bits with negative delta simultaneously.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        deltas = _calc_deltas_vectorized(colors, others_padded, valid_mask)
        
        # Find positions where flipping improves the score
        flip_mask = deltas < 0
        
        if np.any(flip_mask):
            # Update colors
            colors[flip_mask] = 1 - colors[flip_mask]
            # Update count
            current_count += int(np.sum(deltas[flip_mask]))
            improved = True
            
    return colors, current_count


def _worker(args):
    worker_id, aps_arr, others_padded, valid_mask, time_budget, seed = args
    rng = random.Random(seed)

    best_count = float('inf')
    best_colors = None

    # Multiple restarts
    num_restarts = max(1, int(time_budget / 2)) # ~2s per restart for vectorized greedy
    time_per_restart = time_budget / num_restarts

    for restart in range(num_restarts):
        if time.time() > time_budget:
            break

        # Initialization
        if restart == 0:
            colors = np.random.randint(0, 2, size=256).astype(np.int8)
        elif restart == 1:
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif restart == 2:
            colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 3:
            colors = np.array([(i // 4) % 2 for i in range(256)], dtype=np.int8)
        else:
            # Random block pattern
            block_size = rng.choice([2, 4, 8, 16])
            colors = np.array([rng.randint(0,1) for _ in range(256//block_size)] * block_size, dtype=np.int8)
            colors = colors[:256]

        current_count = _count_mono(colors, aps_arr)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Run batched greedy
        greedy_deadline = time.time() + time_per_restart
        colors, current_count = _greedy_batched(colors, current_count, greedy_deadline, aps_arr, others_padded, valid_mask)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    if best_colors is not None:
        return (worker_id, best_count, best_colors)
    return (worker_id, float('inf'), None)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    aps_arr, others_padded, valid_mask = _precompute()

    # Determine number of workers
    num_workers = min(os.cpu_count() or 1, 4)
    time_budget = deadline - time.time() - 5

    if time_budget < 5:
        # Fallback single thread
        colors = np.random.randint(0, 2, size=256).astype(np.int8)
        current_count = _count_mono(colors, aps_arr)
        colors, current_count = _greedy_batched(colors, current_count, deadline, aps_arr, others_padded, valid_mask)
        return [int(c) for c in colors]

    worker_args = []
    for i in range(num_workers):
        worker_args.append((i, aps_arr, others_padded, valid_mask, time_budget / num_workers, 12345 + i * 1000))

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker, worker_args)
    except Exception:
        colors = np.random.randint(0, 2, size=256).astype(np.int8)
        current_count = _count_mono(colors, aps_arr)
        colors, current_count = _greedy_batched(colors, current_count, deadline, aps_arr, others_padded, valid_mask)
        return [int(c) for c in colors]

    # Combine results
    best_count = float('inf')
    best_colors = None
    for worker_id, count, colors in results:
        if colors is not None and count < best_count:
            best_count = count
            best_colors = colors.copy()

    if best_colors is None:
        best_colors = np.random.randint(0, 2, size=256).astype(np.int8)
        best_count = _count_mono(best_colors, aps_arr)

    # Final polish
    remaining = deadline - time.time()
    if remaining > 2:
        colors = best_colors.copy()
        current_count = best_count
        colors, current_count = _greedy_batched(colors, current_count, time.time() + remaining, aps_arr, others_padded, valid_mask)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    return [int(c) for c in best_colors]
