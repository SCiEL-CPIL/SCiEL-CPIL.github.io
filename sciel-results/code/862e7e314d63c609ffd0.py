"""
Fast optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses vectorized NumPy delta computation, multi-restart simulated annealing,
and parallel workers for maximum exploration within the time budget.
"""
import time
import random
import numpy as np
import multiprocessing as mp
import os


def _precompute():
    """Precompute all AP data shared across workers."""
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

    # For each position, store (ap_index, other_positions) for fast delta
    pos_to_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        for pos in (i0, i1, i2, i3):
            others = tuple(x for x in (i0, i1, i2, i3) if x != pos)
            pos_to_aps[pos].append((idx, others))

    # Precompute as numpy arrays for vectorized access
    pos_others_arrays = []
    for pos in range(256):
        if pos_to_aps[pos]:
            others_list = [o for (_, o) in pos_to_aps[pos]]
            pos_others_arrays.append(np.array(others_list, dtype=np.int32))
        else:
            pos_others_arrays.append(np.empty((0, 3), dtype=np.int32))

    aps_arr = np.array(aps, dtype=np.int32)

    return aps_arr, pos_others_arrays


def _count_mono(colors, aps_arr):
    c = colors[aps_arr]
    return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))


def _calc_delta(colors, pos, pos_others_arrays):
    others = pos_others_arrays[pos]
    if len(others) == 0:
        return 0
    c_others = colors[others]
    all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
    if not np.any(all_same):
        return 0
    pos_color = colors[pos]
    other_color = c_others[all_same, 0]
    delta = int(np.sum(1 - 2 * (pos_color == other_color)))
    return delta


def _greedy_local_search(colors, current_count, time_limit, aps_arr, pos_others_arrays):
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for pos in range(256):
            delta = _calc_delta(colors, pos, pos_others_arrays)
            if delta < 0:
                colors[pos] = 1 - colors[pos]
                current_count += delta
                improved = True
    return colors, current_count


def _simulated_annealing(colors, current_count, T_start, T_end, cooling, time_limit, rng, aps_arr, pos_others_arrays):
    T = T_start
    while time.time() < time_limit and T > T_end:
        pos = rng.randint(0, 255)
        delta = _calc_delta(colors, pos, pos_others_arrays)
        if delta <= 0:
            colors[pos] = 1 - colors[pos]
            current_count += delta
        else:
            if rng.random() < np.exp(-delta / T):
                colors[pos] = 1 - colors[pos]
                current_count += delta
        T *= cooling
    return colors, current_count


def _worker(args):
    worker_id, aps_arr, pos_others_arrays, time_budget, seed = args
    rng = random.Random(seed)

    best_count = float('inf')
    best_colors = None

    # Multiple restarts within this worker
    num_restarts = max(1, int(time_budget / 5))  # ~5s per restart
    time_per_restart = time_budget / num_restarts

    for restart in range(num_restarts):
        remaining = time_budget - (time.time() - args[0] if isinstance(args[0], float) else 0)
        if remaining < 2:
            break

        sa_time = time_per_restart * 0.6
        greedy_time = time_per_restart * 0.4

        # Different initializations
        if restart == 0:
            colors = np.random.randint(0, 2, size=256).astype(np.int8)
        elif restart == 1:
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif restart == 2:
            colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 3:
            colors = np.array([(i // 4) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 4:
            colors = np.array([(i // 8) % 2 for i in range(256)], dtype=np.int8)
        else:
            colors = np.random.randint(0, 2, size=256).astype(np.int8)

        current_count = _count_mono(colors, aps_arr)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Run simulated annealing
        T_start = 20.0
        T_end = 0.3
        cooling = 0.9995
        sa_deadline = time.time() + sa_time
        colors, current_count = _simulated_annealing(
            colors, current_count, T_start, T_end, cooling, sa_deadline, rng, aps_arr, pos_others_arrays
        )

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Run greedy local search
        greedy_deadline = time.time() + greedy_time
        colors, current_count = _greedy_local_search(colors, current_count, greedy_deadline, aps_arr, pos_others_arrays)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    if best_colors is not None:
        return (worker_id, best_count, best_colors)
    return (worker_id, float('inf'), None)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    aps_arr, pos_others_arrays = _precompute()

    # Determine number of workers
    num_workers = min(os.cpu_count() or 1, 4)
    time_budget = deadline - time.time() - 5  # Leave 5s for final polish

    if time_budget < 10:
        # Not enough time for parallel; run single-threaded
        rng = random.Random(12345)
        best_count = float('inf')
        best_colors = None

        colors = np.random.randint(0, 2, size=256).astype(np.int8)
        current_count = _count_mono(colors, aps_arr)
        best_count = current_count
        best_colors = colors.copy()

        sa_deadline = time.time() + time_budget * 0.6
        colors, current_count = _simulated_annealing(
            colors, current_count, 20.0, 0.3, 0.9995, sa_deadline, rng, aps_arr, pos_others_arrays
        )
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        greedy_deadline = time.time() + time_budget * 0.4
        colors, current_count = _greedy_local_search(colors, current_count, greedy_deadline, aps_arr, pos_others_arrays)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        return [int(c) for c in best_colors]

    # Run parallel workers
    worker_args = []
    for i in range(num_workers):
        worker_args.append((i, aps_arr, pos_others_arrays, time_budget / num_workers, 12345 + i * 1000))

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker, worker_args)
    except Exception:
        # Fallback to single-threaded if multiprocessing fails
        rng = random.Random(12345)
        best_count = float('inf')
        best_colors = None

        colors = np.random.randint(0, 2, size=256).astype(np.int8)
        current_count = _count_mono(colors, aps_arr)
        best_count = current_count
        best_colors = colors.copy()

        sa_deadline = time.time() + time_budget * 0.6
        colors, current_count = _simulated_annealing(
            colors, current_count, 20.0, 0.3, 0.9995, sa_deadline, rng, aps_arr, pos_others_arrays
        )
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        greedy_deadline = time.time() + time_budget * 0.4
        colors, current_count = _greedy_local_search(colors, current_count, greedy_deadline, aps_arr, pos_others_arrays)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        return [int(c) for c in best_colors]

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

    # Final polish with remaining time
    remaining = deadline - time.time()
    if remaining > 3:
        rng = random.Random(99999)
        colors = best_colors.copy()
        current_count = best_count

        # Quick SA polish
        sa_deadline = time.time() + remaining * 0.5
        colors, current_count = _simulated_annealing(
            colors, current_count, 5.0, 0.1, 0.999, sa_deadline, rng, aps_arr, pos_others_arrays
        )
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Final greedy
        greedy_deadline = time.time() + remaining * 0.5
        colors, current_count = _greedy_local_search(colors, current_count, greedy_deadline, aps_arr, pos_others_arrays)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    return [int(c) for c in best_colors]
