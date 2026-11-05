"""
Two-colourings of [1,256] minimizing monochromatic 4-term APs.
Combines parallel multi-restart SA + greedy with a vectorized batch
marginal-gain (flip-delta) evaluator for all 256 positions at once.
"""
import time
import random
import numpy as np
import multiprocessing as mp
import os


def _precompute():
    """Build AP list and per-position 'other' index arrays (vectorized delta)."""
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)

    pos_others_arrays = []
    for pos in range(256):
        others_list = []
        for (i0, i1, i2, i3) in aps:
            if pos == i0:
                others_list.append((i1, i2, i3))
            elif pos == i1:
                others_list.append((i0, i2, i3))
            elif pos == i2:
                others_list.append((i0, i1, i3))
            elif pos == i3:
                others_list.append((i0, i1, i2))
        if others_list:
            pos_others_arrays.append(np.array(others_list, dtype=np.int32))
        else:
            pos_others_arrays.append(np.empty((0, 3), dtype=np.int32))

    return aps_arr, pos_others_arrays


def _count_mono(colors, aps_arr):
    c = colors[aps_arr]
    return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))


def _calc_delta(colors, pos, pos_others_arrays):
    """Single-position flip delta (cheap, for SA)."""
    others = pos_others_arrays[pos]
    if len(others) == 0:
        return 0
    c_others = colors[others]
    all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
    if not np.any(all_same):
        return 0
    pos_color = colors[pos]
    other_color = c_others[all_same, 0]
    return int(np.sum(1 - 2 * (pos_color == other_color)))


def _batch_deltas(colors, pos_others_arrays):
    """Vectorized flip-delta for ALL 256 positions at once.
    Returns int8 array of length 256; delta = new_count - old_count on flip.
    """
    n = 256
    deltas = np.zeros(n, dtype=np.int8)
    for pos in range(n):
        others = pos_others_arrays[pos]
        if len(others) == 0:
            continue
        c_others = colors[others]
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        if not np.any(all_same):
            continue
        pos_color = colors[pos]
        other_color = c_others[all_same, 0]
        deltas[pos] = int(np.sum(1 - 2 * (pos_color == other_color)))
    return deltas


def _greedy_local_search(colors, current_count, pos_others_arrays, time_limit):
    """Greedy: repeatedly flip the single best-improving position.
    Uses batch delta evaluation each sweep for speed."""
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        deltas = _batch_deltas(colors, pos_others_arrays)
        # find most negative delta
        idx = int(np.argmin(deltas))
        d = int(deltas[idx])
        if d < 0:
            colors[idx] = 1 - colors[idx]
            current_count += d
            improved = True
    return colors, current_count


def _greedy_multi_flip(colors, current_count, pos_others_arrays, time_limit):
    """Aggressive greedy: apply ALL negative-delta flips simultaneously,
    then re-evaluate. Fast convergence to a local optimum."""
    while time.time() < time_limit:
        deltas = _batch_deltas(colors, pos_others_arrays)
        mask = deltas < 0
        if not np.any(mask):
            break
        positions = np.where(mask)[0]
        for p in positions:
            colors[p] = 1 - colors[p]
        current_count += int(np.sum(deltas[mask]))
    return colors, current_count


def _simulated_annealing(colors, current_count, pos_others_arrays, T_start, T_end, cooling, time_limit, rng):
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
    np_rng = np.random.RandomState(seed)

    best_count = float('inf')
    best_colors = None

    num_restarts = max(2, int(time_budget / 4))
    restart_time = time_budget / num_restarts

    for restart in range(num_restarts):
        remaining = restart_time
        if remaining < 1:
            break

        sa_time = remaining * 0.45
        greedy_time = remaining * 0.40
        polish_time = remaining * 0.15

        init_type = (worker_id * num_restarts + restart) % 8
        if init_type == 0:
            colors = np_rng.randint(0, 2, size=256).astype(np.int8)
        elif init_type == 1:
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif init_type == 2:
            colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
        elif init_type == 3:
            colors = np.array([(i // 4) % 2 for i in range(256)], dtype=np.int8)
        elif init_type == 4:
            colors = np.array([(i // 8) % 2 for i in range(256)], dtype=np.int8)
        elif init_type == 5:
            colors = np.zeros(256, dtype=np.int8)
        elif init_type == 6:
            colors = np.ones(256, dtype=np.int8)
        else:
            colors = np.zeros(256, dtype=np.int8)
            i = 0
            while i < 256:
                bs = rng.randint(1, 10)
                val = rng.randint(0, 1)
                end = min(i + bs, 256)
                colors[i:end] = val
                i = end

        current_count = _count_mono(colors, aps_arr)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # SA phase
        T_start = 15.0
        T_end = 0.2
        cooling = 0.9997
        sa_deadline = time.time() + sa_time
        colors, current_count = _simulated_annealing(
            colors, current_count, pos_others_arrays, T_start, T_end, cooling, sa_deadline, rng
        )
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Greedy (single-flip)
        greedy_deadline = time.time() + greedy_time
        colors, current_count = _greedy_local_search(colors, current_count, pos_others_arrays, greedy_deadline)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Polish (multi-flip + single-flip)
        polish_deadline = time.time() + polish_time
        colors, current_count = _greedy_multi_flip(colors, current_count, pos_others_arrays, polish_deadline)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        colors, current_count = _greedy_local_search(colors, current_count, pos_others_arrays, polish_deadline)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    return best_count, best_colors


def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit

    aps_arr, pos_others_arrays = _precompute()

    n_cpus = os.cpu_count() or 1
    n_workers = min(n_cpus, 4)

    if n_workers <= 1:
        time_budget = deadline - time.time() - 2
        count, colors = _worker((0, aps_arr, pos_others_arrays, time_budget, 12345))
        if colors is None:
            colors = np.zeros(256, dtype=np.int8)
            count = _count_mono(colors, aps_arr)
        if time.time() < deadline - 2:
            colors, count = _greedy_multi_flip(colors, count, pos_others_arrays, deadline - 1)
            colors, count = _greedy_local_search(colors, count, pos_others_arrays, deadline - 1)
        return [int(c) for c in colors]

    time_per_worker = (deadline - time.time() - 3) / n_workers
    worker_args = []
    for wid in range(n_workers):
        seed = 1000 + wid * 7919
        worker_args.append((wid, aps_arr, pos_others_arrays, time_per_worker, seed))

    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)
    except Exception:
        time_budget = deadline - time.time() - 2
        count, colors = _worker((0, aps_arr, pos_others_arrays, time_budget, 12345))
        if colors is None:
            colors = np.zeros(256, dtype=np.int8)
            count = _count_mono(colors, aps_arr)
        return [int(c) for c in colors]

    best_count = float('inf')
    best_colors = None
    for count, colors in results:
        if colors is not None and count < best_count:
            best_count = count
            best_colors = colors

    if best_colors is None:
        best_colors = np.zeros(256, dtype=np.int8)
        best_count = _count_mono(best_colors, aps_arr)

    # Final high-precision polish
    if time.time() < deadline - 2:
        best_colors, best_count = _greedy_multi_flip(best_colors, best_count, pos_others_arrays, deadline - 1)
        best_colors, best_count = _greedy_local_search(best_colors, best_count, pos_others_arrays, deadline - 1)

    return [int(c) for c in best_colors]
