"""Colorings with three colors in every four-term progression.
Uses randomized local search (simulated annealing) to minimize distinct colors
while ensuring every nonconstant 4-term progression uses >=3 distinct colors.
"""
import time
import random
import numpy as np

N = 128
DEADLINE = 75.0  # safety margin under 90s wall-clock

def _build_progressions():
    """Precompute all nonconstant 4-term progressions (i, i+d, i+2d, i+3d) with 0-based indices."""
    progs = []
    for i in range(N):
        for d in range(1, N):
            if i + 3 * d < N:
                progs.append((i, i + d, i + 2 * d, i + 3 * d))
    return progs

PROGS = _build_progressions()
# For fast local checks: for each position, which progressions it participates in
POS_PROGS = [[] for _ in range(N)]
for pi, (a, b, c, d) in enumerate(PROGS):
    for p in (a, b, c, d):
        POS_PROGS[p].append(pi)

def _count_colors(arr):
    return len(set(arr))

def _violations(arr):
    """Count number of 4-term progressions using fewer than 3 distinct colors."""
    v = 0
    for (a, b, c, d) in PROGS:
        ca, cb, cc, cd = arr[a], arr[b], arr[c], arr[d]
        # fewer than 3 distinct means at most 2 distinct
        if ca == cb == cc == cd:
            v += 1
        elif ca == cb == cc or ca == cb == cd or ca == cc == cd or ca == cd == cc:
            v += 1
        elif cb == cc == cd or ca == cb == cd:
            v += 1
        elif ca == cb and cc == cd and ca != cc:
            v += 1
        elif ca == cc and cb == cd and ca != cb:
            v += 1
        elif ca == cd and cb == cc and ca != cb:
            v += 1
    return v

def _local_violations(arr, pos, new_color):
    """Count violations involving `pos` if we set arr[pos]=new_color, given current arr."""
    v = 0
    old = arr[pos]
    if old == new_color:
        return 0
    arr[pos] = new_color
    for pi in POS_PROGS[pos]:
        a, b, c, d = PROGS[pi]
        ca, cb, cc, cd = arr[a], arr[b], arr[c], arr[d]
        distinct = len({ca, cb, cc, cd})
        if distinct < 3:
            v += 1
    arr[pos] = old
    return v

def _current_violations_at(arr, pos):
    """Count violations involving pos in current arr."""
    v = 0
    for pi in POS_PROGS[pos]:
        a, b, c, d = PROGS[pi]
        ca, cb, cc, cd = arr[a], arr[b], arr[c], arr[d]
        if len({ca, cb, cc, cd}) < 3:
            v += 1
    return v

def _anneal(arr, start_temp, min_temp, cooling, max_iters, start_time, deadline):
    """Simulated annealing. Returns (arr, num_colors, violations)."""
    ncolors = _count_colors(arr)
    cur_v = _violations(arr)
    best_arr = arr.copy()
    best_nc = ncolors
    best_v = cur_v

    # We want to minimize (violations, ncolors). Use a combined objective.
    # Primary: violations must be 0. Secondary: minimize ncolors.
    def objective(v, nc):
        # Strong penalty for violations
        return v * 1000 + nc

    cur_obj = objective(cur_v, ncolors)
    best_obj = cur_obj

    i = 0
    while i < max_iters:
        if time.time() > deadline:
            break
        temp = start_temp * (min_temp / start_temp) ** (i / max(1, max_iters))
        i += 1

        # Pick a position
        pos = random.randrange(N)
        # Pick a new color: either a currently-used color or a new one
        used = set(arr)
        if used and random.random() < 0.85:
            new_color = random.choice(list(used))
        else:
            # Try to introduce a new color (could help reduce distinct count later,
            # or just explore). Actually introducing new colors increases ncolors,
            # so mostly we want to merge. Let's mostly pick from used.
            new_color = random.choice(list(used)) if used else 0

        if new_color == arr[pos]:
            continue

        old = arr[pos]
        arr[pos] = new_color
        # Compute delta violations
        # violations change only in progressions containing pos
        # delta = new_v_at_pos - old_v_at_pos
        new_vp = 0
        for pi in POS_PROGS[pos]:
            a, b, c, d = PROGS[pi]
            if len({arr[a], arr[b], arr[c], arr[d]}) < 3:
                new_vp += 1
        # old violations at pos (recompute with old color)
        arr[pos] = old
        old_vp = 0
        for pi in POS_PROGS[pos]:
            a, b, c, d = PROGS[pi]
            if len({arr[a], arr[b], arr[c], arr[d]}) < 3:
                old_vp += 1
        # set new color back
        arr[pos] = new_color
        delta_v = new_vp - old_vp
        cur_v += delta_v

        # ncolors change
        old_nc = ncolors
        if new_color not in used:
            # new color introduced
            ncolors = old_nc + 1
        else:
            # check if old color is still used elsewhere
            if old not in arr:
                ncolors = old_nc - 1
            else:
                ncolors = old_nc

        cur_obj = objective(cur_v, ncolors)
        delta_obj = cur_obj - best_obj  # not quite; use current vs previous
        # Actually for SA we compare to previous current objective
        # Let me restructure: track prev_obj
        # Simpler: accept if improves or with probability exp(-delta/temp)
        # delta = cur_obj - prev_obj
        # We didn't track prev_obj. Let's recompute: prev_obj = cur_obj - delta_obj_cur
        # Easier: just recompute objective before and after.
        # Let's do it cleanly:
        # (This is getting messy; let me just recompute.)

        # Recompute approach: we already have cur_v and ncolors.
        # prev state had cur_v - delta_v and old_nc.
        prev_obj = objective(cur_v - delta_v, old_nc)
        delta = cur_obj - prev_obj

        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
            # accept
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_v = cur_v
                best_nc = ncolors
                best_arr = arr.copy()
        else:
            # reject
            arr[pos] = old
            cur_v -= delta_v
            ncolors = old_nc

    return best_arr, best_nc, best_v

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    # Start with a valid-ish coloring: greedy assignment
    # Actually start with a random small palette
    best_arr = None
    best_score = None  # (violations, ncolors)

    # Initial: use 3 colors in a pattern that's likely to work
    # Try a base pattern: color = (i mod something)
    # Let's start with 2-color alternating and fix violations
    arr = np.zeros(N, dtype=int)
    for i in range(N):
        arr[i] = i % 3

    # Run multiple annealing restarts
    best_result = None
    best_nc = N
    best_v = _violations(arr)

    restart = 0
    while time.time() < deadline - 5.0:
        restart += 1
        # Fresh random start with small palette
        palette_size = 2 + (restart % 3)  # 2,3,4
        arr = np.zeros(N, dtype=int)
        for i in range(N):
            arr[i] = rng.randrange(palette_size)

        # Anneal
        start_temp = 10.0
        min_temp = 0.01
        max_iters = 20000
        res_arr, res_nc, res_v = _anneal(arr, start_temp, min_temp, 0.999, max_iters, start, deadline)

        if res_v == 0:
            if res_nc < best_nc:
                best_nc = res_nc
                best_result = res_arr.copy()

        if best_nc <= 3:
            break

    if best_result is not None:
        # Relabel to 0..k-1
        labels = {}
        next_label = 0
        out = []
        for c in best_result:
            if c not in labels:
                labels[c] = next_label
                next_label += 1
            out.append(labels[c])
        return out
    else:
        # Fallback: return a valid coloring (may use more colors)
        # Use the greedy from seed
        colors = []
        for x in range(1, N + 1):
            for color in range(N):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    prog = {color}
                    for j in (1, 2, 3):
                        prog.add(colors[x - 1 - j * d])
                    if len(prog) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                colors.append(color)
                break
        return colors

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print("time:", t1 - t0)
    print("colors used:", len(set(res)))
    # Verify
    ok = True
    for (a, b, c, d) in PROGS:
        if len({res[a], res[b], res[c], res[d]}) < 3:
            ok = False
            print("VIOLATION at", a, b, c, d)
            break
    print("valid:", ok)
