# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    total_points = q * q  # 2401
    n_dirs = 50  # 49 slopes + vertical

    start_time = time.time()
    timeout = 610  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element i = a + b*alpha where a = i % 7, b = i // 7
    def add_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        return (a1 + a2) % p + p * ((b1 + b2) % p)

    def sub_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        return (a1 - a2) % p + p * ((b1 - b2) % p)

    def mul_f(i, j):
        a1, b1 = i % p, i // p
        a2, b2 = j % p, j // p
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + b1 * a2) % p
        return a + p * b

    def inv_f(i):
        if i == 0:
            return 0
        a, b = i % p, i // p
        norm = (a * a - 3 * b * b) % p
        norm_inv = pow(norm, p - 2, p)
        return mul_f(i, norm_inv)

    # Precompute multiplication table
    mul_table = [[mul_f(i, j) for j in range(q)] for i in range(q)]
    inv_table = [inv_f(i) for i in range(q)]

    # Precompute all lines
    line_id = np.zeros((total_points, n_dirs), dtype=np.int32)
    for pt in range(total_points):
        xa = pt // (q * p)
        xb = (pt // q) % p
        x_elem = xa + p * xb
        line_id[pt, 49] = x_elem

    for d in range(49):
        for pt in range(total_points):
            xa = pt // (q * p)
            xb = (pt // q) % p
            ya = (pt // p) % p
            yb = pt % p
            x_elem = xa + p * xb
            y_elem = ya + p * yb
            c = sub_f(y_elem, mul_f(d, x_elem))
            line_id[pt, d] = c

    # Build line_points: for each (d, line_val), collect all points
    line_points = {}
    for d in range(n_dirs):
        for lv in range(q):
            pts = []
            for pt in range(total_points):
                if line_id[pt, d] == lv:
                    pts.append(pt)
            line_points[(d, lv)] = np.array(pts, dtype=np.int32)

    # Precompute other_pts[pt][d] = 48 other point indices
    print("Precomputing line members...", flush=True)
    t0 = time.time()
    other_pts = np.zeros((total_points, n_dirs, q - 1), dtype=np.int32)
    for d in range(n_dirs):
        for lv in range(q):
            pts_arr = line_points[(d, lv)]
            for i, pt in enumerate(pts_arr):
                other_pts[pt, d, :] = np.delete(pts_arr, i)
    print(f"Precomputation done in {time.time()-t0:.1f}s", flush=True)

    # Greedy build with incremental line counts
    def greedy_build(S_init):
        s_in = np.zeros(total_points, dtype=bool)
        for pt in S_init:
            s_in[pt] = True

        line_count = np.zeros((total_points, n_dirs), dtype=np.int32)
        for pt in range(total_points):
            for d in range(n_dirs):
                line_count[pt, d] = np.sum(s_in[other_pts[pt, d, :]])

        free_count = np.sum(line_count == 0, axis=1)
        current_S_size = int(np.sum(s_in))
        candidates = [pt for pt in range(total_points) if not s_in[pt] and free_count[pt] >= 1]

        iteration = 0
        while candidates and time.time() - start_time < timeout - 30:
            iteration += 1
            if random.random() < 0.3:
                best_pt = -1
                best_damage = float('inf')
                sample = random.sample(candidates, min(20, len(candidates)))
                for pt in sample:
                    damage = 0
                    for d in range(n_dirs):
                        for y in other_pts[pt, d, :]:
                            if line_count[y, d] == 0 and free_count[y] == 1:
                                damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_pt = pt
                if best_pt >= 0:
                    pt = best_pt
                else:
                    pt = random.choice(candidates)
            else:
                pt = random.choice(candidates)

            if free_count[pt] < 1:
                try:
                    candidates.remove(pt)
                except:
                    pass
                continue

            valid = True
            for d in range(n_dirs):
                for y in other_pts[pt, d, :]:
                    if line_count[y, d] == 0 and free_count[y] == 1:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                try:
                    candidates.remove(pt)
                except:
                    pass
                continue

            s_in[pt] = True
            current_S_size += 1

            for d in range(n_dirs):
                for y in other_pts[pt, d, :]:
                    if line_count[y, d] == 0:
                        free_count[y] -= 1
                    line_count[y, d] += 1

            try:
                candidates.remove(pt)
            except:
                pass

            if iteration % 100 == 0:
                candidates = [c for c in candidates if not s_in[c] and free_count[c] >= 1]

            if time.time() - start_time > timeout - 30:
                break

        return s_in, current_S_size

    # Initial S
    S_init = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_elem = xa + p * xb
                    y_elem = ya + p * yb
                    trace_x = (2 * xa) % p
                    norm_y = (ya * ya - 3 * yb * yb) % p
                    in_unital = (trace_x + norm_y) % p == 0
                    in_Y = (ya in [u * u % p for u in range(p)])
                    if in_unital and not in_Y:
                        pt = ((xa * p + xb) * q) + (ya * p + yb)
                        S_init.append(pt)

    print(f"Initial S size: {len(S_init)}", flush=True)
    t0 = time.time()
    s_in, S_size = greedy_build(S_init)
    print(f"Greedy build done in {time.time()-t0:.1f}s, S size: {S_size}", flush=True)

    # Local search with swaps
    best_S_size = S_size
    best_s_in = s_in.copy()

    S_pts = [pt for pt in range(total_points) if s_in[pt]]
    non_S_pts = [pt for pt in range(total_points) if not s_in[pt]]

    line_count = np.zeros((total_points, n_dirs), dtype=np.int32)
    for pt in range(total_points):
        for d in range(n_dirs):
            line_count[pt, d] = np.sum(s_in[other_pts[pt, d, :]])
    free_count = np.sum(line_count == 0, axis=1)

    print("Starting local search...", flush=True)
    t0 = time.time()
    no_improve = 0
    max_no_improve = 50

    while time.time() - start_time < timeout - 20 and no_improve < max_no_improve:
        improved = False
        for _ in range(200):
            if time.time() - start_time > timeout - 20:
                break
            if not S_pts or not non_S_pts:
                break
            rm_idx = random.randrange(len(S_pts))
            rm_pt = S_pts[rm_idx]
            add_idx = random.randrange(len(non_S_pts))
            add_pt = non_S_pts[add_idx]

            s_in[rm_pt] = False
            for d in range(n_dirs):
                for y in other_pts[rm_pt, d, :]:
                    if line_count[y, d] > 0:
                        line_count[y, d] -= 1
                        if line_count[y, d] == 0:
                            free_count[y] += 1

            valid = True
            if free_count[add_pt] < 1:
                valid = False
            else:
                for d in range(n_dirs):
                    for y in other_pts[add_pt, d, :]:
                        if line_count[y, d] == 0 and free_count[y] == 1:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                s_in[add_pt] = True
                for d in range(n_dirs):
                    for y in other_pts[add_pt, d, :]:
                        if line_count[y, d] == 0:
                            free_count[y] -= 1
                        line_count[y, d] += 1
                S_pts[rm_idx] = add_pt
                non_S_pts[add_idx] = rm_pt
                improved = True
                no_improve = 0
            else:
                s_in[rm_pt] = True
                for d in range(n_dirs):
                    for y in other_pts[rm_pt, d, :]:
                        if line_count[y, d] == 0:
                            free_count[y] -= 1
                        line_count[y, d] += 1

        if not improved:
            no_improve += 1
        if S_size > best_S_size:
            best_S_size = S_size
            best_s_in = s_in.copy()

    print(f"Local search done in {time.time()-t0:.1f}s, best S size: {best_S_size}", flush=True)

    # Final add phase
    t0 = time.time()
    s_in = best_s_in.copy()
    line_count = np.zeros((total_points, n_dirs), dtype=np.int32)
    for pt in range(total_points):
        for d in range(n_dirs):
            line_count[pt, d] = np.sum(s_in[other_pts[pt, d, :]])
    free_count = np.sum(line_count == 0, axis=1)
    S_size = int(np.sum(s_in))

    candidates = [pt for pt in range(total_points) if not s_in[pt] and free_count[pt] >= 1]
    while candidates and time.time() - start_time < timeout - 10:
        pt = random.choice(candidates)
        valid = True
        for d in range(n_dirs):
            for y in other_pts[pt, d, :]:
                if line_count[y, d] == 0 and free_count[y] == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            s_in[pt] = True
            S_size += 1
            for d in range(n_dirs):
                for y in other_pts[pt, d, :]:
                    if line_count[y, d] == 0:
                        free_count[y] -= 1
                    line_count[y, d] += 1
            try:
                candidates.remove(pt)
            except:
                pass
            if S_size > best_S_size:
                best_S_size = S_size
                best_s_in = s_in.copy()
        else:
            try:
                candidates.remove(pt)
            except:
                pass

    print(f"Final add phase done in {time.time()-t0:.1f}s, best S size: {best_S_size}", flush=True)

    points = []
    for pt in range(total_points):
        if not best_s_in[pt]:
            xa = pt // (q * p)
            xb = (pt // q) % p
            ya = (pt // p) % p
            yb = pt % p
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
