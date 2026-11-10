# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # State variables
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    good_line_count = np.zeros(q2, dtype=np.int64)

    # Initialize good_line_count for S = empty
    # All lines have count 0.
    # For z not in S, good if count == 0. All lines are good.
    # Count of lines through any point is q + 1 = 50.
    good_line_count[:] = q + 1

    # Precompute line points as lists for faster iteration in Python
    line_points_list = [line_points[i].tolist() for i in range(n_lines)]
    point_lines_list = [point_lines[i] for i in range(q2)]

    def compute_invalid_count():
        return int(np.sum(good_line_count == 0))

    def get_energy(invalid_count, s_size):
        # We want to minimize invalid_count, then maximize s_size
        # Energy = invalid_count * 10000 - s_size
        return invalid_count * 10000 - s_size

    def unsafe_add_point(z):
        """Adds z to S. Returns list of changes to revert."""
        S[z] = True
        changes = []
        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))
        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 1:
                z_good += 1
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        return changes

    def unsafe_remove_point(z):
        """Removes z from S. Returns list of changes to revert."""
        S[z] = False
        changes = []
        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))
        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 0:
                z_good += 1
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        return changes

    def revert_changes(changes):
        for w, delta in changes:
            good_line_count[w] -= delta

    # Greedy initialization function
    def greedy_init(seed):
        S[:] = False
        line_S_count[:] = 0
        good_line_count[:] = q + 1
        
        rng = np.random.default_rng(seed)
        points_to_try = np.arange(q2)
        for _pass in range(10):
            if _time_left() <= 0:
                break
            rng.shuffle(points_to_try)
            added_in_pass = 0
            for z in points_to_try:
                if _time_left() <= 0:
                    break
                if not S[z]:
                    # Try to add
                    changes = unsafe_add_point(z)
                    # Check validity
                    if np.all(good_line_count > 0):
                        added_in_pass += 1
                    else:
                        # Revert
                        S[z] = False
                        for li in point_lines_list[z]:
                            line_S_count[li] -= 1
                        revert_changes(changes)
            if added_in_pass == 0:
                break
        return int(np.sum(S))

    # Run multiple greedy inits
    best_s_size = 0
    for seed in range(20):
        if _time_left() <= 10:
            break
        size = greedy_init(seed)
        if size > best_s_size:
            best_s_size = size
            # Save state
            # (State is already in S, line_S_count, good_line_count)
            # We need to keep the best one. Let's copy.
            best_S = S.copy()
            best_line_S_count = line_S_count.copy()
            best_good_line_count = good_line_count.copy()

    # Restore best state
    S[:] = best_S
    line_S_count[:] = best_line_S_count
    good_line_count[:] = best_good_line_count

    # SA with swap moves
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    current_invalid = compute_invalid_count()
    current_s_size = int(np.sum(S))
    current_energy = get_energy(current_invalid, current_s_size)
    
    best_s_size = current_s_size
    best_S = S.copy()
    best_line_S_count = line_S_count.copy()
    best_good_line_count = good_line_count.copy()

    rng_sa = np.random.default_rng(123)
    
    # Precompute list of indices for random selection
    indices = np.arange(q2)
    
    while _time_left() > 5.0:
        if T < min_T:
            T = min_T

        # Pick u in S, v not in S
        # To speed up, we can just pick two random points and check
        u = int(rng_sa.integers(0, q2))
        v = int(rng_sa.integers(0, q2))
        
        if u == v:
            continue
        if not S[u] or S[v]:
            continue

        # Perform swap: remove u, add v
        changes_remove = unsafe_remove_point(u)
        changes_add = unsafe_add_point(v)
        
        new_invalid = compute_invalid_count()
        new_s_size = int(np.sum(S))
        new_energy = get_energy(new_invalid, new_s_size)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng_sa.random() < math.exp(-delta_E / T):
            # Accept
            current_invalid = new_invalid
            current_s_size = new_s_size
            current_energy = new_energy
            
            if current_invalid == 0 and current_s_size > best_s_size:
                best_s_size = current_s_size
                best_S = S.copy()
                best_line_S_count = line_S_count.copy()
                best_good_line_count = good_line_count.copy()
        else:
            # Revert: remove v, add u
            revert_changes(changes_add)
            # Note: unsafe_add_point sets S[v]=True. We need to set it back to False?
            # unsafe_add_point does S[z]=True.
            # unsafe_remove_point does S[z]=False.
            # So after revert_changes(changes_add), S[v] is still True?
            # No, revert_changes only reverts good_line_count.
            # We need to manually revert S and line_S_count.
            
            # This is getting messy. Let's use a simpler revert for swaps.
            # Actually, let's just implement a dedicated swap revert.
            
            # Revert add v
            S[v] = False
            for li in point_lines_list[v]:
                line_S_count[li] -= 1
            # Revert good_line_count changes from add v
            revert_changes(changes_add)
            
            # Revert remove u
            S[u] = True
            for li in point_lines_list[u]:
                line_S_count[li] += 1
            # Revert good_line_count changes from remove u
            revert_changes(changes_remove)
            
            # Wait, the order of reversion must be exact reverse.
            # Forward: Remove u (changes_remove), Add v (changes_add)
            # Reverse: Remove v (revert add v), Add u (revert remove u)
            # "Remove v" means undo "Add v".
            # "Add u" means undo "Remove u".
            
            # My manual revert above:
            # 1. S[v]=False, line_S_count[lines(v)] -= 1. This undoes S[v]=True and line_S_count += 1.
            # 2. revert_changes(changes_add). This undoes good_line_count changes.
            # 3. S[u]=True, line_S_count[lines(u)] += 1. This undoes S[u]=False and line_S_count -= 1.
            # 4. revert_changes(changes_remove). This undoes good_line_count changes.
            
            # This looks correct.

        T *= cooling

    # Final Greedy Polish
    S[:] = best_S
    line_S_count[:] = best_line_S_count
    good_line_count[:] = best_good_line_count

    for _pass in range(50):
        if _time_left() <= 0:
            break
        improved = False
        for z in range(q2):
            if _time_left() <= 0:
                break
            if not S[z]:
                changes = unsafe_add_point(z)
                if np.all(good_line_count > 0):
                    improved = True
                else:
                    S[z] = False
                    for li in point_lines_list[z]:
                        line_S_count[li] -= 1
                    revert_changes(changes)
        if not improved:
            break

    # Build output
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
