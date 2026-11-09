# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _lp_pack(sides, orientations, M):
    """
    Given fixed side lengths and binary orientation choices,
    solve the LP to find feasible centers.
    Returns (centers, success).
    """
    n = len(sides)
    # Variables: cx_0..cx_{n-1}, cy_0..cy_{n-1}  => 2n vars
    # Objective: minimize 0 (feasibility)
    c_obj = np.zeros(2 * n)

    A_ub = []
    b_ub = []

    # Boundary constraints:
    # cx_i - s_i/2 >= 0  =>  -cx_i <= -s_i/2
    # cx_i + s_i/2 <= 1  =>   cx_i <= 1 - s_i/2
    # Same for cy
    for i in range(n):
        # -cx_i <= -s_i/2
        row = np.zeros(2 * n)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i] / 2.0)

        # cx_i <= 1 - s_i/2
        row = np.zeros(2 * n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i] / 2.0)

        # -cy_i <= -s_i/2
        row = np.zeros(2 * n)
        row[n + i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i] / 2.0)

        # cy_i <= 1 - s_i/2
        row = np.zeros(2 * n)
        row[n + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i] / 2.0)

    # Disjunctive constraints for each pair (i, j)
    # orientations[(i,j)] in {0,1,2,3}:
    # 0: i left of j  => cx_j - cx_i >= (s_i+s_j)/2
    # 1: i right of j => cx_i - cx_j >= (s_i+s_j)/2
    # 2: i below j    => cy_j - cy_i >= (s_i+s_j)/2
    # 3: i above j    => cy_i - cy_j >= (s_i+s_j)/2
    for i in range(n):
        for j in range(i + 1, n):
            o = orientations[(i, j)]
            sep = (sides[i] + sides[j]) / 2.0
            row = np.zeros(2 * n)
            if o == 0:  # cx_j - cx_i >= sep  =>  -cx_j + cx_i <= -sep
                row[i] = 1.0
                row[j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 1:  # cx_i - cx_j >= sep  =>  -cx_i + cx_j <= -sep
                row[i] = -1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 2:  # cy_j - cy_i >= sep  =>  -cy_j + cy_i <= -sep
                row[n + i] = 1.0
                row[n + j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            else:  # o == 3: cy_i - cy_j >= sep  =>  -cy_i + cy_j <= -sep
                row[n + i] = -1.0
                row[n + j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)

    if not A_ub:
        return np.zeros((n, 2)), True

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (2 * n), method='highs')
    if res.status == 0:
        centers = res.x.reshape(n, 2)
        return centers, True
    return None, False


def _solve_lp_relaxation(M, side_lb=0.01, side_ub=0.5):
    """
    Solve the full LP relaxation with binary variables.
    Variables: cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}, z_{i,j}^0..z_{i,j}^3
    Maximize sum(s_i).
    Returns (centers, sides, orientations, success).
    """
    n = 10
    num_pairs = n * (n - 1) // 2
    # Total vars: 2n (centers) + n (sides) + 4*num_pairs (binaries)
    total_vars = 2 * n + n + 4 * num_pairs

    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c_obj = np.zeros(total_vars)
    for i in range(n):
        c_obj[2 * n + i] = -1.0

    A_ub = []
    b_ub = []

    # Boundary: cx_i - s_i/2 >= 0 => -cx_i - s_i/2 <= 0
    # cx_i + s_i/2 <= 1 => cx_i + s_i/2 <= 1
    # Same for cy
    for i in range(n):
        # -cx_i - 0.5*s_i <= 0
        row = np.zeros(total_vars)
        row[i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)

        # cx_i + 0.5*s_i <= 1
        row = np.zeros(total_vars)
        row[i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

        # -cy_i - 0.5*s_i <= 0
        row = np.zeros(total_vars)
        row[n + i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)

        # cy_i + 0.5*s_i <= 1
        row = np.zeros(total_vars)
        row[n + i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # Disjunctive constraints for each pair
    # For pair (i,j), orientation k:
    # If k=0 (i left of j): cx_j - cx_i - (s_i+s_j)/2 >= 0
    #   => -cx_j + cx_i + 0.5*s_i + 0.5*s_j <= M*(1 - z^0_{i,j})
    # We express as: cx_i - cx_j + 0.5*s_i + 0.5*s_j <= M*(1 - z^0_{i,j})
    # Which is: cx_i - cx_j + 0.5*s_i + 0.5*s_j + M*z^0_{i,j} <= M
    # Similarly for other orientations.
    for i in range(n):
        for j in range(i + 1, n):
            pair_idx = (i * (n - 1) - i * (i + 1) // 2 + j)  # index in pairs
            # Actually, let's use a simpler indexing: enumerate pairs
            # We'll track pair_index separately
            pass

    # Let me re-index properly
    A_ub = []
    b_ub = []

    # Re-add boundary constraints
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(total_vars)
        row[i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

        row = np.zeros(total_vars)
        row[n + i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(total_vars)
        row[n + i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # Disjunctive constraints
    pair_counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            z_base = 2 * n + n + 4 * pair_counter  # start index for z^0..z^3 of this pair
            # Orientation 0: i left of j
            # cx_i - cx_j + 0.5*s_i + 0.5*s_j <= M*(1 - z^0)
            # => cx_i - cx_j + 0.5*s_i + 0.5*s_j + M*z^0 <= M
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 0] = M
            A_ub.append(row)
            b_ub.append(M)

            # Orientation 1: i right of j
            # -cx_i + cx_j + 0.5*s_i + 0.5*s_j <= M*(1 - z^1)
            row = np.zeros(total_vars)
            row[i] = -1.0
            row[j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 1] = M
            A_ub.append(row)
            b_ub.append(M)

            # Orientation 2: i below j
            # cy_i - cy_j + 0.5*s_i + 0.5*s_j <= M*(1 - z^2)
            row = np.zeros(total_vars)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 2] = M
            A_ub.append(row)
            b_ub.append(M)

            # Orientation 3: i above j
            # -cy_i + cy_j + 0.5*s_i + 0.5*s_j <= M*(1 - z^3)
            row = np.zeros(total_vars)
            row[n + i] = -1.0
            row[n + j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 3] = M
            A_ub.append(row)
            b_ub.append(M)

            pair_counter += 1

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    bounds = [(0.0, 1.0)] * (2 * n)  # centers
    bounds += [(side_lb, side_ub)] * n  # sides
    bounds += [(0.0, 1.0)] * (4 * num_pairs)  # binaries (relaxed to [0,1])

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.status != 0:
        return None, None, None, False

    x = res.x
    centers = x[:2 * n].reshape(n, 2)
    sides = x[2 * n:3 * n]
    # Round binaries
    orientations = {}
    pair_counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            z_base = 3 * n + 4 * pair_counter
            z_vals = x[z_base:z_base + 4]
            # Pick the orientation with smallest z (most "active")
            o = int(np.argmin(z_vals))
            orientations[(i, j)] = o
            pair_counter += 1

    return centers, sides, orientations, True


def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True


def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)

    # Try different M values and orientation seeds
    M_values = [1.0, 1.5, 2.0]
    np.random.seed(42)

    for M in M_values:
        if time.time() > deadline:
            break

        # Solve LP relaxation
        centers, sides, orientations, success = _solve_lp_relaxation(M)
        if not success:
            continue

        # Polish: fix orientations, solve for centers
        c_polished, ok = _lp_pack(sides, orientations, M)
        if ok and _is_valid(c_polished, sides):
            update_best(c_polished, sides)

        # Also try without polish (use LP relaxation centers directly)
        if _is_valid(centers, sides):
            update_best(centers, sides)

        # Try perturbing orientations slightly
        for trial in range(3):
            if time.time() > deadline:
                break
            # Randomly flip some orientations
            new_orientations = dict(orientations)
            keys = list(new_orientations.keys())
            for _ in range(len(keys) // 3):
                k = keys[np.random.randint(len(keys))]
                new_orientations[k] = np.random.randint(4)
            c_trial, ok_trial = _lp_pack(sides, new_orientations, M)
            if ok_trial and _is_valid(c_trial, sides):
                update_best(c_trial, sides)

    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s

    return best_centers, best_sides

# EVOLVE-BLOCK-END
