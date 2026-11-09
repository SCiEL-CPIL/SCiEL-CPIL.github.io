# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 with
    pairwise disjoint interiors (touching allowed) to maximize the sum of
    their side lengths.

    Strategy:
    1. Generate a known good deterministic packing (3+8 layout) as a safe
       baseline.
    2. Run a multi-start local optimization using gradient-free coordinate
       descent / Nelder-Mead style moves to refine positions and sizes.
    3. Use a smooth surrogate for the non-overlap constraints to guide the
       search, then verify with hard constraints at the end.
    4. Return the best valid configuration found within the time budget.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 580  # 20s safety margin

    def compute_score(centers, sides):
        """Compute the sum of side lengths, return -inf if invalid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return -np.inf
        if np.any(sides <= 0):
            return -np.inf
        # Check boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return -np.inf
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return -np.inf
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i, s_j = sides[i], sides[j]
                t = (s_i + s_j)/2 - tol
                if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                    return -np.inf
        return np.sum(sides)

    def try_improve(centers, sides, max_iter=2000):
        """Local refinement via coordinate descent with small random perturbations."""
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_score = compute_score(best_centers, best_sides)
        if best_score == -np.inf:
            return best_centers, best_sides, best_score

        rng = np.random.default_rng(42)
        step = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Try moving one square in one dimension
            i = rng.integers(0, n)
            dim = rng.integers(0, 2)
            delta = rng.uniform(-step, step)
            new_centers = best_centers.copy()
            new_sides = best_sides.copy()
            new_centers[i, dim] += delta
            # Clip to stay in bounds
            s_i = new_sides[i]
            if dim == 0:
                new_centers[i, 0] = np.clip(new_centers[i, 0], s_i/2, 1 - s_i/2)
            else:
                new_centers[i, 1] = np.clip(new_centers[i, 1], s_i/2, 1 - s_i/2)
            sc = compute_score(new_centers, new_sides)
            if sc > best_score:
                best_centers = new_centers
                best_sides = new_sides
                best_score = sc
            # Try growing/shrinking
            if it % 3 == 0:
                factor = 1.0 + rng.uniform(-0.02, 0.03)
                new_sides2 = best_sides.copy()
                new_sides2[i] *= factor
                if new_sides2[i] > 0 and new_sides2[i] < 1.0:
                    # Adjust center to stay in bounds
                    s_new = new_sides2[i]
                    new_centers2 = best_centers.copy()
                    new_centers2[i, 0] = np.clip(new_centers2[i, 0], s_new/2, 1 - s_new/2)
                    new_centers2[i, 1] = np.clip(new_centers2[i, 1], s_new/2, 1 - s_new/2)
                    sc2 = compute_score(new_centers2, new_sides2)
                    if sc2 > best_score:
                        best_centers = new_centers2
                        best_sides = new_sides2
                        best_score = sc2
        return best_centers, best_sides, best_score

    # --- Baseline: 3+8 packing ---
    # One large square of side 1/2 in a corner, 8 smaller squares around it.
    # Known optimal-ish layout for 10 squares.
    def baseline_3plus8():
        """
        Layout:
        - Square 0: side 0.5, centered at (0.25, 0.25) -> [0, 0.5] x [0, 0.5]
        - Remaining 8 squares fill the rest.
        The right half [0.5, 1] x [0, 1] and top strip [0, 0.5] x [0.5, 1]
        are available.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Large square
        sides[0] = 0.5
        centers[0] = [0.25, 0.25]
        # Right half: 4 squares of side 0.25
        for i in range(4):
            sides[1+i] = 0.25
            cx = 0.625 + (i % 2) * 0.25
            cy = 0.125 + (i // 2) * 0.25
            centers[1+i] = [cx, cy]
        # Top strip: [0, 0.5] x [0.5, 1], 4 squares of side 0.25
        for i in range(4):
            sides[5+i] = 0.25
            cx = 0.125 + (i % 2) * 0.25
            cy = 0.625 + (i // 2) * 0.25
            centers[5+i] = [cx, cy]
        return centers, sides

    centers, sides = baseline_3plus8()
    best_score = compute_score(centers, sides)
    best_centers = centers.copy()
    best_sides = sides.copy()

    # Multi-start local search
    rng = np.random.default_rng(123)
    n_starts = 50
    for start in range(n_starts):
        if time.time() > deadline - 10:
            break
        # Perturb the baseline
        c = best_centers.copy()
        s = best_sides.copy()
        noise_c = rng.normal(0, 0.02, size=(n, 2))
        noise_s = rng.normal(0, 0.01, size=n)
        c += noise_c
        s += noise_s
        # Clip to valid ranges
        for i in range(n):
            s[i] = np.clip(s[i], 0.05, 0.5)
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)
        sc = compute_score(c, s)
        if sc != -np.inf:
            c2, s2, sc2 = try_improve(c, s, max_iter=500)
            if sc2 > best_score:
                best_score = sc2
                best_centers = c2
                best_sides = s2

    # Also try a grid-based start
    for grid_k in [3, 4]:
        if time.time() > deadline - 10:
            break
        side = 1.0 / grid_k
        idx = np.arange(n)
        cols = idx % grid_k
        rows = idx // grid_k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        c = np.stack([cx, cy], axis=1).astype(np.float64)
        s = np.full(n, side, dtype=np.float64)
        sc = compute_score(c, s)
        if sc != -np.inf:
            c2, s2, sc2 = try_improve(c, s, max_iter=500)
            if sc2 > best_score:
                best_score = sc2
                best_centers = c2
                best_sides = s2

    # Final polish: very fine coordinate descent
    if time.time() < deadline - 5:
        best_centers, best_sides, best_score = try_improve(
            best_centers, best_sides, max_iter=2000)

    # Ensure validity
    final_score = compute_score(best_centers, best_sides)
    if final_score == -np.inf:
        # Fall back to baseline
        best_centers, best_sides = baseline_3plus8()
        final_score = compute_score(best_centers, best_sides)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
