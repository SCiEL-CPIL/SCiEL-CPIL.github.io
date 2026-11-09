# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices (flattened)
    XY_diff = (X - Y + (n - 1)).flatten()   # 0..18
    X_flat = X.flatten()                     # 0..9
    XY_sum = (X + Y).flatten()               # 0..18
    X2Y = (X + 2 * Y).flatten()              # 0..27

    def compute_ratio(P):
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        Pf = P.flatten()

        def entropy(indices, minlength):
            dist = np.bincount(indices, weights=Pf, minlength=minlength)
            nz = dist[dist > 1e-15]
            if len(nz) == 0:
                return 0.0
            return -np.sum(nz * np.log2(nz))

        H_diff = entropy(XY_diff, 2 * n - 1)
        H_x = entropy(X_flat, n)
        H_sum = entropy(XY_sum, 2 * n - 1)
        H_2y = entropy(X2Y, 3 * n - 1)

        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 0.0
        return H_diff / M

    def make_starting_points():
        starts = []
        # 1. Uniform on X-Y: P[y,x] ∝ 1/(n-|x-y|)
        P1 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P1[y, x] = 1.0 / (n - abs(x - y))
        P1 /= P1.sum()
        starts.append(P1.copy())

        # 2. Uniform
        P2 = np.ones((n, n)) / (n * n)
        starts.append(P2.copy())

        # 3. Diagonal concentration (x=y)
        P3 = np.zeros((n, n))
        for d in range(n):
            for i in range(n - d):
                P3[i, i + d] = 1.0 / (n - d)
        P3 /= P3.sum()
        starts.append(P3.copy())

        # 4. Anti-diagonal concentration (x+y=9)
        P4 = np.zeros((n, n))
        for s in range(2 * n - 1):
            count = min(s + 1, 2 * n - 1 - s, n)
            for y in range(n):
                x = s - y
                if 0 <= x < n:
                    P4[y, x] = 1.0 / count
        P4 /= P4.sum()
        starts.append(P4.copy())

        # 5. Random Dirichlet starts
        rng = np.random.RandomState(12345)
        for i in range(20):
            P_r = rng.dirichlet(np.ones(n * n)).reshape(n, n)
            starts.append(P_r.copy())

        # 6. Structured: weight by exp(-alpha * |x-y|)
        for alpha in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            P_a = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_a[y, x] = np.exp(-alpha * abs(x - y))
            P_a /= P_a.sum()
            starts.append(P_a.copy())

        # 7. Weight by exp(-beta * (x+y)^2) - concentrate near corners
        for beta in [0.1, 0.3, 0.5, 1.0]:
            P_b = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_b[y, x] = np.exp(-beta * (x + y) ** 2)
            P_b /= P_b.sum()
            starts.append(P_b.copy())

        # 8. Weight by exp(-gamma * (x-y)^2) - concentrate near main diagonal
        for gamma in [0.1, 0.3, 0.5, 1.0, 2.0]:
            P_c = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_c[y, x] = np.exp(-gamma * (x - y) ** 2)
            P_c /= P_c.sum()
            starts.append(P_c.copy())

        return starts

    def coordinate_descent(P, max_rounds=200, step=0.05, tol=1e-10):
        """
        Coordinate descent on the simplex. For each cell, try shifting mass
        from one cell to another (or to/from a uniform baseline) to improve ratio.
        """
        P = P.copy()
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            P = np.ones((n, n)) / (n * n)
            return P
        P = P / total

        best_ratio = compute_ratio(P)

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break

            improved = False
            for y in range(n):
                if time.time() > deadline - 5:
                    break
                for x in range(n):
                    if time.time() > deadline - 5:
                        break
                    # Try to increase P[y,x] by taking mass from a random other cell
                    # Or decrease it by giving mass to another cell
                    # Use a simple scheme: for each pair (y,x) and (y2,x2),
                    # try moving a small fraction.
                    # To keep it fast, only try a few targets per cell.
                    # We'll try moving mass to/from the "weakest" and "strongest" cells.
                    # Simpler: try moving a fixed fraction to/from each other cell is too slow (100*100*steps).
                    # Instead, use a randomized approach: pick a random other cell.
                    
                    # We'll do: for this cell, try to add mass by removing from a random cell.
                    # And try to remove mass by adding to a random cell.
                    for direction in range(2):
                        if direction == 0:
                            # Add mass to (y,x), remove from random (y2,x2)
                            y2 = np.random.randint(n)
                            x2 = np.random.randint(n)
                            if y2 == y and x2 == x:
                                continue
                            if P[y2, x2] <= 1e-12:
                                continue
                            for frac in [0.01, 0.05, 0.1, 0.2, 0.3]:
                                amount = P[y2, x2] * frac
                                P_new = P.copy()
                                P_new[y, x] += amount
                                P_new[y2, x2] -= amount
                                r = compute_ratio(P_new)
                                if r > best_ratio + tol:
                                    P = P_new
                                    best_ratio = r
                                    improved = True
                                    break
                        else:
                            # Remove mass from (y,x), add to random (y2,x2)
                            if P[y, x] <= 1e-12:
                                continue
                            y2 = np.random.randint(n)
                            x2 = np.random.randint(n)
                            if y2 == y and x2 == x:
                                continue
                            for frac in [0.01, 0.05, 0.1, 0.2, 0.3]:
                                amount = P[y, x] * frac
                                P_new = P.copy()
                                P_new[y, x] -= amount
                                P_new[y2, x2] += amount
                                r = compute_ratio(P_new)
                                if r > best_ratio + tol:
                                    P = P_new
                                    best_ratio = r
                                    improved = True
                                    break
            if not improved:
                break

        P = np.maximum(P, 0.0)
        P /= P.sum()
        return P

    def local_search(P, max_iter=500):
        """
        More systematic local search: for each cell, try adjusting its value
        by moving mass to/from a specific set of "partner" cells.
        """
        P = P.copy()
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            P = np.ones((n, n)) / (n * n)
            return P
        P = P / total

        best_ratio = compute_ratio(P)

        # Precompute for each cell, the list of other cells (for efficiency)
        # We'll iterate over cells and try moving mass to/from each other cell
        # but only with a few step sizes to keep it fast.
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            improved = False
            for y in range(n):
                if time.time() > deadline - 5:
                    break
                for x in range(n):
                    if time.time() > deadline - 5:
                        break
                    # For this cell, try to adjust by moving mass to/from each other cell
                    # This is O(n^2 * steps) per cell, which is 100*100*5 = 50000 per cell,
                    # too slow. Instead, we'll do a randomized subset.
                    # Pick a few random partners.
                    for _ in range(5):
                        y2 = np.random.randint(n)
                        x2 = np.random.randint(n)
                        if y2 == y and x2 == x:
                            continue
                        # Try moving mass from (y2,x2) to (y,x)
                        if P[y2, x2] > 1e-12:
                            for frac in [0.02, 0.1, 0.3]:
                                amount = P[y2, x2] * frac
                                P_new = P.copy()
                                P_new[y, x] += amount
                                P_new[y2, x2] -= amount
                                r = compute_ratio(P_new)
                                if r > best_ratio + 1e-10:
                                    P = P_new
                                    best_ratio = r
                                    improved = True
                                    break
                        # Try moving mass from (y,x) to (y2,x2)
                        if P[y, x] > 1e-12:
                            for frac in [0.02, 0.1, 0.3]:
                                amount = P[y, x] * frac
                                P_new = P.copy()
                                P_new[y, x] -= amount
                                P_new[y2, x2] += amount
                                r = compute_ratio(P_new)
                                if r > best_ratio + 1e-10:
                                    P = P_new
                                    best_ratio = r
                                    improved = True
                                    break
            if not improved:
                break

        P = np.maximum(P, 0.0)
        P /= P.sum()
        return P

    starting_points = make_starting_points()

    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # First pass: quick evaluation of all starting points
    for sp in starting_points:
        P = sp.copy()
        P = np.maximum(P, 0.0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Second pass: coordinate descent from each starting point
    for idx, sp in enumerate(starting_points):
        if time.time() > deadline - 15:
            break
        P = sp.copy()
        P = np.maximum(P, 0.0)
        P /= P.sum()
        P_opt = coordinate_descent(P, max_rounds=100)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Third pass: local search from the best solution and a few others
    if time.time() < deadline - 10:
        P_opt = local_search(P_best, max_iter=300)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Fourth pass: coordinate descent from the current best
    if time.time() < deadline - 10:
        P_opt = coordinate_descent(P_best, max_rounds=200)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Fifth pass: local search from the current best
    if time.time() < deadline - 10:
        P_opt = local_search(P_best, max_iter=500)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Final refinement: coordinate descent with more rounds
    if time.time() < deadline - 5:
        P_opt = coordinate_descent(P_best, max_rounds=300)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
# EVOLVE-BLOCK-END
