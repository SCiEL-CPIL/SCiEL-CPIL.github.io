# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices (flattened)
    XY_diff = (X - Y + (n - 1)).flatten()   # 0..18
    X_flat = X.flatten()                     # 0..9
    XY_sum = (X + Y).flatten()               # 0..18
    X2Y = (X + 2 * Y).flatten()              # 0..27

    # Precompute bin membership masks
    def get_bin_masks(bin_indices, n_bins):
        """Return a list of boolean masks, one per bin."""
        masks = []
        for b in range(n_bins):
            mask = (bin_indices == b)
            masks.append(mask)
        return masks

    diff_masks = get_bin_masks(XY_diff, 2 * n - 1)
    x_masks = get_bin_masks(X_flat, n)
    sum_masks = get_bin_masks(XY_sum, 2 * n - 1)
    x2y_masks = get_bin_masks(X2Y, 3 * n - 1)

    def compute_ratio(P):
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        Pf = P.flatten()

        def entropy_from_masks(masks):
            total_entropy = 0.0
            for mask in masks:
                p = Pf[mask].sum()
                if p > 1e-15:
                    total_entropy -= p * np.log2(p)
            return total_entropy

        H_diff = entropy_from_masks(diff_masks)
        H_x = entropy_from_masks(x_masks)
        H_sum = entropy_from_masks(sum_masks)
        H_2y = entropy_from_masks(x2y_masks)

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
        for i in range(30):
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

        # 9. Weight by exp(-delta * (x+y-9)^2) - concentrate near anti-diagonal
        for delta in [0.1, 0.3, 0.5, 1.0, 2.0]:
            P_d = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_d[y, x] = np.exp(-delta * (x + y - (n - 1)) ** 2)
            P_d /= P_d.sum()
            starts.append(P_d.copy())

        # 10. Line patterns: concentrate on specific lines
        # Lines x = c for c in 0..9
        for c in range(n):
            P_line = np.zeros((n, n))
            for y in range(n):
                P_line[y, c] = 1.0 / n
            P_line /= P_line.sum()
            starts.append(P_line.copy())

        # Lines y = c for c in 0..9
        for c in range(n):
            P_line = np.zeros((n, n))
            for x in range(n):
                P_line[c, x] = 1.0 / n
            P_line /= P_line.sum()
            starts.append(P_line.copy())

        # Lines x+y=c for c in 0..18
        for c in range(2 * n - 1):
            P_line = np.zeros((n, n))
            count = 0
            for y in range(n):
                x = c - y
                if 0 <= x < n:
                    P_line[y, x] = 1.0
                    count += 1
            if count > 0:
                P_line /= P_line.sum()
                starts.append(P_line.copy())

        # Lines x-y=c for c in -9..9
        for c in range(-(n - 1), n):
            P_line = np.zeros((n, n))
            count = 0
            for y in range(n):
                x = c + y
                if 0 <= x < n:
                    P_line[y, x] = 1.0
                    count += 1
            if count > 0:
                P_line /= P_line.sum()
                starts.append(P_line.copy())

        return starts

    def epigraph_objective(params):
        """
        Epigraph formulation: minimize -D/M where M is an auxiliary variable
        subject to M >= H_x, M >= H_sum, M >= H_2y.
        params = [log(P_0), ..., log(P_99), M]
        """
        log_P = params[:n * n]
        M_val = params[n * n]
        
        P = np.exp(log_P).reshape(n, n)
        total = P.sum()
        if total < 1e-9:
            return 1e10
        P = P / total
        Pf = P.flatten()

        def entropy_from_masks(masks):
            total_entropy = 0.0
            for mask in masks:
                p = Pf[mask].sum()
                if p > 1e-15:
                    total_entropy -= p * np.log2(p)
            return total_entropy

        H_diff = entropy_from_masks(diff_masks)
        H_x = entropy_from_masks(x_masks)
        H_sum = entropy_from_masks(sum_masks)
        H_2y = entropy_from_masks(x2y_masks)

        # Penalty for M < H_x, M < H_sum, M < H_2y
        penalty = 0.0
        if M_val < H_x:
            penalty += (H_x - M_val) ** 2
        if M_val < H_sum:
            penalty += (H_sum - M_val) ** 2
        if M_val < H_2y:
            penalty += (H_2y - M_val) ** 2

        if M_val <= 1e-9:
            return 1e10

        return -H_diff / M_val + penalty

    def epigraph_constraints(params):
        """
        Constraints for epigraph formulation.
        Returns list of constraint dicts.
        """
        log_P = params[:n * n]
        M_val = params[n * n]
        
        P = np.exp(log_P).reshape(n, n)
        total = P.sum()
        if total < 1e-9:
            return [{'type': 'ineq', 'fun': lambda: -1e10}]
        P = P / total
        Pf = P.flatten()

        def entropy_from_masks(masks):
            total_entropy = 0.0
            for mask in masks:
                p = Pf[mask].sum()
                if p > 1e-15:
                    total_entropy -= p * np.log2(p)
            return total_entropy

        H_x = entropy_from_masks(x_masks)
        H_sum = entropy_from_masks(sum_masks)
        H_2y = entropy_from_masks(x2y_masks)

        constraints = [
            {'type': 'ineq', 'fun': lambda: M_val - H_x},
            {'type': 'ineq', 'fun': lambda: M_val - H_sum},
            {'type': 'ineq', 'fun': lambda: M_val - H_2y},
            {'type': 'ineq', 'fun': lambda: M_val - 1e-9},
        ]
        return constraints

    def slsqp_optimize(P_init, max_iter=200):
        """
        Use SLSQP with epigraph formulation to optimize.
        """
        P_init = np.maximum(P_init, 1e-15)
        log_P_init = np.log(P_init).flatten()
        
        # Initial M estimate
        P_init_norm = P_init / P_init.sum()
        Pf = P_init_norm.flatten()
        
        def entropy_from_masks(masks):
            total_entropy = 0.0
            for mask in masks:
                p = Pf[mask].sum()
                if p > 1e-15:
                    total_entropy -= p * np.log2(p)
            return total_entropy

        H_x = entropy_from_masks(x_masks)
        H_sum = entropy_from_masks(sum_masks)
        H_2y = entropy_from_masks(x2y_masks)
        M_init = max(H_x, H_sum, H_2y) * 1.1 + 1e-6

        params_init = np.concatenate([log_P_init, [M_init]])

        try:
            res = minimize(
                epigraph_objective,
                params_init,
                method='SLSQP',
                constraints=epigraph_constraints,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )

            if hasattr(res, 'x') and res.x is not None:
                log_P_opt = res.x[:n * n]
                M_opt = res.x[n * n]
                P_opt = np.exp(log_P_opt).reshape(n, n)
                P_opt = np.maximum(P_opt, 0.0)
                total = P_opt.sum()
                if total > 1e-9:
                    P_opt /= total
                    return P_opt
        except Exception:
            pass

        return P_init

    def nelder_mead_polish(P_init, max_iter=100):
        """
        Nelder-Mead polish on the direct ratio objective.
        """
        P_init = np.maximum(P_init, 1e-15)
        
        def direct_obj(params):
            P = np.exp(params).reshape(n, n)
            total = P.sum()
            if total < 1e-9:
                return 1e10
            P = P / total
            r = compute_ratio(P)
            return -r

        params_init = np.log(P_init).flatten()
        
        try:
            res = minimize(
                direct_obj,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
            )

            if hasattr(res, 'x') and res.x is not None:
                P_opt = np.exp(res.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0.0)
                total = P_opt.sum()
                if total > 1e-9:
                    P_opt /= total
                    return P_opt
        except Exception:
            pass

        return P_init

    def coordinate_descent(P, max_rounds=100, tol=1e-10):
        """
        Fast coordinate descent: for each cell, try moving mass to/from
        the best partner cell (determined by bin structure).
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
            # Shuffle cell order for better exploration
            order = np.random.permutation(n * n)

            for idx in order:
                if time.time() > deadline - 5:
                    break
                y, x = divmod(idx, n)

                # Find the best partner to exchange with
                # Try a few random partners
                for _ in range(3):
                    y2 = np.random.randint(n)
                    x2 = np.random.randint(n)
                    if y2 == y and x2 == x:
                        continue

                    # Try moving mass from (y2,x2) to (y,x)
                    if P[y2, x2] > 1e-12:
                        best_frac = None
                        best_r = best_ratio
                        for frac in [0.05, 0.15, 0.3, 0.5]:
                            amount = P[y2, x2] * frac
                            P_new = P.copy()
                            P_new[y, x] += amount
                            P_new[y2, x2] -= amount
                            r = compute_ratio(P_new)
                            if r > best_r:
                                best_r = r
                                best_frac = frac

                        if best_frac is not None:
                            amount = P[y2, x2] * best_frac
                            P[y, x] += amount
                            P[y2, x2] -= amount
                            best_ratio = best_r
                            improved = True

                    # Try moving mass from (y,x) to (y2,x2)
                    if P[y, x] > 1e-12:
                        best_frac = None
                        best_r = best_ratio
                        for frac in [0.05, 0.15, 0.3, 0.5]:
                            amount = P[y, x] * frac
                            P_new = P.copy()
                            P_new[y, x] -= amount
                            P_new[y2, x2] += amount
                            r = compute_ratio(P_new)
                            if r > best_r:
                                best_r = r
                                best_frac = frac

                        if best_frac is not None:
                            amount = P[y, x] * best_frac
                            P[y, x] -= amount
                            P[y2, x2] += amount
                            best_ratio = best_r
                            improved = True

            if not improved:
                break

        P = np.maximum(P, 0.0)
        P /= P.sum()
        return P

    starting_points = make_starting_points()

    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # First pass: evaluate all starting points
    for sp in starting_points:
        P = sp.copy()
        P = np.maximum(P, 0.0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Second pass: SLSQP+epigraph from top starting points
    sorted_starts = []
    for sp in starting_points:
        P = sp.copy()
        P = np.maximum(P, 0.0)
        P /= P.sum()
        r = compute_ratio(P)
        sorted_starts.append((r, P))
    sorted_starts.sort(key=lambda x: -x[0])

    # Optimize top 20 starting points with SLSQP+epigraph
    for idx, (r, sp) in enumerate(sorted_starts[:20]):
        if time.time() > deadline - 60:
            break
        P_opt = slsqp_optimize(sp, max_iter=150)
        r_opt = compute_ratio(P_opt)
        if r_opt > best_ratio:
            best_ratio = r_opt
            P_best = P_opt.copy()

    # Third pass: Nelder-Mead polish from best solution
    if time.time() < deadline - 30:
        P_opt = nelder_mead_polish(P_best, max_iter=100)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Fourth pass: coordinate descent polish from best solution
    if time.time() < deadline - 20:
        P_opt = coordinate_descent(P_best, max_rounds=50)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Fifth pass: SLSQP+epigraph from current best
    if time.time() < deadline - 20:
        P_opt = slsqp_optimize(P_best, max_iter=200)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Sixth pass: Nelder-Mead polish again
    if time.time() < deadline - 15:
        P_opt = nelder_mead_polish(P_best, max_iter=150)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Seventh pass: coordinate descent again
    if time.time() < deadline - 10:
        P_opt = coordinate_descent(P_best, max_rounds=100)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Eighth pass: final SLSQP+epigraph
    if time.time() < deadline - 10:
        P_opt = slsqp_optimize(P_best, max_iter=200)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Ninth pass: final Nelder-Mead polish
    if time.time() < deadline - 5:
        P_opt = nelder_mead_polish(P_best, max_iter=200)
        r = compute_ratio(P_opt)
        if r > best_ratio:
            best_ratio = r
            P_best = P_opt.copy()

    # Tenth pass: final coordinate descent polish
    if time.time() < deadline - 5:
        P_opt = coordinate_descent(P_best, max_rounds=200)
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
