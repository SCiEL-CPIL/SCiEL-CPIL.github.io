"""Multi-objective optimizer for RE34: Bayesian Optimization with Expected Hypervolume Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with EHVI acquisition for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Normalize bounds to [0, 1] for GP
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 + 1e-8

        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # 1. Initial Sampling (Latin Hypercube for better coverage)
        n_init = min(20, budget)
        
        # Simple LHS
        lxs = []
        for i in range(n_init):
            x = np.array([(i + rng.random()) / n_init for _ in range(dim)])
            x = lo + x * (hi - lo)
            lxs.append(x)
        
        for x in lxs:
            f = problem(x)
            add(x, f)

        used = len(F)

        # GP Model
        def gp_predict(X_train, Y_train, X_test, noise_var=1e-4, length_scale=0.5, signal_var=1.0):
            """
            Simple GP implementation using RBF kernel.
            Returns mean and variance for each test point.
            """
            n_train = len(X_train)
            n_test = len(X_test)
            
            # Standardize Y for stability? GP is scale invariant if kernel is scaled properly.
            # Let's just use raw Y.
            
            # Kernel matrix K
            def rbf_kernel(X1, X2, ls, sv):
                # X1: (n1, d), X2: (n2, d)
                # dist^2 = sum(x1^2) + sum(x2^2) - 2*x1*x2^T
                sq1 = np.sum(X1**2, axis=1)[:, np.newaxis]
                sq2 = np.sum(X2**2, axis=1)[np.newaxis, :]
                dist_sq = sq1 + sq2 - 2 * np.dot(X1, X2.T)
                dist_sq = np.maximum(dist_sq, 0.0)
                return sv * np.exp(-0.5 * dist_sq / (ls**2))

            K = rbf_kernel(X_train, X_train, length_scale, signal_var)
            K += noise_var * np.eye(n_train)
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n_train)
                L = np.linalg.cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            K_s = rbf_kernel(X_test, X_train, length_scale, signal_var)
            v = np.linalg.solve(L, K_s.T)
            
            mean = v.T @ alpha
            var = signal_var - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-12)
            
            return mean, var

        # Hypervolume Calculation (Exact for 3D, simplified for speed)
        def hypervolume_3d(F, ref):
            """
            Calculate hypervolume for 3 objectives.
            F: (N, 3) array of points.
            ref: (3,) array.
            """
            if len(F) == 0:
                return 0.0
            
            # Filter dominated points? No, HV is defined over the set.
            # But only points better than ref contribute.
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            
            total_hv = 0.0
            # We need to calculate the volume of the union of boxes.
            # For 3D, we can iterate over points sorted by f0.
            # For each point i, the contribution is (ref[0] - F[i][0]) * Area_2D(F[i], F[i+1:])
            # where Area_2D is the area in (f1, f2) plane dominated by F[i] but not by F[j] for j > i (which have larger f0, so worse in obj 0).
            # Wait, if j > i, F[j][0] >= F[i][0]. So F[j] does NOT dominate F[i] in obj 0.
            # So the "shadow" cast by previous points (better in obj 0) is what we need to subtract.
            
            # Let's use the standard algorithm:
            # Sort by x ascending.
            # Keep a list of "active" points in y-z plane.
            # This is complex to implement robustly in a few lines.
            
            # Alternative: Since N is small (< 200), we can use a grid-based approximation or a simpler exact method.
            # For 3D, a common exact method is:
            # HV = sum_{i} (x_{i+1} - x_i) * A_i
            # where x is sorted, and A_i is the 2D hypervolume of the points with index >= i, projected to y-z, 
            # restricted to the box defined by x_i and ref.
            
            # Actually, let's use a simpler O(N^2) or O(N^3) method if N is small enough.
            # N <= 200. O(N^3) = 8M ops. Might be okay if called infrequently.
            # But we call it for every candidate in EHVI.
            # Let's try a faster 2D HV first.
            
            def hv_2d(points, ref_2d):
                """
                Calculate 2D hypervolume.
                points: (M, 2)
                ref_2d: (2,)
                """
                if len(points) == 0:
                    return 0.0
                # Sort by y ascending
                idx = np.argsort(points[:, 1])
                p = points[idx]
                hv = 0.0
                max_x = -np.inf
                for i in range(len(p)):
                    x, y = p[i]
                    if y < ref_2d[1] and x < ref_2d[0]:
                        # Contribution is (ref_x - x) * (y_next - y) ?? No.
                        # Standard 2D HV:
                        # Sort by x.
                        # hv = sum (x_{i+1} - x_i) * (ref_y - y_i)
                        # where x_{N+1} = ref_x.
                        pass
                
                # Let's do it properly.
                idx = np.argsort(points[:, 0])
                p = points[idx]
                hv = 0.0
                max_y = -np.inf
                for i in range(len(p)):
                    x, y = p[i]
                    if x < ref_2d[0]:
                        # The height is max_y - y? No, it's ref_y - min_y_so_far?
                        # The region dominated by point i and not by previous points (which have smaller x)
                        # is (ref_x - x) * (ref_y - y) minus what was already covered?
                        # No, if we sort by x, the points with smaller x are "better" in x.
                        # They cover the region from x to ref_x? No, they cover from their x to ref_x.
                        # So point i (larger x) is dominated in x by previous points.
                        # It only contributes in the y-direction if its y is smaller than the minimum y of all previous points.
                        
                        if y < max_y:
                            # It extends the front in y.
                            # The new area added is (ref_x - x) * (max_y - y) ?
                            # No. The area added is the rectangle from x to ref_x, and y to max_y.
                            # Wait, the "shadow" of previous points covers y < prev_min_y.
                            # So if y < prev_min_y, it adds area (ref_x - x) * (prev_min_y - y).
                            # And then prev_min_y becomes y.
                            pass
                
                # Let's just use a simple grid or a known library if available? No, standard lib only.
                # Let's implement the O(N log N) 2D HV.
                # Sort by x ascending.
                # Initialize max_y = -inf.
                # For each point (x, y):
                #   if y < max_y:
                #      # This point is dominated in y by a previous point with smaller x? No.
                #      # Previous points have smaller x. If they have smaller y, they dominate this point?
                #      # Dominance: p1 dominates p2 if p1 <= p2 and p1 != p2.
                #      # So if we have a point p_prev with x_prev < x and y_prev < y, p_prev dominates p.
                #      # So we only care about the "Pareto front" of the processed points.
                #      # The Pareto front in 2D is a curve where x increases and y decreases.
                #      # We can maintain a list of (x, y) on the front.
                #      # But for HV, we can just integrate.
                
                # Correct 2D HV Algorithm:
                # 1. Filter points: x < ref_x and y < ref_y.
                # 2. Sort by x ascending.
                # 3. Iterate. Keep track of the minimum y seen so far (min_y).
                # 4. For each point (x, y):
                #    If y < min_y:
                #       The contribution is (ref_x - x) * (min_y - y).
                #       Wait, this assumes the region from x to ref_x and y to min_y is free.
                #       Is it? The previous points have x' < x. They cover regions from x' to ref_x.
                #       So the region (x, ref_x) x (y, min_y) is indeed not covered by previous points
                #       because previous points have x' < x, so their "box" starts at x'.
                #       However, the "box" of a point (x', y') is (x', ref_x) x (y', ref_y).
                #       So the union of these boxes...
                #       Actually, the standard formula is:
                #       HV = sum_{i} (x_{i+1} - x_i) * (ref_y - y_i)
                #       where x_{N+1} = ref_x, and the points are sorted by x.
                #       This works if we only consider the Pareto front.
                #       Let's extract the Pareto front first.
                
                pf = []
                # Sort by x
                idx = np.argsort(points[:, 0])
                p = points[idx]
                min_y = np.inf
                for i in range(len(p)):
                    x, y = p[i]
                    if y < min_y:
                        pf.append([x, y])
                        min_y = y
                
                # Now calculate HV from the front
                hv = 0.0
                for i in range(len(pf)):
                    x, y = pf[i]
                    x_next = pf[i+1][0] if i + 1 < len(pf) else ref_2d[0]
                    # The width is x_next - x
                    # The height is ref_y - y
                    # But wait, this formula assumes the front is a step function.
                    # The area under the step function from x to x_next is (x_next - x) * (ref_y - y).
                    # Yes, this is correct for the union of boxes.
                    hv += (x_next - x) * (ref_2d[1] - y)
                
                return hv

            # Back to 3D HV.
            # We can use the property: HV_3D = sum_{i} (x_{i+1} - x_i) * HV_2D(F[i:], ref_2d)
            # where F is sorted by x, and F[i:] are the points with index >= i.
            # But we must be careful: the 2D HV should be calculated for the points in F[i:]
            # restricted to the box defined by y, z of the "current" front?
            # Actually, the standard recursive formula is:
            # Sort by x.
            # HV = sum_{i} (x_{i+1} - x_i) * HV_2D( { (y_j, z_j) | j >= i }, ref_2d )
            # This works if we process the points in order of increasing x.
            # The term HV_2D considers all points with x >= x_i.
            # This is correct because the volume between x_i and x_{i+1} is the cross-section
            # of the dominated region at x in (x_i, x_{i+1}).
            # The cross-section is the 2D dominated region of all points with x <= x.
            # Since x > x_i, the points with x <= x are exactly those with index >= i (if sorted by x ascending? No, x_j >= x_i).
            # Wait, if sorted by x ascending, x_0 < x_1 < ...
            # For x in (x_i, x_{i+1}), the points that dominate in x are those with x_j <= x.
            # These are points with index j such that x_j <= x.
            # Since x > x_i, all points j <= i have x_j <= x_i < x.
            # So the set of points is {0, ..., i}.
            # So the cross-section is the 2D HV of points {0, ..., i}.
            
            # Let's re-verify.
            # Minimization. Smaller x is better.
            # Point p dominates region [x_p, ref_x] x [y_p, ref_y] x [z_p, ref_z].
            # We want the volume of the union of these boxes.
            # Sort points by x ascending: x_0 <= x_1 <= ... <= x_{N-1}.
            # Consider a slice at x. The cross-section is the 2D region dominated by points with x_j <= x.
            # For x in (x_i, x_{i+1}), the set of active points is {0, ..., i}.
            # So the area of the slice is HV_2D( { (y_j, z_j) | j <= i }, ref_2d ).
            # The total HV is the integral of this area from x_0 to ref_x.
            # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * HV_2D( { (y_j, z_j) | j <= i }, ref_2d )
            # where x_N = ref_x.
            # Note: x_0 is the smallest x. The integral starts from x_0?
            # No, the region starts from x_0? No, the boxes start from x_j.
            # The union of boxes starts from min(x_j).
            # The slice for x < x_0 is empty.
            # So the integral is from x_0 to ref_x.
            # The first term is (x_1 - x_0) * HV_2D({0}).
            # The last term is (ref_x - x_{N-1}) * HV_2D({0, ..., N-1}).
            
            # This requires calculating HV_2D for prefixes of the sorted list.
            # N=200. 200 HV_2D calls. Each HV_2D is O(N log N). Total O(N^2 log N).
            # 200 * 200 * 8 = 320k ops. Very fast.
            
            idx = np.argsort(F[:, 0])
            F_sorted = F[idx]
            
            hv = 0.0
            # We need to maintain the 2D HV of the prefix.
            # We can just re-calculate it each time for simplicity, or update incrementally.
            # Re-calculation is fine for N=200.
            
            for i in range(len(F_sorted)):
                x_i = F_sorted[i][0]
                x_next = F_sorted[i+1][0] if i + 1 < len(F_sorted) else ref[0]
                
                # Get points 0..i
                sub_F = F_sorted[:i+1]
                # Extract y, z
                sub_yz = sub_F[:, 1:]
                hv_2d_val = hv_2d(sub_yz, ref[1:])
                
                hv += (x_next - x_i) * hv_2d_val
            
            return hv

        # EHVI Acquisition Function
        def ehvi(X_cand, F_pred_mean, F_pred_var, F_current, ref):
            """
            Calculate Expected Hypervolume Improvement for a candidate point.
            X_cand: (d,)
            F_pred_mean: (n_obj,)
            F_pred_var: (n_obj,)
            F_current: (N, n_obj) current front
            ref: (n_obj,)
            
            Returns: EHVI value (scalar)
            """
            # Sample from the GP posterior
            n_samples = 50
            samples = F_pred_mean + np.sqrt(F_pred_var) * rng.standard_normal((n_samples, n_obj))
            
            # Calculate HV for current front
            hv_current = hypervolume_3d(F_current, ref)
            
            # Calculate HV for each sample
            hv_samples = 0.0
            for s in samples:
                # Add sample to current front
                F_new = np.vstack([F_current, s])
                hv_new = hypervolume_3d(F_new, ref)
                improvement = hv_new - hv_current
                if improvement > 0:
                    hv_samples += improvement
            
            return hv_samples / n_samples

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Prepare training data
            X_train = np.array(X)
            F_train = np.array(F)
            
            # Normalize X_train to [0, 1]
            X_train_norm = (X_train - lo) / (hi - lo)
            
            # Fit GPs for each objective
            means = np.zeros((dim, n_obj)) # Placeholder
            vars = np.zeros((dim, n_obj))
            
            # Actually, we need to evaluate EHVI for a set of candidate points.
            # Let's generate a set of candidates.
            # 1. Local search around Pareto points
            # 2. Global random points
            
            # Identify Pareto points
            pareto_idx = []
            for i in range(len(F_train)):
                dominated = False
                for j in range(len(F_train)):
                    if i != j and np.all(F_train[j] <= F_train[i]) and np.any(F_train[j] < F_train[i]):
                        dominated = True
                        break
                if not dominated:
                    pareto_idx.append(i)
            
            candidates = []
            
            # Add random candidates
            n_random = 10
            for _ in range(n_random):
                x_cand = lo + rng.random(dim) * (hi - lo)
                candidates.append(x_cand)
            
            # Add local candidates around Pareto points
            n_local = 5
            for i in pareto_idx:
                x_base = X[i]
                for _ in range(n_local):
                    # Gaussian perturbation
                    scale = 0.1 * (hi - lo)
                    x_cand = x_base + scale * rng.standard_normal(dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)
            
            # Evaluate EHVI for each candidate
            best_ehvi = -np.inf
            best_x = None
            
            for x_cand in candidates:
                # Normalize
                x_cand_norm = (x_cand - lo) / (hi - lo)
                
                # Predict
                f_mean = np.zeros(n_obj)
                f_var = np.zeros(n_obj)
                
                for k in range(n_obj):
                    m, v = gp_predict(X_train_norm, F_train[:, k], x_cand_norm.reshape(1, -1))
                    f_mean[k] = m[0]
                    f_var[k] = v[0]
                
                # Calculate EHVI
                ehvi_val = ehvi(x_cand, f_mean, f_var, F_train, ref)
                
                if ehvi_val > best_ehvi:
                    best_ehvi = ehvi_val
                    best_x = x_cand
            
            # If no good candidate, pick random
            if best_x is None:
                best_x = lo + rng.random(dim) * (hi - lo)
            
            # Evaluate
            f_new = problem(best_x)
            add(best_x, f_new)
            used += 1

        return X
# EVOLVE-BLOCK-END
