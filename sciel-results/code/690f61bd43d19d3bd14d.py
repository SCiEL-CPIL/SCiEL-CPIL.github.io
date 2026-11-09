import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s

    # Pre-compute indices for upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)  # 28 pairs

    def compute_score(pts):
        """Compute the actual score: prod(d_ij) / D^n where D is diameter, scaled to D=2."""
        # Actually score = Delta / n^n where Delta = prod over ordered pairs of distances after scaling to diameter 2
        # After scaling to diameter 2, distances are d_ij * (2/D)
        # Product over ordered pairs = (product over unordered pairs)^2 since each pair counted twice
        # Wait, let me re-read: "product over ordered pairs of distances after scaling the set to diameter 2"
        # Ordered pairs (i,j) with i != j: there are n*(n-1) = 56 such pairs
        # But typically "product of distances" means over all unordered pairs. Let me check the regular octagon.
        # For regular octagon with diameter 2, the score is exactly 1.
        # Let me compute: regular octagon inscribed in circle of radius 1 (diameter 2).
        # Vertices at angles k*pi/4, k=0..7.
        # Distances: chord length for angle theta is 2*sin(theta/2) for radius 1... wait, radius is 1 since diameter is 2.
        # Chord length = 2*1*sin(theta/2) = 2*sin(theta/2)
        # Pairs: for regular octagon, distances are 2*sin(k*pi/8) for k=1,2,3 (k=4 is diameter=2)
        # Number of pairs at each distance:
        # k=1: 8 pairs, distance = 2*sin(pi/8)
        # k=2: 8 pairs, distance = 2*sin(2*pi/8) = 2*sin(pi/4)
        # k=3: 8 pairs, distance = 2*sin(3*pi/8)
        # k=4: 4 pairs, distance = 2*sin(4*pi/8) = 2
        # Product over unordered pairs = (2*sin(pi/8))^8 * (2*sin(pi/4))^8 * (2*sin(3*pi/8))^8 * 2^4
        # Product over ordered pairs = (product over unordered)^2
        # Delta = product over ordered pairs = (product over unordered)^2
        # Score = Delta / n^n = Delta / 8^8
        # For regular octagon, score should be 1.
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
        D = np.max(dists)
        if D < 1e-15:
            return 0.0
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / D)
        # Product over ordered pairs (i != j)
        mask = ~np.eye(n, dtype=bool)
        prod_ordered = np.prod(scaled_dists[mask])
        score = prod_ordered / (n ** n)
        return score

    def neg_log_score(params):
        """
        Objective to minimize: -log(score) = -log(prod_ordered) + n*log(n)
        = -sum_{i!=j} log(d_ij * 2/D) + n*log(n)
        = -sum_{i!=j} [log(d_ij) + log(2) - log(D)] + n*log(n)
        = -sum_{i!=j} log(d_ij) - n*(n-1)*log(2) + n*(n-1)*log(D) + n*log(n)
        
        To simplify, we can work with:
        f = n*(n-1)*log(D) - sum_{i!=j} log(d_ij)
        Since the other terms are constants, minimizing f is equivalent to minimizing -log(score).
        """
        pts = params.reshape(n, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists2 = np.sum(diff * diff, axis=-1)
        
        # Get distances for i != j
        mask = ~np.eye(n, dtype=bool)
        pair_dists2 = dists2[mask]
        
        if np.any(pair_dists2 < 1e-12):
            return 1e10
        
        D2 = np.max(pair_dists2)
        
        try:
            # log(d_ij) = 0.5 * log(d_ij^2)
            log_D = 0.5 * np.log(D2)
            log_sum = 0.5 * np.sum(np.log(pair_dists2))
            # f = n*(n-1)*log(D) - sum log(d_ij) = n*(n-1)*log_D - log_sum
            val = n * (n - 1) * log_D - log_sum
            return val
        except Exception:
            return 1e10

    # Regular octagon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_params = regular_oct.flatten()
    best_score = compute_score(regular_oct)
    
    print(f"Regular octagon score: {best_score:.6f}")

    bounds = [(-2.0, 2.0)] * (2 * n)

    # Strategy 1: Differential Evolution with good settings
    try:
        de_res = differential_evolution(
            neg_log_score,
            bounds=bounds,
            seed=42,
            maxiter=80,
            tol=1e-10,
            popsize=20,
            mutation=(0.5, 1.5),
            recombination=0.7,
            init='latinhypercube',
            polish=False,
            updating='immediate'
        )
        
        current_pts = de_res.x.reshape(n, 2)
        current_score = compute_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_params = de_res.x
            print(f"DE Score: {best_score:.6f}")
            
    except Exception as e:
        print(f"DE failed: {e}")

    # Strategy 2: Local refinement with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            local_res = minimize(
                neg_log_score,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            local_pts = local_res.x.reshape(n, 2)
            local_score = compute_score(local_pts)
            
            if local_score > best_score:
                best_score = local_score
                best_params = local_res.x
                print(f"Local Refinement Score: {best_score:.6f}")
        except Exception as e:
            print(f"Local refinement failed: {e}")

    # Strategy 3: Try dual annealing for global search
    if time.time() < deadline - 20:
        try:
            da_res = dual_annealing(
                neg_log_score,
                bounds=bounds,
                seed=42,
                maxiter=100,
                restarts=5,
                visit=2.62,
                clip=0.2
            )
            
            da_pts = da_res.x.reshape(n, 2)
            da_score = compute_score(da_pts)
            
            if da_score > best_score:
                best_score = da_score
                best_params = da_res.x
                print(f"Dual Annealing Score: {best_score:.6f}")
                
                # Polish the result
                if time.time() < deadline - 5:
                    try:
                        polish_res = minimize(
                            neg_log_score,
                            best_params,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        polish_pts = polish_res.x.reshape(n, 2)
                        polish_score = compute_score(polish_pts)
                        if polish_score > best_score:
                            best_score = polish_score
                            best_params = polish_res.x
                            print(f"Polished DA Score: {best_score:.6f}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"Dual annealing failed: {e}")

    # Strategy 4: Another DE run with different seed
    if time.time() < deadline - 15:
        try:
            de_res2 = differential_evolution(
                neg_log_score,
                bounds=bounds,
                seed=123,
                maxiter=60,
                tol=1e-10,
                popsize=15,
                mutation=(0.4, 1.2),
                recombination=0.6,
                init='soo',
                polish=False,
                updating='immediate'
            )
            
            pts2 = de_res2.x.reshape(n, 2)
            score2 = compute_score(pts2)
            
            if score2 > best_score:
                best_score = score2
                best_params = de_res2.x
                print(f"DE 2nd Attempt Score: {best_score:.6f}")
                
                # Polish
                if time.time() < deadline - 5:
                    try:
                        local_res2 = minimize(
                            neg_log_score,
                            best_params,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        pts3 = local_res2.x.reshape(n, 2)
                        score3 = compute_score(pts3)
                        if score3 > best_score:
                            best_score = score3
                            best_params = local_res2.x
                            print(f"Polished DE 2nd Score: {best_score:.6f}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"DE 2nd attempt failed: {e}")

    # Final polish on the best found
    if time.time() < deadline - 2:
        try:
            final_res = minimize(
                neg_log_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = final_res.x.reshape(n, 2)
            final_score = compute_score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_params = final_res.x
                print(f"Final NM Score: {best_score:.6f}")
        except Exception:
            pass

    final_pts = best_params.reshape(n, 2)
    
    # Ensure distinctness
    dists = np.sqrt(np.sum((final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :])**2, axis=-1))
    min_dist = np.min(dists[i_idx, j_idx])
    if min_dist < 1e-6:
        final_pts = regular_oct
        best_score = 1.0

    # Ensure coordinates are within bounds
    final_pts = np.clip(final_pts, -1e6, 1e6)

    return final_pts.tolist()
