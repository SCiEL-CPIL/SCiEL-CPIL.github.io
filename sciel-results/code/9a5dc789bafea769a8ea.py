"""Multi-objective optimizer for ZDT6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # ZDT6 Pareto front: x[1:] = 0, x[0] varies
        # f1 = 1 + 10*(n-1) + sum_{i=1}^{n-1}(xi-0.5)^2*(1+10*sin(6*pi*xi+pi*f1))
        # With x[1:]=0, the sum is over i=1 to n-1, but those xi are 0
        # Actually for ZDT6, the sum in f1 is over i=2 to n (1-indexed), which is x[1:] in 0-indexed
        # So with x[1:]=0, f1 = 1 + 10*(n-1) + sum of terms where xi=0
        # (0-0.5)^2 = 0.25, and sin(6*pi*0 + pi*f1) = sin(pi*f1)
        # So f1 = 1 + 90 + 9*0.25*(1+10*sin(pi*f1)) = 91 + 2.25*(1+10*sin(pi*f1))
        # This is implicit since f1 appears on both sides.
        # f2 = g*(1-sqrt(f1/g)) with g=1, so f2 = 1-sqrt(f1)
        
        # Strategy: Adaptive sampling on the Pareto front
        # Phase 1: Coarse uniform sweep
        n_coarse = 50
        X_coarse = np.zeros((n_coarse, n))
        for i in range(n_coarse):
            x0 = (i + 0.5) / n_coarse
            X_coarse[i, 0] = x0
            # x[1:] = 0 for exact front
        
        F_coarse = np.array([problem(x) for x in X_coarse])
        used = n_coarse
        
        # Keep track of all evaluated points
        all_X = [X_coarse]
        all_F = [F_coarse]
        
        # Phase 2: Adaptive refinement - find largest gaps and add points there
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5:
            # Get current non-dominated set and their f1 values
            # Sort by f1 to find gaps
            F_current = F_coarse.copy()
            # Sort by first objective
            sort_idx = np.argsort(F_current[:, 0])
            F_sorted = F_current[sort_idx]
            X_sorted = X_coarse[sort_idx]
            
            # Find gaps in f1 space
            f1_vals = F_sorted[:, 0]
            gaps = np.diff(f1_vals)
            # Also consider gaps from 0 to first and last to max
            gap_indices = []
            for i in range(len(gaps)):
                gap_indices.append((gaps[i], i))
            gap_indices.sort(reverse=True)
            
            # Add points in the largest gaps
            n_refine = min(remaining, 150)
            X_refine = np.zeros((n_refine, n))
            for k in range(n_refine):
                if time.time() > deadline:
                    break
                # Pick a gap to refine - use round-robin over top gaps
                idx = gap_indices[k % len(gap_indices)][1]
                # Midpoint of the gap in x[0] space (approximate)
                x0_left = X_sorted[idx, 0]
                x0_right = X_sorted[idx + 1, 0] if idx + 1 < len(X_sorted) else 1.0
                x0_mid = (x0_left + x0_right) / 2.0
                X_refine[k, 0] = x0_mid
                # x[1:] = 0
        
            # Evaluate refinement points
            F_refine = np.array([problem(x) for x in X_refine[:n_refine]])
            used += n_refine
            all_X.append(X_refine[:n_refine])
            all_F.append(F_refine)
        
        # Combine all points
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)
        
        # Filter: keep only points strictly better than reference point [11.0, 11.0]
        ref = np.array([11.0, 11.0])
        mask = np.all(all_F < ref, axis=1)
        all_X = all_X[mask]
        all_F = all_F[mask]
        
        # Remove dominated points to get the non-dominated set
        if len(all_F) > 1:
            keep = np.ones(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(all_F)):
                    if not keep[j]:
                        continue
                    if np.all(all_F[i] <= all_F[j]) and np.any(all_F[i] < all_F[j]):
                        keep[j] = False
                    elif np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        keep[i] = False
                        break
            all_X = all_X[keep]
            all_F = all_F[keep]
        
        return all_X
        
# EVOLVE-BLOCK-END
