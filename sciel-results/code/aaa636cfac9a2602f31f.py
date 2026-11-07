"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        used = 0
        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fc)
            return fc

        def get_front(F_list):
            """Return indices of non-dominated points in F_list (2D minimization)."""
            n = len(F_list)
            if n == 0:
                return np.array([], dtype=int)
            F = np.array(F_list, dtype=float)
            # Filter points strictly better than reference
            mask = np.all(F < ref, axis=1)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return np.array([], dtype=int)
            F_valid = F[valid_indices]
            
            # Sort by obj0 ascending
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            # Keep only non-dominated (obj1 must be strictly decreasing)
            front_idx_sorted = [0]
            min_obj1 = F_sorted[0][1]
            for i in range(1, len(F_sorted)):
                if F_sorted[i][1] < min_obj1:
                    front_idx_sorted.append(i)
                    min_obj1 = F_sorted[i][1]
            
            # Map back to original indices
            front_indices = valid_indices[order[front_idx_sorted]]
            return front_indices

        def compute_hv_2d(F_list):
            """Compute hypervolume of a set of 2D points w.r.t. reference."""
            if len(F_list) == 0:
                return 0.0
            F = np.array(F_list, dtype=float)
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            # Build non-dominated front
            front = [F_sorted[0]]
            for p in F_sorted[1:]:
                if p[1] < front[-1][1]:
                    front.append(p)
            hv = 0.0
            for i in range(len(front)):
                x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
                y_curr = front[i][1]
                if y_curr >= ref[1]:
                    continue
                width = x_next - front[i][0]
                height = ref[1] - y_curr
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        # Phase 1: Initial exploration with structured sampling
        init_budget = min(20, self.budget // 10)
        
        # Structured initialization: center, corners, and mid-edges
        init_fracs = []
        # Center
        init_fracs.append(np.full(self.dim, 0.5))
        # 4 corners
        corners = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ]
        for c in corners:
            init_fracs.append(np.array(c, dtype=float))
        # 4 mid-edges (vary one pair)
        mid_edges = [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5],
        ]
        for m in mid_edges:
            init_fracs.append(np.array(m, dtype=float))
        # Additional structured points
        extra_structured = [
            [0.25, 0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75, 0.75],
            [0.25, 0.75, 0.25, 0.75],
            [0.75, 0.25, 0.75, 0.25],
            [0.5, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0, 0.5],
            [1.0, 0.5, 0.0, 0.5],
        ]
        for e in extra_structured:
            init_fracs.append(np.array(e, dtype=float))
        
        # Shuffle and take init_budget
        rng.shuffle(init_fracs)
        init_fracs = init_fracs[:init_budget]
        
        for frac in init_fracs:
            if time.time() > deadline:
                break
            x = lo + frac * (hi - lo)
            evaluate(x)

        # Phase 2: Main optimization loop
        F_mut = 0.5
        CR = 0.4
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current front
            front_idx = get_front(all_F)
            if len(front_idx) == 0:
                # Fallback: random exploration
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            # Selection: prefer points on the front, weighted by their HV contribution
            # Compute HV contribution efficiently: for each front point, the area it "owns"
            F_front = np.array([all_F[i] for i in front_idx], dtype=float)
            order_f = np.argsort(F_front[:, 0])
            F_sorted = F_front[order_f]
            
            # Compute per-point HV contribution (area of rectangle owned)
            contributions = np.zeros(len(F_front))
            for i in range(len(F_sorted)):
                x_left = F_sorted[i][0]
                x_right = F_sorted[i + 1][0] if i + 1 < len(F_sorted) else ref[0]
                y_val = F_sorted[i][1]
                if y_val < ref[1] and x_right > x_left:
                    width = x_right - x_left
                    height = ref[1] - y_val
                    contributions[order_f[i]] = max(0.0, width * height)
            
            # Also include non-front points that are close to the front
            # Use all points for DE variation, but weight front points more
            n_total = len(all_F)
            if n_total < 3:
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            # Build selection weights: front points get high weight, others get lower
            weights = np.zeros(n_total)
            for fi, idx in enumerate(front_idx):
                weights[idx] = contributions[fi] + 1e-6
            # Non-front points get small weight based on how close they are to being non-dominated
            for idx in range(n_total):
                if idx not in set(front_idx):
                    f = all_F[idx]
                    if np.all(f < ref):
                        # Give some weight to near-front points
                        weights[idx] = 0.1
                    else:
                        weights[idx] = 0.01
            
            total_w = weights.sum()
            if total_w > 1e-10:
                probs = weights / total_w
            else:
                probs = np.ones(n_total) / n_total
            
            # DE/rand/1/bin from all points
            parent_indices = rng.choice(n_total, size=3, replace=False, p=probs)
            p1 = all_X[parent_indices[0]]
            p2 = all_X[parent_indices[1]]
            p3 = all_X[parent_indices[2]]
            
            v = p1 + F_mut * (p2 - p3)
            mask = rng.random(self.dim) < CR
            child = np.where(mask, v, p1)
            
            # Adaptive Gaussian perturbation
            progress = used / self.budget
            # Reduce mutation as we progress, but keep minimum exploration
            mut_scale = 0.08 * (1.0 - progress) + 0.01
            child = child + rng.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = np.clip(child, lo, hi)
            
            evaluate(child)
            
            # Periodic random restart for exploration (every 25 evals)
            if used % 25 == 0 and used < self.budget - 1:
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x_rand)

        return all_X
# EVOLVE-BLOCK-END
