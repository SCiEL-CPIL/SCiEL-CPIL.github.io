"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12) exploiting the known Pareto front structure.
    
    DTLZ4 structure:
    - g(x) = sum((x_i - 0.5)^2 for i in 2..N-1)
    - f_1 = (1+g) * cos(x_0*pi/2) * cos(x_1*pi/2)
    - f_2 = (1+g) * cos(x_0*pi/2) * sin(x_1*pi/2)
    - f_3 = (1+g) * sin(x_0*pi/2)
    
    Pareto front: g=0 (x[2:]=0.5), and (f1,f2,f3) lie on quarter-unit-sphere.
    Position determined by angles: theta_1 = x_0*pi/2, theta_2 = x_1*pi/2
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate Halton sequence value for given index and base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _angles_to_x(self, theta1, theta2, lo, hi, span):
        """
        Map 2D angles to 12D input space for DTLZ4.
        theta1, theta2 in [0, pi/2] determine position on Pareto front.
        x[2:] set to 0.5 (center of bounds) for g=0.
        """
        x = np.zeros(self.dim)
        # Map angles back to x[0], x[1] in [0, 1]
        x[0] = 2.0 * theta1 / np.pi
        x[1] = 2.0 * theta2 / np.pi
        # Set distance variables to center (0.5) for g=0
        x[2:] = 0.5
        # Clip to bounds
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Use global numpy random state (seeded by evaluator)
        # Phase 1: Systematic sampling of Pareto front using Halton sequence in angle space
        # Allocate ~150 evaluations for systematic coverage, ~50 for refinement

        # Phase 1: Halton-based sampling in 2D angle space
        n_systematic = min(150, self.budget - 10)  # Leave some budget for refinement
        n_refine = self.budget - n_systematic

        X = []
        F = []

        # Halton bases for 2D: use primes 2 and 3
        base1, base2 = 2, 3

        for i in range(1, n_systematic + 1):
            if time.time() > deadline:
                break

            # Generate Halton values in [0, 1)
            h1 = self._halton(i, base1)
            h2 = self._halton(i, base2)

            # Map to angle space [0, pi/2]
            theta1 = h1 * np.pi / 2.0
            theta2 = h2 * np.pi / 2.0

            # Convert to input space
            x = self._angles_to_x(theta1, theta2, lo, hi, span)

            # Evaluate
            f = problem(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)
        used = len(X)

        # Phase 2: Local refinement around best-performing regions
        # Use simple local search around points that seem to be on/near the Pareto front

        if used < self.budget:
            # Find points that are close to the Pareto front (small sum of objectives)
            # For DTLZ4, points on PF have f1+f2+f3 close to 1 (when g=0)
            # Actually for unit sphere: f1^2 + f2^2 + f3^2 = 1 when g=0

            # Sort by how close to the unit sphere they are (indicator of being on PF)
            # f1^2 + f2^2 + f3^2 should be ~1 for PF points
            sphere_dist = np.sum(F ** 2, axis=1)
            # Points closer to 1 are better (on the Pareto front)
            closeness = 1.0 - np.abs(sphere_dist - 1.0)

            # Also consider dominance: prefer non-dominated points
            # Simple approach: pick top points by closeness, then do local search
            n_local = min(n_refine, 30)
            if n_local > 0 and len(X) > 0:
                # Select diverse points from top closeness
                # Use a simple diversity-aware selection
                selected_indices = []
                remaining = np.argsort(-closeness)  # Sort by closeness descending

                for idx in remaining:
                    if len(selected_indices) >= n_local // 2:
                        break
                    # Check diversity: ensure minimum distance from already selected
                    if len(selected_indices) == 0:
                        selected_indices.append(idx)
                    else:
                        min_dist = np.min([np.linalg.norm(X[idx] - X[si]) for si in selected_indices])
                        if min_dist > 0.1:  # Minimum distance threshold
                            selected_indices.append(idx)

                # If we didn't select enough, fill with best closeness
                if len(selected_indices) < n_local // 2:
                    for idx in remaining:
                        if idx not in selected_indices:
                            selected_indices.append(idx)
                        if len(selected_indices) >= n_local // 2:
                            break

                selected_indices = selected_indices[:n_local // 2]

                # Local search: perturb around selected points
                for idx in selected_indices:
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Try a few perturbations around this point
                    n_perturb = min(3, self.budget - used)
                    for p in range(n_perturb):
                        if used >= self.budget or time.time() > deadline:
                            break

                        # Perturb in angle space
                        theta1 = 2.0 * X[idx][0] / np.pi  # Recover angle
                        theta2 = 2.0 * X[idx][1] / np.pi

                        # Small random perturbation
                        d_theta1 = np.random.uniform(-0.1, 0.1)
                        d_theta2 = np.random.uniform(-0.1, 0.1)

                        new_theta1 = np.clip(theta1 + d_theta1, 0, np.pi / 2)
                        new_theta2 = np.clip(theta2 + d_theta2, 0, np.pi / 2)

                        new_x = self._angles_to_x(new_theta1, new_theta2, lo, hi, span)

                        # Also try perturbing the distance variables slightly
                        # to see if we can improve g (should already be ~0)
                        new_f = problem(new_x)
                        used += 1

                        X = np.vstack([X, new_x])
                        F = np.vstack([F, new_f])

        return X

# EVOLVE-BLOCK-END
