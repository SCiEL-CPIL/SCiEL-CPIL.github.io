"""Deterministic Halton-based global search with local refinement for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def halton_sequence(n, dim, base=None):
    """Generate n points from a Halton sequence in [0,1]^dim."""
    if base is None:
        # First 'dim' primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        base = primes[:dim]
    
    points = np.zeros((n, dim))
    for d in range(dim):
        b = base[d]
        f = 1.0 / b
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                i_val, remainder = divmod(i_val, b)
                points[i, d] += f * remainder
                f /= b
    return points


class Optimizer:
    """Deterministic Halton-based global search with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Halton sequence for global exploration (60% of budget)
        n_coarse = int(budget * 0.6)
        # Generate Halton points in [0,1]^dim and map to [-5,5]^dim
        halton_pts = halton_sequence(n_coarse, dim)
        # Map to [-5, 5]^dim
        halton_pts = halton_pts * 10.0 - 5.0

        for pt in halton_pts:
            if call_count >= budget or time.time() > deadline - 5.0:
                break
            safe_func(pt)

        # Phase 2: Medium-resolution Halton around best region (25% of budget)
        n_medium = int(budget * 0.25)
        # Center around best_x with radius ~2.0
        radius = 2.5
        halton_pts2 = halton_sequence(n_medium, dim)
        # Map to [best_x - radius, best_x + radius]
        halton_pts2 = best_x + (halton_pts2 * 2.0 - 1.0) * radius
        # Clip to box
        halton_pts2 = np.clip(halton_pts2, -5.0, 5.0)

        for pt in halton_pts2:
            if call_count >= budget or time.time() > deadline - 5.0:
                break
            safe_func(pt)

        # Phase 3: Fine-resolution Halton around best region (10% of budget)
        n_fine = int(budget * 0.10)
        radius2 = 0.8
        halton_pts3 = halton_sequence(n_fine, dim)
        halton_pts3 = best_x + (halton_pts3 * 2.0 - 1.0) * radius2
        halton_pts3 = np.clip(halton_pts3, -5.0, 5.0)

        for pt in halton_pts3:
            if call_count >= budget or time.time() > deadline - 5.0:
                break
            safe_func(pt)

        # Phase 4: Local refinement using a simple Nelder-Mead-like approach
        # Use remaining budget for local search
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 3.0:
            # Simple local refinement: perturb best_x in small steps
            # Use a deterministic pattern of perturbations
            rng = np.random.default_rng(123)
            sigma_local = 0.3
            for _ in range(min(remaining, 50)):
                if call_count >= budget or time.time() > deadline - 3.0:
                    break
                # Generate a few perturbation directions
                for _ in range(3):
                    if call_count >= budget or time.time() > deadline - 3.0:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction) + 1e-12
                    x_trial = best_x + sigma_local * direction
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    fx = safe_func(x_trial)
                    if fx < best_f:
                        # Move in this direction
                        sigma_local *= 1.1
                    else:
                        sigma_local *= 0.9
                if sigma_local < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
