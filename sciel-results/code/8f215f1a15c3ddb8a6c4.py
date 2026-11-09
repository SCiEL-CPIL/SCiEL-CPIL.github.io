"""Optimizer for BBOB f6 (Attractive Sector) using fast-adapting (1+1)-ES."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES optimizer with immediate adaptive step size and minimal global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        calls_used = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Minimal Global Exploration
        # We use a small set of structured points to ensure we are in the basin of attraction.
        # BBOB f6 has a broad basin, so this is mostly a safety net.
        # We limit this to ~10 calls to maximize budget for Phase 2.
        
        # Structured candidates: Origin, axes +/-, and a few randoms
        candidates = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        
        # Add a few random points for robustness
        for _ in range(5):
            candidates.append(rng.uniform(-5.0, 5.0, dim))

        # Shuffle to avoid bias, though origin is often good for f6
        rng.shuffle(candidates)

        for x_cand in candidates:
            if calls_used >= budget or time.time() > deadline:
                break
            x_cand = np.clip(x_cand, -5.0, 5.0)
            fx = float(func(x_cand))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_cand.copy()

        # If no valid point was found (should not happen), fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            calls_used += 1

        # Phase 2: Fast-Adapting (1+1)-ES
        # Based on Inspiration 1's strategy which scored 1.0
        # Immediate adaptation: increase sigma on success, decrease on failure.
        
        x = best_x.copy()
        f = best_f
        sigma = 2.0  # Start with a reasonable step size for dim=5

        while calls_used < budget and time.time() < deadline:
            # Generate offspring
            z = rng.standard_normal(dim)
            x_off = x + sigma * z
            x_off = np.clip(x_off, -5.0, 5.0)

            f_off = float(func(x_off))
            calls_used += 1

            # Immediate Adaptation
            if f_off < f:
                # Success: move and increase step size
                x = x_off
                f = f_off
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                sigma *= 1.5
            else:
                # Failure: keep position, decrease step size
                sigma *= 0.87

            # Keep sigma in reasonable bounds to prevent divergence or stalling
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Early termination if sigma is effectively zero
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
