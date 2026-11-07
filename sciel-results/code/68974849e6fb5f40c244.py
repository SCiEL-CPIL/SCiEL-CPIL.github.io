"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical multi-scale stochastic tunneling with per-dimension sigma
    and periodic finite-difference gradient polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center of box (often near the shifted optimum)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension sigma for adaptive step sizes
        sigma = np.full(dim, 2.0)
        min_sigma = 1e-6
        max_sigma = 5.0

        # Track consecutive failures for sigma adaptation
        fail_counts = np.zeros(dim, dtype=int)
        success_counts = np.zeros(dim, dtype=int)

        call_count = 1

        def try_polish(cx, cf, max_iters=30):
            """Finite-difference gradient descent polish."""
            nonlocal best_x, best_fx, call_count
            px = cx.copy()
            pfx = cf
            step = 0.01
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                # Compute gradient via central differences
                grad = np.zeros(dim)
                for d in range(dim):
                    e = np.zeros(dim)
                    e[d] = step
                    xp = np.clip(px + e, -5, 5)
                    xm = np.clip(px - e, -5, 5)
                    f_plus = func(xp)
                    call_count += 1
                    f_minus = func(xm)
                    call_count += 1
                    grad[d] = (f_plus - f_minus) / (2 * step)
                    if time.time() > deadline:
                        break
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break
                # Gradient descent step
                lr = step / gnorm
                new_x = np.clip(px - lr * grad, -5, 5)
                new_fx = func(new_x)
                call_count += 1
                if new_fx < pfx:
                    px = new_x
                    pfx = new_fx
                    if pfx < best_fx:
                        best_x = px.copy()
                        best_fx = pfx
                    step *= 0.95
                else:
                    step *= 0.5
                    if step < 1e-8:
                        break
            return px, pfx

        # Phase 1: Coarse exploration with large steps (multi-scale tunneling)
        # Start with large sigma, gradually reduce
        scale = 0.0
        scales = [3.0, 1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375,
                  0.01171875, 0.005859375, 0.0029296875, 0.00146484375,
                  0.000732421875, 0.000366210938, 0.000183105469, 0.000091552734]
        scale_idx = 0
        no_improve = 0
        max_no_improve = 15

        while call_count < budget:
            if time.time() > deadline:
                break

            # Determine current scale
            current_scale = scales[min(scale_idx, len(scales) - 1)]
            sigma[:] = current_scale

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                # Accept: move to new point
                x = y
                fx = fy
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                no_improve = 0
                # Slightly increase sigma on success (explore more)
                sigma *= 1.2
                sigma = np.clip(sigma, min_sigma, max_sigma)
            else:
                no_improve += 1
                # Reduce sigma on failure
                sigma *= 0.85
                sigma = np.clip(sigma, min_sigma, max_sigma)

                # If stuck, try a larger jump (tunneling)
                if no_improve >= 10:
                    jump_dir = rng.standard_normal(dim)
                    jump_mag = current_scale * 2.0
                    y2 = np.clip(x + jump_mag * jump_dir, -5, 5)
                    fy2 = func(y2)
                    call_count += 1
                    if fy2 < fx:
                        x = y2
                        fx = fy2
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                        no_improve = 0
                        sigma *= 1.5
                        sigma = np.clip(sigma, min_sigma, max_sigma)
                    else:
                        no_improve += 1

            # Advance scale when stuck or after enough iterations
            if no_improve >= max_no_improve and scale_idx < len(scales) - 1:
                scale_idx += 1
                no_improve = 0
                sigma[:] = scales[min(scale_idx, len(scales) - 1)]

            # Periodic polish every ~50 calls when sigma is small enough
            if call_count % 50 == 0 and current_scale < 0.1:
                x, fx = try_polish(x, fx, max_iters=15)
                if call_count >= budget:
                    break

        # Phase 2: Final high-precision polish
        if call_count < budget and time.time() < deadline:
            x, fx = try_polish(best_x, best_fx, max_iters=50)

        # Final check
        if fx < best_fx:
            best_x = x.copy()
            best_fx = fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
