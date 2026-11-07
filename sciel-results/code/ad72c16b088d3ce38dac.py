"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate descent with adaptive 1D golden-section refinement and multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        phi = (np.sqrt(5.0) - 1.0) / 2.0

        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_f, best_x
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def golden_section_1d(x_base, coord, lo, hi, max_iter):
            """Minimize along coordinate `coord` of x_base in [lo, hi] using golden-section.
            Returns (best_x, best_f, num_calls_used)."""
            a, b = lo, hi
            if b - a < 1e-12:
                return x_base.copy(), np.inf, 0

            # Initial two points
            c = b - phi * (b - a)
            d = a + phi * (b - a)

            xc = x_base.copy()
            xc[coord] = c
            fc = evaluate(xc)
            
            xd = x_base.copy()
            xd[coord] = d
            fd = evaluate(xd)

            best_val = c if fc < fd else d
            best_f_local = fc if fc < fd else fd

            for _ in range(max(0, max_iter - 2)):
                if calls >= budget:
                    break
                
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    xc = x_base.copy()
                    xc[coord] = c
                    fc = evaluate(xc)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    xd = x_base.copy()
                    xd[coord] = d
                    fd = evaluate(xd)

                if fc < fd:
                    if fc < best_f_local:
                        best_f_local = fc
                        best_val = c
                else:
                    if fd < best_f_local:
                        best_f_local = fd
                        best_val = d

            best_x_local = x_base.copy()
            best_x_local[coord] = best_val
            return best_x_local, best_f_local

        def coordinate_descent_sweep(x_start, window, gs_iters):
            """Perform one full sweep of coordinate descent."""
            x = x_start.copy()
            fx = evaluate(x)
            if calls >= budget:
                return x, fx

            # Randomize coordinate order to break symmetry
            coords = np.arange(dim)
            rng.shuffle(coords)

            for coord in coords:
                if calls >= budget:
                    break

                lo = max(-5.0, x[coord] - window)
                hi = min(5.0, x[coord] + window)

                if hi - lo < 1e-12:
                    continue

                x_new, fx_new = golden_section_1d(x, coord, lo, hi, gs_iters)

                if fx_new < fx:
                    x = x_new
                    fx = fx_new

            return x, fx

        # Phase 1: Exploration
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        # Evaluate 5 random points with a quick sweep (4 iterations per coord)
        n_exploration = 5
        for i in range(n_exploration):
            if calls >= budget:
                break
            x_rand = rng.uniform(-5, 5, dim)
            # Quick refinement: 1 sweep, 4 GS iterations per coord
            # Cost: 1 (eval) + 5 coords * (2 + 4) = 1 + 30 = 31 calls approx
            # Actually GS uses 2 initial + (iter-2) more. So 4 iter = 4 calls.
            # Total per point: 1 + 5*4 = 21 calls.
            x_ref, f_ref = coordinate_descent_sweep(x_rand, window=3.0, gs_iters=4)
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        # Phase 2: Refinement
        if best_x is not None and calls < budget:
            window = 2.0
            gs_iters = 8
            sweep_count = 0
            
            while calls < budget:
                # Perform a sweep
                x_cur, f_cur = coordinate_descent_sweep(best_x, window, gs_iters)
                
                if f_cur < best_f:
                    best_f = f_cur
                    best_x = x_cur.copy()
                
                sweep_count += 1
                
                # Adaptive strategy:
                # Shrink window as we converge
                window *= 0.8
                if window < 0.01:
                    window = 0.01
                    gs_iters = 12 # Increase precision when window is small
                
                # Every 3rd sweep, try a small perturbation to escape local minima
                if sweep_count % 3 == 0:
                    if calls < budget - 50: # Keep budget reserve
                        x_pert = best_x + rng.normal(0, window * 0.5, dim)
                        x_pert = np.clip(x_pert, -5, 5)
                        x_ref, f_ref = coordinate_descent_sweep(x_pert, window, gs_iters)
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
