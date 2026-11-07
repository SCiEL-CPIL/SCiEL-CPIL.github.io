"""Black-box continuous optimizer on the BBOB-style function f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for a linear (or near-linear) function on [-5, 5]^d.
    Strategy:
      1. Evaluate the center (origin) as the first, guaranteed best-so-far.
      2. Probe the NEGATIVE axis directions (-step*e_i) first. For a linear
         function, each such probe that decreases the value immediately
         improves the best-so-far, which boosts early AOCC.
      3. Use the probe values to estimate the gradient and locate the
         analytic minimum vertex of the linear model on the box.
      4. Evaluate that vertex (the true minimum for a linear function).
      5. Polish with a (1+1)-ES to handle slight non-linearities / noise.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before the 330s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_f = float('inf')

        def _call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            call_count += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Step 1: Center (origin) -- first guaranteed best-so-far.
        base = np.zeros(d)
        f0 = _call(base)
        if f0 is None:
            # No budget / time; return whatever we have.
            return best_x, best_f

        # Step 2: Probe the NEGATIVE axis directions first.
        # For a linear function, a probe that decreases the value updates
        # best-so-far immediately (good for early AOCC).
        step = 4.0  # large step for strong gradient signal, still in box
        grad = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            p_neg = base.copy()
            p_neg[i] -= step
            fp_neg = _call(p_neg)
            if fp_neg is None:
                grad[i] = 0.0
                continue
            # grad[i] = (f(+step) - f(-step)) / (2*step)  (central-ish, but
            # we only have the -probe and the center). Use one-sided from
            # center: f(-step) - f(0) = -step*grad[i]  =>  grad[i] = (f0 - fp_neg)/step
            grad[i] = (f0 - fp_neg) / step

        # Step 3: Determine the analytic minimum vertex of the linear model.
        # f(x) = f0 + grad^T x.  Minimize over [-5,5]^d:
        #   x_i = -5 if grad[i] > 0,  x_i = +5 if grad[i] < 0,  else 0.
        vertex = np.zeros(d)
        for i in range(d):
            if grad[i] > 0:
                vertex[i] = -5.0
            elif grad[i] < 0:
                vertex[i] = 5.0
            else:
                vertex[i] = 0.0

        # Step 4: Evaluate the vertex (true minimum for a linear function).
        f_vertex = _call(vertex)
        if f_vertex is not None and f_vertex < best_f:
            best_f = f_vertex
            best_x = vertex.copy()

        # Step 5: Polish with (1+1)-ES to handle non-linearities / noise.
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - call_count
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5.0, 5.0)
                fy = _call(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if sigma < 1e-6:
                    # Converged; keep spending budget to lock in best-so-far
                    # (the evaluator repeats the last n_i for unused calls).
                    continue

        return best_x, best_f
# EVOLVE-BLOCK-END
