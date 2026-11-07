"""Multi-start Powell's method with stratified LHS initialization for bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Powell with dense stratified LHS global sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, n, dim, rng):
        """Stratified Latin hypercube samples in [0,1]^dim."""
        pts = np.empty((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            pts[:, j] = (perm + rng.random(n)) / n
        return pts

    def _powell(self, func, x0, fx0, deadline, state):
        """Powell's method with budget and time guards. Returns (best_x, best_fx)."""
        d = self.dim
        x = x0.copy()
        f = fx0
        # Initial direction set: standard basis
        directions = [np.eye(d, dtype=float)[i] for i in range(d)]

        def line_search(direction, start, cur_f):
            """Golden-section / parabolic 1-D line search along direction from start."""
            # Find bracket by expanding
            step = 0.5
            a = start - step * direction
            fa = func(a)
            b = start
            fb = cur_f
            # Expand in the direction of steepest descent
            c = start + step * direction
            fc = func(c)
            state['calls'] += 2
            # Determine bracket [lo, hi] containing minimum
            if fc < fb:
                lo, f_lo, hi, f_hi = b, fb, c, fc
                # extend hi
                while fc < f_lo and state['calls'] < state['budget'] and time.time() < deadline:
                    step *= 2.0
                    new_hi = hi + step * direction
                    f_new_hi = func(new_hi)
                    state['calls'] += 1
                    lo, f_lo = hi, f_hi
                    hi, f_hi = new_hi, f_new_hi
                    if f_new_hi >= f_lo:
                        break
            else:
                lo, f_lo, hi, f_hi = a, fa, b, fb
                # extend lo
                while fa < fb and state['calls'] < state['budget'] and time.time() < deadline:
                    step *= 2.0
                    new_lo = lo - step * direction
                    f_new_lo = func(new_lo)
                    state['calls'] += 1
                    hi, f_hi = lo, f_lo
                    lo, f_lo = new_lo, f_new_lo
                    if f_new_lo >= f_hi:
                        break

            # Golden section within [lo, hi]
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            lo_f = f_lo
            hi_f = f_hi
            for _ in range(30):
                if state['calls'] >= state['budget'] or time.time() >= deadline:
                    break
                if hi - lo < 1e-10:
                    break
                m1 = lo + gr * (hi - lo)
                m2 = hi - gr * (hi - lo)
                f_m1 = func(m1)
                f_m2 = func(m2)
                state['calls'] += 2
                if f_m1 < f_m2:
                    hi, hi_f = m2, f_m2
                else:
                    lo, lo_f = m1, f_m1
            mid = 0.5 * (lo + hi)
            return mid, min(lo_f, hi_f)

        # Main Powell loop
        for _ in range(3):
            if state['calls'] >= state['budget'] or time.time() >= deadline:
                break
            old_f = f
            for i in range(d):
                if state['calls'] >= state['budget'] or time.time() >= deadline:
                    break
                new_pt, new_f = line_search(directions[i], x, f)
                if new_f < f:
                    x = new_pt
                    f = new_f
            # Check if full cycle improved; if so, add a new direction
            if f < old_f:
                new_dir = x - x0
                norm = np.linalg.norm(new_dir)
                if norm > 1e-12:
                    new_dir = new_dir / norm
                    # Replace the direction that gave least improvement
                    directions[-1] = new_dir
                x0 = x.copy()

        return x, f

    def __call__(self, func):
        deadline = time.time() + 295.0  # safety margin before 300s
        state = {'calls': 0, 'budget': self.budget}
        rng = np.random.default_rng(0)
        d = self.dim

        # Dense stratified LHS for global basin identification
        n_lhs = 150
        samples_unit = self._latin_hypercube(n_lhs, d, rng)
        samples = samples_unit * 10.0 - 5.0  # map to [-5, 5]^d

        best_x = None
        best_fx = np.inf

        # Evaluate LHS points (cheap global scan)
        for s in samples:
            if state['calls'] >= state['budget'] or time.time() >= deadline:
                break
            fs = func(s)
            state['calls'] += 1
            if fs < best_fx:
                best_fx = fs
                best_x = s.copy()

        if best_x is None:
            return None, np.inf

        # Refine the best LHS point with Powell
        x, fx = self._powell(func, best_x, best_fx, deadline, state)
        if fx < best_fx:
            best_fx, best_x = fx, x

        # Multi-start: refine next-best LHS basins with smaller Powell starts
        # Sort LHS points by value (we have them implicitly; re-evaluate top candidates cheaply)
        # Use a few additional starts near the best to catch nearby basins
        for k in range(1, 6):
            if state['calls'] >= state['budget'] or time.time() >= deadline:
                break
            # Perturb the best point to find sibling basins
            pert = best_x + 0.25 * rng.standard_normal(d)
            pert = np.clip(pert, -5, 5)
            fp = func(pert)
            state['calls'] += 1
            if fp < best_fx:
                best_fx, best_x = fp, pert.copy()
            x, fx = self._powell(func, pert, fp, deadline, state)
            if fx < best_fx:
                best_fx, best_x = fx, x

        return best_x, best_fx
# EVOLVE-BLOCK-END
