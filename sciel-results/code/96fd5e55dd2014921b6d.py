"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    """Generate the Halton sequence value for a given index and base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_sequence(n, dim):
    """Generate n points in [0, 1)^dim using the Halton sequence."""
    points = np.zeros((n, dim))
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
    for i in range(1, n + 1):
        for d in range(dim):
            points[i - 1, d] = _halton(i, bases[d])
    return points


class Optimizer:
    """Halton global search + coordinate-wise lattice refinement + NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, calls_left, deadline):
        """Nelder-Mead polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        if calls >= calls_left or time.time() > deadline:
            return best_x, best_f, calls

        # Build initial simplex
        simplex = [x.copy()]
        fvals = [fx]
        step = 0.5
        for i in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            xi = x.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        if len(simplex) < 2:
            return best_x, best_f, calls

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        max_iter = 150

        for _ in range(max_iter):
            if calls >= calls_left or time.time() > deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()

            if calls >= calls_left or time.time() > deadline:
                break

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls < calls_left and time.time() <= deadline:
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        else:
                            break
                    if calls >= calls_left or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def _coord_refine(self, func, x0, calls_left, deadline):
        """Coordinate-wise local refinement exploiting Rastrigin's periodic structure."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # For each coordinate, try small perturbations to find the local minimum
        # Rastrigin has period 2 in each coordinate, so try offsets near multiples of 2
        step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125]

        for step in step_sizes:
            if calls >= calls_left or time.time() > deadline:
                break
            for coord in range(d):
                if calls >= calls_left or time.time() > deadline:
                    break
                # Try both directions
                for direction in [-1, 1]:
                    if calls >= calls_left or time.time() > deadline:
                        break
                    x_trial = x.copy()
                    x_trial[coord] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        x = x_trial.copy()
                    # Also try a slightly larger step in the same direction
                    if calls < calls_left and time.time() <= deadline:
                        x_trial2 = x.copy()
                        x_trial2[coord] += direction * step * 1.5
                        x_trial2 = np.clip(x_trial2, -5, 5)
                        f_trial2 = func(x_trial2)
                        calls += 1
                        if f_trial2 < best_f:
                            best_f = f_trial2
                            best_x = x_trial2.copy()
                            x = x_trial2.copy()

        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0
        calls = 0

        # Phase 1: Halton sequence global exploration (~60% budget)
        halton_budget = int(budget * 0.6)
        n_halton = min(halton_budget, 300)
        
        # Map Halton [0,1)^dim to [-5,5]^dim
        halton_pts = _halton_sequence(n_halton, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        
        # Add some structured points at the beginning
        structured = np.zeros((10, dim))
        structured[0] = 0.0
        structured[1] = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        structured[2] = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        structured[3] = np.array([2.0, -2.0, 2.0, -2.0, 2.0])
        structured[4] = np.array([-2.0, 2.0, -2.0, 2.0, -2.0])
        structured[5] = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
        structured[6] = np.array([-0.5, 0.5, -0.5, 0.5, -0.5])
        structured[7] = np.array([1.5, -1.5, 1.5, -1.5, 1.5])
        structured[8] = np.array([-1.5, 1.5, -1.5, 1.5, -1.5])
        structured[9] = np.array([3.0, -3.0, 3.0, -3.0, 3.0])
        
        all_pts = np.vstack([structured, halton_pts])
        
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(len(all_pts)):
            if calls >= budget or time.time() > deadline:
                break
            x = all_pts[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Coordinate-wise lattice refinement (~20% budget)
        refine_budget = min(int(budget * 0.2), budget - calls)
        if refine_budget >= 10 and time.time() < deadline:
            x_ref, f_ref, c_ref = self._coord_refine(
                func, best_x, refine_budget, deadline
            )
            calls += c_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        # Phase 3: Nelder-Mead polish (~20% budget)
        polish_budget = min(80, budget - calls)
        if polish_budget >= 10 and time.time() < deadline:
            x_pol, f_pol, c_pol = self._nm_polish(
                func, best_x, polish_budget, deadline
            )
            calls += c_pol
            if f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        # Phase 4: If budget remains, one more coordinate refinement + NM
        if calls < budget and time.time() < deadline and (budget - calls) >= 20:
            refine_budget2 = min(20, budget - calls)
            x_ref2, f_ref2, c_ref2 = self._coord_refine(
                func, best_x, refine_budget2, deadline
            )
            calls += c_ref2
            if f_ref2 < best_f:
                best_f = f_ref2
                best_x = x_ref2.copy()
            
            if calls < budget and time.time() < deadline and (budget - calls) >= 10:
                polish_budget2 = min(40, budget - calls)
                x_pol2, f_pol2, c_pol2 = self._nm_polish(
                    func, best_x, polish_budget2, deadline
                )
                calls += c_pol2
                if f_pol2 < best_f:
                    best_f = f_pol2
                    best_x = x_pol2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
