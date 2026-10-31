"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _overlap_max(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    
    h is a vector of step heights on [0,2] with equal width (2/N each).
    The function h(x) is constant on each subinterval.
    1-h(x) = 1 - h(x).
    
    The overlap at shift s is the integral of min(h(x), 1-h(x+s)) dx over [0,2].
    We want to minimize the maximum over all shifts s in [0, 2].
    
    Since h is piecewise constant with N steps, the overlap as a function of shift
    is piecewise linear (or piecewise constant depending on interpretation).
    Actually, for step functions, the overlap at a given shift can be computed by
    summing over pairs of steps.
    
    Let's think carefully. h has N steps, each of width w = 2/N.
    h[i] is the value on interval [i*w, (i+1)*w).
    
    For a shift s (in [0, 2]), the shifted function 1-h(x+s) has value 1-h(x+s) at point x.
    The overlap integral is ∫₀² min(h(x), 1-h(x+s)) dx.
    
    Since both functions are piecewise constant, the minimum is also piecewise constant,
    and the integral is a sum over the N+M subintervals where either function changes.
    
    For a given shift s, the number of subintervals is at most 2N.
    To find the maximum over all shifts s, we note that the overlap function is piecewise
    linear in s (actually piecewise constant with jumps? No, it's continuous).
    
    Actually, let me reconsider. The overlap as a function of shift s is continuous and
    piecewise linear. Its maximum occurs at one of the breakpoints, which are at shifts
    that are multiples of w = 2/N, or at shifts where the edges of h and 1-h(·+s) align.
    
    The breakpoints in s occur at s = k*w for k = 0, 1, ..., N (mod 2).
    So we need to check s = k * (2/N) for k = 0, 1, ..., N-1 (shifts modulo 2).
    
    Wait, the shift is on [0, 2]. The function h is defined on [0, 2]. When we shift by s,
    we need to handle the wrap-around or the domain issue. The task says "shifted 1-h", so
    I think we consider h defined on [0,2] and 1-h(x+s) where x+s may go beyond 2.
    But the integral is over [0,2]. So for x+s > 2, we don't have a value.
    
    Hmm, let me re-read: "minimize the maximum overlap of h with shifted 1-h."
    
    I think the standard interpretation is: h is a function on [0,2] (or perhaps periodic on [0,2]).
    The overlap at shift s is ∫₀² min(h(x), 1-h((x+s) mod 2)) dx, assuming periodic extension.
    
    Or perhaps it's ∫₀² min(h(x), 1-h(x+s)) dx where h is only defined on [0,2] and
    1-h(x+s) is only defined when x+s ∈ [0,2], so the effective domain shrinks.
    
    Given the ambiguity, I'll assume periodic extension on [0,2], which is common in such problems.
    
    For periodic extension, the overlap at shift s is:
    O(s) = ∫₀² min(h(x), 1-h((x+s) mod 2)) dx
    
    Since h is piecewise constant with N steps of width w=2/N, and (x+s) mod 2 shifts the grid,
    the function min(h(x), 1-h((x+s) mod 2)) is piecewise constant with at most 2N pieces.
    
    The overlap O(s) is a piecewise linear function of s? No, it's actually piecewise constant
    in s with jumps at s = k*w. Wait, no. As s varies continuously, the alignment of the two
    grids changes continuously, so O(s) is piecewise linear.
    
    Actually, for step functions, O(s) is piecewise linear in s. The maximum of a piecewise
    linear function over [0, 2] occurs at a breakpoint. The breakpoints are at s = k*w for
    k = 0, 1, ..., N (with s mod 2).
    
    So the maximum overlap is max over k=0,...,N-1 of O(k*w).
    
    Let me implement this.
    """
    w = 2.0 / N
    h = np.asarray(h, dtype=float)
    n = len(h)
    
    # For each shift k*w, compute the overlap
    max_ov = 0.0
    for k in range(n):
        s = k * w
        # For each subinterval [i*w, (i+1)*w), compute min(h[i], 1-h[(i+k) mod n]) * w
        # But wait, the shift s = k*w means that the grid of 1-h is shifted by k steps.
        # So at position x in [i*w, (i+1)*w), h(x) = h[i].
        # 1-h((x+s) mod 2): (x+s) mod 2 is in [(i+k)*w, (i+k+1)*w) mod 2.
        # So 1-h((x+s) mod 2) = 1 - h[(i+k) % n].
        # Thus min(h[i], 1-h[(i+k)%n]) is constant on the subinterval.
        # The contribution is w * min(h[i], 1-h[(i+k)%n]).
        
        vals = np.minimum(h, 1.0 - h[np.arange(n) + k] % n if False else 1.0 - np.roll(h, -k))
        ov = w * np.sum(vals)
        if ov > max_ov:
            max_ov = ov
    
    return max_ov


def construct():
    deadline = time.time() + 120  # Safety margin
    
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    # Bounds: each h[i] in [0, 1], sum(h) = N/2
    # Use differential evolution with bounds [0,1]^N, then project to sum constraint.
    
    def objective(h):
        # Project to satisfy sum constraint
        h = np.clip(h, 0.0, 1.0)
        # Adjust to make sum exactly N/2
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            # Distribute the difference evenly, then clip and re-adjust
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            # If clipping changed the sum, do a more careful adjustment
            diff2 = N / 2.0 - np.sum(h)
            if abs(diff2) > 1e-12:
                # Add/subtract from entries that are not at bounds
                mask = (h > 1e-10) & (h < 1.0 - 1e-10)
                if np.any(mask):
                    h[mask] += diff2 / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        return _overlap_max(h)
    
    # Run differential evolution
    bounds = [(0.0, 1.0)] * N
    best = None
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15, tol=1e-6,
            seed=42, polish=False, update_interval=5
        )
        if result.success or result.fun < _overlap_max(h0):
            best = result.x
    except Exception:
        pass
    
    # Local polish with SLSQP
    if best is not None:
        h_best = np.clip(best, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h_best)
        h_best = h_best + diff / N
        h_best = np.clip(h_best, 0.0, 1.0)
        
        # Use equality constraint for sum
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
        bounds_local = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                _overlap_max, h_best, method='SLSQP',
                bounds=bounds_local, constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < _overlap_max(h_best):
                h_best = res.x
        except Exception:
            pass
        
        # Final projection
        h_best = np.clip(h_best, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h_best)
        if abs(diff) > 1e-12:
            h_best = h_best + diff / N
            h_best = np.clip(h_best, 0.0, 1.0)
            diff2 = N / 2.0 - np.sum(h_best)
            if abs(diff2) > 1e-12:
                mask = (h_best > 1e-10) & (h_best < 1.0 - 1e-10)
                if np.any(mask):
                    h_best[mask] += diff2 / np.sum(mask)
                    h_best = np.clip(h_best, 0.0, 1.0)
        
        if _overlap_max(h_best) < _overlap_max(h0):
            return h_best
    
    return h0

# EVOLVE-BLOCK-END
