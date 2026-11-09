# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Strategy:
    1. Enumerate all possible topological orderings (separation patterns) for
       pairs of squares. For each pair (i,j), either i is left of j, i is right
       of j, i is below j, or i is above j. This gives 4^(n*(n-1)/2) possible
       orderings, which is too many to enumerate fully.
    2. Instead, we use a MILP formulation with binary variables to encode the
       separation constraints, and solve using scipy's linprog with integer
       constraints (via branch and bound if available, or we use a heuristic
       approach).
    3. Since scipy.optimize.linprog doesn't support integer variables directly,
       we use a different approach: we enumerate a subset of likely optimal
       topological structures and solve LPs for each.

    Actually, let's use a simpler but effective approach:
    - Use scipy.optimize.milp if available (SciPy >= 1.7)
    - Otherwise, fall back to a structured search over topological patterns

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 570  # safety margin before 630s kill

    # Grid baseline
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    best_cx = cx0.copy()
    best_cy = cy0.copy()
    best_s = s0.copy()
    best_obj = np.sum(s0)

    def check_feasible(cx, cy, s):
        """Check if configuration is feasible."""
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    def violation(x):
        """Compute total violation of constraints. x = [cx(10), cy(10), s(10)]."""
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:]
        v = 0.0
        # Boundary violations
        for i in range(n):
            if cx[i] - s[i]/2 < 0:
                v += (0 - (cx[i] - s[i]/2)) ** 2
            if cx[i] + s[i]/2 > 1:
                v += ((cx[i] + s[i]/2) - 1) ** 2
            if cy[i] - s[i]/2 < 0:
                v += (0 - (cy[i] - s[i]/2)) ** 2
            if cy[i] + s[i]/2 > 1:
                v += ((cy[i] + s[i]/2) - 1) ** 2
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t and dy < t:
                    # Both dimensions violate; use the smaller gap
                    gap_x = t - dx
                    gap_y = t - dy
                    v += min(gap_x, gap_y) ** 2
        return v

    def objective_sum(x):
        """Negative sum of side lengths for minimization."""
        return -np.sum(x[2*n:])

    def penalty_objective(x, penalty_weight=1000.0):
        """Penalized objective for differential evolution."""
        return objective_sum(x) + penalty_weight * violation(x)

    def is_feasible(x):
        """Check if x is feasible."""
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:]
        if not np.all(s > 0):
            return False
        return check_feasible(cx, cy, s)

    # Differential evolution bounds
    # cx, cy in [0, 1], s in [0.01, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.01, 0.5)] * n

    # Run differential evolution
    de_result = differential_evolution(
        penalty_objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=20,
        tol=1e-8,
        polish=True,
        updating='immediate',
        init='sobol'
    )

    if de_result.success or violation(de_result.x) < 1e-6:
        de_x = de_result.x
        if is_feasible(de_x):
            de_cx = de_x[:n].copy()
            de_cy = de_x[n:2*n].copy()
            de_s = de_x[2*n:].copy()
            de_obj = np.sum(de_s)
            if de_obj > best_obj:
                best_obj = de_obj
                best_cx = de_cx
                best_cy = de_cy
                best_s = de_s

    # Now use SLSQP to refine from multiple starting points
    def slsqp_constraints(x):
        """Constraints for SLSQP: return list of (type, fun, jac) tuples."""
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:]
        constraints = []

        # Boundary constraints
        for i in range(n):
            # cx_i - s_i/2 >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2.0})
            # 1 - cx_i - s_i/2 >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i] - x[2*n+i]/2.0})
            # cy_i - s_i/2 >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2.0})
            # 1 - cy_i - s_i/2 >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[n+i] - x[2*n+i]/2.0})

        # Non-overlap constraints: for each pair (i,j), at least one of the four separations holds
        # We model this as: max(0, (s_i+s_j)/2 - |cx_i - cx_j|) + max(0, (s_i+s_j)/2 - |cy_i - cy_j|) <= (s_i+s_j)/2
        # A simpler smooth approximation: use the constraint that the minimum of the four separations is >= 0
        # Separation 1: cx_i + s_i/2 <= cx_j - s_j/2  => cx_j - cx_i >= (s_i+s_j)/2
        # Separation 2: cx_j + s_j/2 <= cx_i - s_i/2  => cx_i - cx_j >= (s_i+s_j)/2
        # Separation 3: cy_i + s_i/2 <= cy_j - s_j/2  => cy_j - cy_i >= (s_i+s_j)/2
        # Separation 4: cy_j + s_j/2 <= cy_i - s_i/2  => cy_i - cy_j >= (s_i+s_j)/2
        # At least one must hold. We can use: min(sep1, sep2, sep3, sep4) >= 0
        # But min is non-differentiable. Use a smooth approximation or just add all four as separate
        # constraints with a big-M formulation is complex. Instead, use the fact that if squares don't
        # overlap, then |cx_i - cx_j| >= (s_i+s_j)/2 OR |cy_i - cy_j| >= (s_i+s_j)/2.
        # We'll use a penalty-based approach within SLSQP by adding constraints that push toward
        # non-overlap.
        
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0
                # We need: |cx_i - cx_j| >= t OR |cy_i - cy_j| >= t
                # This is non-convex. For SLSQP, we'll add constraints that encourage separation.
                # Use the constraint: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= t^2 is not sufficient for axis-aligned
                # Instead, use: min(|cx_i-cx_j|, t) + min(|cy_i-cy_j|, t) >= t is also non-smooth.
                # For now, let's just rely on the DE result and do a simple local search.
                pass

        return constraints

    # Since SLSQP with these constraints is complex, let's do a simpler local improvement
    def local_improve(cx, cy, s):
        """Try to improve the configuration locally using SLSQP on a simplified problem."""
        cx = cx.copy()
        cy = cy.copy()
        s = s.copy()
        
        # Try to increase each side while maintaining feasibility
        for iteration in range(100):
            if time.time() > deadline - 60:
                break
            improved = False
            for i in range(n):
                # Try increasing side i
                delta = 0.001
                s_new = s.copy()
                s_new[i] += delta
                if check_feasible(cx, cy, s_new):
                    s = s_new
                    improved = True
                else:
                    # Try moving square i to make room
                    for dx in [-0.005, 0, 0.005]:
                        for dy in [-0.005, 0, 0.005]:
                            cx_new = cx.copy()
                            cy_new = cy.copy()
                            cx_new[i] += dx
                            cy_new[i] += dy
                            if check_feasible(cx_new, cy_new, s_new):
                                cx, cy, s = cx_new, cy_new, s_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break

        return cx, cy, s

    # Apply local improvement to the best solution
    if best_obj > 0:
        best_cx, best_cy, best_s = local_improve(best_cx, best_cy, best_s)
        new_obj = np.sum(best_s)
        if new_obj > best_obj:
            best_obj = new_obj

    # Final cleanup
    best_s = np.maximum(best_s, 1e-8)
    for i in range(n):
        best_cx[i] = np.clip(best_cx[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)
        best_cy[i] = np.clip(best_cy[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
