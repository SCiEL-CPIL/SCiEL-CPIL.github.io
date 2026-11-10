# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools
import math


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles using a convex program
    with auxiliary variables for sorted left endpoints and their consecutive gaps.
    
    The key insight: at each height y_m, the sorted left endpoints L_(1) <= L_(2) <= ... <= L_(n)
    can be represented as a convex combination of all permutations. We introduce auxiliary
    variables s_{i,m} representing the sorted position of triangle i at height m, and
    enforce the permutation constraints via linear inequalities that define the Birkhoff
    polytope (convex hull of permutation matrices).
    
    For computational tractability, we use a reduced set of quadrature points and solve
    the resulting LP using scipy's linprog with the interior-point method.
    """
    n = 64
    k = int(np.log2(n))
    a = np.array([(i + 1) / n for i in range(n)])
    
    # Use a reduced number of quadrature points for the LP (we'll verify with full M later)
    M_lp = 100  # Reduced for LP tractability
    M_full = 20000
    
    y_vals_lp = (np.arange(M_lp) + 0.5) / M_lp
    w_vals_lp = (1.0 - y_vals_lp) / n
    
    # Variables:
    # x_i (n variables): the positions
    # s_{i,m} (n*M_lp variables): sorted left endpoint of triangle i at height m
    # t_{j,m} (M_lp variables): the "excess" beyond w for each height, to handle min(gap, w)
    # 
    # Actually, let's use a simpler formulation:
    # At each height m, we have sorted endpoints L_(1,m) <= L_(2,m) <= ... <= L_(n,m)
    # The union length is: w_m + sum_{j=1}^{n-1} min(L_(j+1,m) - L_(j,m), w_m)
    #
    # We introduce variables:
    # - x_i: position of triangle i
    # - L_{i,m}: left endpoint of triangle i at height m (= x_i + a_i * y_m)
    # - S_{j,m}: the j-th smallest value among {L_{i,m}} for i=1..n
    # - G_{j,m}: gap between S_{j+1,m} and S_{j,m}
    # - H_{j,m}: min(G_{j,m}, w_m)
    #
    # Constraints:
    # 1. L_{i,m} = x_i + a_i * y_m (equality)
    # 2. S_{j,m} is the j-th order statistic: S_{j,m} >= L_{i,m} for at most j-1 values of i
    #    This is enforced by: sum_i max(0, L_{i,m} - S_{j,m}) <= (j-1) * (max_L - S_{j,m})
    #    But this is nonlinear. Instead, use the standard LP formulation for order statistics.
    #
    # Alternative approach: Use the fact that the union of intervals [L_i, L_i + w]
    # has length w + sum of overlaps. The overlap between consecutive sorted intervals
    # is min(gap, w). We can linearize min(gap, w) by introducing h_{j,m} <= gap_{j,m}
    # and h_{j,m} <= w_m, and maximizing h_{j,m} (since we're minimizing, we want to
    # minimize the sum, so we set h_{j,m} = min(gap_{j,m}, w_m)).
    #
    # Let's use a different strategy: since n=64 is small, we can enumerate a few
    # candidate permutations and solve the LP for each, taking the best.
    
    # Generate candidate permutations based on the Keich construction
    def keich_permutation():
        """Generate the permutation order from the Keich construction."""
        j_arr = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
        positions = np.array(positions, dtype=np.float64)
        # The sorted order at y=0 is just the order of positions
        return np.argsort(positions)
    
    # For a fixed permutation sigma, the problem becomes:
    # Minimize sum_m [w_m + sum_j min(x_{sigma(j+1)} + a_{sigma(j+1)}*y_m - x_{sigma(j)} - a_{sigma(j)}*y_m, w_m)]
    # subject to x being any real vector
    #
    # This is a convex piecewise-linear function of x, which can be solved as an LP.
    
    def solve_for_permutation(sigma, y_vals, w_vals, M):
        """
        Solve the LP for a fixed permutation sigma.
        
        Variables: x_0, ..., x_{n-1} (n variables)
        And auxiliary variables h_{j,m} for j=0..n-2, m=0..M-1 (n-1*M variables)
        
        Objective: minimize (1/M) * sum_m [w_m + sum_j h_{j,m}]
        
        Constraints:
        For each m and j:
            h_{j,m} <= (x_{sigma(j+1)} - x_{sigma(j)}) + (a_{sigma(j+1)} - a_{sigma(j)}) * y_m
            h_{j,m} <= w_m
            h_{j,m} >= 0
        """
        # Number of variables: n + (n-1)*M
        n_vars = n + (n - 1) * M
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # x variables: 0 (they don't appear directly in objective)
        # h variables: 1/M for each
        for m in range(M):
            for j in range(n - 1):
                c[n + m * (n - 1) + j] = 1.0 / M
        
        # Constraints: h_{j,m} <= gap_{j,m} and h_{j,m} <= w_m and h_{j,m} >= 0
        # Inequality form: A_ub @ vars <= b_ub
        
        # First set: h_{j,m} - gap_{j,m} <= 0
        # gap_{j,m} = (x_{sigma(j+1)} - x_{sigma(j)}) + (a_{sigma(j+1)} - a_{sigma(j)}) * y_m
        # So: -x_{sigma(j+1)} + x_{sigma(j)} + h_{j,m} <= -(a_{sigma(j+1)} - a_{sigma(j)}) * y_m
        
        n_constraints = 2 * (n - 1) * M
        
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        for m in range(M):
            for j in range(n - 1):
                idx_h = n + m * (n - 1) + j
                idx_x_prev = sigma[j]
                idx_x_next = sigma[j + 1]
                
                # Constraint 1: h_{j,m} - (x_{next} - x_{prev}) - (a_{next} - a_{prev})*y_m <= 0
                # => -x_{next} + x_{prev} + h_{j,m} <= -(a_{next} - a_{prev})*y_m
                row1 = m * (n - 1) + j
                A_ub[row1, idx_x_prev] = 1.0
                A_ub[row1, idx_x_next] = -1.0
                A_ub[row1, idx_h] = 1.0
                b_ub[row1] = -(a[sigma[j + 1]] - a[sigma[j]]) * y_vals[m]
                
                # Constraint 2: h_{j,m} <= w_m
                row2 = (n - 1) * M + m * (n - 1) + j
                A_ub[row2, idx_h] = 1.0
                b_ub[row2] = w_vals[m]
        
        # Lower bounds: h >= 0, x unbounded
        bounds = [(None, None)] * n + [(0, None)] * ((n - 1) * M)
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            return result.x[:n], result.fun
        else:
            return None, np.inf
    
    # Generate several candidate permutations
    sigma_keich = keich_permutation()
    
    # Also try the identity permutation and a few others
    candidate_sigmoids = [
        sigma_keich,
        np.arange(n),
        np.array([(i * 7) % n for i in range(n)]),  # Some mixing
        np.array([(i * 13) % n for i in range(n)]),
    ]
    
    best_x = None
    best_area = np.inf
    
    for sigma in candidate_sigmoids:
        # Ensure it's a valid permutation
        if len(set(sigma)) != n:
            continue
        x_sol, area = solve_for_permutation(sigma, y_vals_lp, w_vals_lp, M_lp)
        if x_sol is not None and area < best_area:
            best_area = area
            best_x = x_sol
    
    # If LP failed, fall back to Keich construction
    if best_x is None:
        j_arr = np.arange(1, k + 1)
        positions_keich = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions_keich.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
        best_x = np.array(positions_keich, dtype=np.float64)
    
    # Verify with full quadrature
    y_vals_full = (np.arange(M_full) + 0.5) / M_full
    w_vals_full = (1.0 - y_vals_full) / n
    
    def compute_area(x):
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals_full[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        min_gaps = np.minimum(gaps, w_vals_full[np.newaxis, :])
        union_lengths = w_vals_full + np.sum(min_gaps, axis=0)
        return np.mean(union_lengths)
    
    final_area = compute_area(best_x)
    
    # If the LP solution is bad, try local refinement
    from scipy.optimize import minimize
    
    result = minimize(
        compute_area,
        best_x,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success and result.fun < final_area:
        best_x = result.x
        final_area = result.fun
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_x)):
        j_arr = np.arange(1, k + 1)
        positions_keich = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions_keich.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
        best_x = np.array(positions_keich, dtype=np.float64)
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
