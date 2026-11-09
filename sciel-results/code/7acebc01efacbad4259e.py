import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # Variables: x0, y0, ..., x12, y12, delta
    # Total variables: 2*n + 1
    num_vars = 2 * n + 1
    
    # Objective: Minimize -delta
    c = np.zeros(num_vars)
    c[-1] = -1.0
    
    # Equality constraints:
    # 1. Centroid x = 0
    # 2. Centroid y = 0
    # 3. Hull Area = 1 (Shoelace formula sum = 2)
    # 4. y0 = 0 (Fix rotation)
    
    A_eq = np.zeros((4, num_vars))
    b_eq = np.zeros(4)
    
    # Centroid x
    for i in range(n):
        A_eq[0, 2*i] = 1.0 / n
    b_eq[0] = 0.0
    
    # Centroid y
    for i in range(n):
        A_eq[1, 2*i+1] = 1.0 / n
    b_eq[1] = 0.0
    
    # Hull Area = 1 => 0.5 * sum(xi*y(i+1) - x(i+1)*yi) = 1 => sum(...) = 2
    # Note: The shoelace formula for a convex polygon in CCW order gives positive area.
    # We assume the order 0..12 is CCW.
    for i in range(n):
        j = (i + 1) % n
        # Term: xi * yj - xj * yi
        A_eq[2, 2*i] = 1.0    # coeff for xi
        A_eq[2, 2*j+1] = 1.0  # coeff for yj
        A_eq[2, 2*j] = -1.0   # coeff for xj
        A_eq[2, 2*i+1] = -1.0 # coeff for yi
    b_eq[2] = 2.0
    
    # y0 = 0
    A_eq[3, 1] = 1.0
    b_eq[3] = 0.0
    
    # Inequality constraints:
    # A_ub @ v <= b_ub
    # 1. Triangle areas: -Area(i,j,k) + delta <= 0
    #    Area(i,j,k) = 0.5 * | ... |. Assuming CCW convex, the expression inside is positive.
    #    So Area = 0.5 * (xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj)
    #    Constraint: -(xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj) + 2*delta <= 0
    #    (Using 2*delta to avoid 0.5 fractions, or just keep 0.5). Let's use 2*delta for integer-like coeffs? No, floats are fine.
    #    Let's define the linear form L(i,j,k) = xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj
    #    Area = 0.5 * L
    #    Constraint: 0.5 * L >= delta  =>  -0.5 * L + delta <= 0
    
    num_triangles = n * (n-1) * (n-2) // 6
    A_ub = np.zeros((num_triangles, num_vars))
    b_ub = np.zeros(num_triangles)
    
    row = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Coefficients for L(i,j,k)
                # xi * (yj - yk)
                A_ub[row, 2*i] = 0.5
                A_ub[row, 2*j+1] = -0.5
                A_ub[row, 2*k+1] = 0.5
                
                # xj * (yk - yi)
                A_ub[row, 2*j] = 0.5
                A_ub[row, 2*k+1] -= 0.5 # Wait, yk coeff is 0.5 - 0.5 = 0?
                # Let's re-derive:
                # L = xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj
                # Coeff xi: yj - yk
                # Coeff yi: -xj + xk
                # Coeff xj: yk - yi
                # Coeff yj: xi - xk
                # Coeff xk: yi - yj
                # Coeff yk: xj - xi
                
                # Let's just fill the matrix directly from the expanded form to avoid mistakes.
                # L = xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj
                # Constraint: -0.5 * L + delta <= 0
                # So coeffs are -0.5 * (coeff of L)
                
                # Reset row
                A_ub[row, :] = 0.0
                
                # xi terms
                A_ub[row, 2*i] = -0.5 * (1.0)   # from xi*yj
                A_ub[row, 2*i] += -0.5 * (-1.0) # from -xi*yk => +0.5? No.
                # Let's do it term by term.
                # Term 1: xi * yj. Coeffs: x_i: 1, y_j: 1.
                # Term 2: -xi * yk. Coeffs: x_i: -1, y_k: -1.
                # Term 3: xj * yk. Coeffs: x_j: 1, y_k: 1.
                # Term 4: -xj * yi. Coeffs: x_j: -1, y_i: -1.
                # Term 5: xk * yi. Coeffs: x_k: 1, y_i: 1.
                # Term 6: -xk * yj. Coeffs: x_k: -1, y_j: -1.
                
                # Sum of coeffs for each variable in L:
                # x_i: 1 - 1 = 0? No.
                # x_i appears in Term 1 (with yj) and Term 2 (with yk).
                # The coefficient of x_i in the linear form L is (yj - yk).
                # Since yj and yk are variables, this is not a constant coefficient!
                # L is BILINEAR, not LINEAR!
                
    # STOP. The area of a triangle is NOT a linear function of the coordinates.
    # It is bilinear. $x_i y_j$ is a product of two variables.
    # Therefore, the problem is NOT a Linear Program.
    
    # The directive said "SOCP". An SOCP can handle quadratic constraints.
    # But `scipy.optimize` does not have a built-in SOCP solver.
    # We have `scipy.optimize.minimize` (non-convex) and `linprog` (linear).
    
    # Re-evaluating the directive: "Formulate ... as a convex feasibility optimization ... SOCP".
    # Since I cannot use CVXPY, I must implement the convex optimization myself or use a trick.
    
    # Is there a way to make it linear?
    # If we fix the convex hull vertices to be on a circle? No.
    # If we fix the points to be in a specific grid? No.
    
    # Let's look at the "binary search on delta" part.
    # Feasibility: Exists points such that Area(i,j,k) >= delta and HullArea=1.
    # This is a non-convex feasibility problem.
    
    # However, for N=13, the optimal solution is likely the regular 13-gon.
    # Score of regular 13-gon:
    # A_hull = Area(Regular 13-gon, side 1) ? No, we normalize.
    # For a regular N-gon, the minimum triangle area is formed by 3 consecutive vertices or similar.
    
    # If I can't solve the convex program, I should improve the local search significantly.
    # The current program uses Nelder-Mead.
    # I can use `scipy.optimize.differential_evolution` for a global search, which is more robust for this non-convex problem.
    # Or, I can use the fact that the problem is symmetric.
    
    # Let's try `differential_evolution`. It is a global optimization algorithm.
    # It might be slow, but 600s is a lot of time.
    
    # Strategy:
    # 1. Use `differential_evolution` to maximize the score.
    # 2. Objective: -Score.
    # 3. Bounds: Points should be within a reasonable box, e.g., [-1, 1].
    # 4. We need to handle the normalization (Hull Area = 1) inside the objective or by transforming the output.
    #    Better: Normalize inside the objective function.
    
    # Let's switch to Differential Evolution.

    # Define the objective function for DE
    def objective_de(p):
        pts = p.reshape(n, 2)
        # Check for NaN/Inf
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        # Normalize to Hull Area = 1
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e10
            pts = pts / np.sqrt(hull_area)
        except Exception:
            return 1e10
        
        # Calculate min area
        min_area = 1e10
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    if area < min_area:
                        min_area = area
        
        # Return negative score (min area / hull area). Hull area is 1 now.
        return -min_area

    # Run DE
    from scipy.optimize import differential_evolution
    
    # Bounds: Each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * n)
    
    # DE can be slow. Set maxiter and popsize appropriately.
    # 13 points = 26 variables.
    # popsize=15, maxiter=500 might be too slow?
    # Let's try a smaller budget first.
    
    result = differential_evolution(objective_de, bounds, maxiter=200, popsize=10, seed=42, polish=True)
    
    best_points = result.x.reshape(n, 2)
    
    # Normalize best_points
    try:
        hull = ConvexHull(best_points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(best_points)
        best_points = best_points / np.sqrt(hull.volume)

    return best_points
