"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    # Initial points: use a grid-like or random start within the triangle
    # We'll use a deterministic initial layout that is reasonably spread out.
    # Let's start with a few points near the vertices and some in the center.
    # Or just use random points with a fixed seed for reproducibility in optimization.
    
    # To make it robust, let's define the objective function.
    # Objective: maximize the minimum area of any triangle formed by 3 points.
    # Minimize -min_area.
    
    def min_triangle_area(points):
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(x):
        points = x.reshape(-1, 2)
        # Check if points are inside the triangle
        # Barycentric coordinates check
        # For a point P, and triangle vertices A, B, C:
        # P = a*A + b*B + c*C, with a+b+c=1, a,b,c >= 0
        # We can use the area method or linear algebra.
        # Let's use the cross product method.
        for i in range(N):
            p = points[i]
            # Check if p is inside triangle V
            # Using barycentric coordinates
            # A = V[0], B = V[1], C = V[2]
            A, B, C = V
            # Calculate areas
            # Total area
            total_area = 0.5 * abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))
            
            # Area of sub-triangles
            area1 = 0.5 * abs((B[0] - p[0]) * (C[1] - p[1]) - (C[0] - p[0]) * (B[1] - p[1]))
            area2 = 0.5 * abs((p[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (p[1] - A[1]))
            area3 = 0.5 * abs((B[0] - A[0]) * (p[1] - A[1]) - (p[0] - A[0]) * (B[1] - A[1]))
            
            # Due to floating point errors, we allow a small tolerance
            if area1 + area2 + area3 > total_area + 1e-9:
                # Point is outside
                return -10.0  # Penalty
        
        area = min_triangle_area(points)
        return -area

    # Initial guess: random points inside the triangle
    # We can use a simple rejection sampling for the initial guess
    import random
    rng = random.Random(42)
    x0 = []
    while len(x0) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            # Map to triangle
            px = u + v * 0.5
            py = v * h
            # Add a small offset to keep strictly inside if needed, but boundary is allowed
            x0.append([px, py])
    
    x0 = np.array(x0).flatten()
    
    # Use L-BFGS-B or SLSQP for optimization
    # SLSQP is good for constrained problems, but we handled constraints in objective with penalty.
    # Let's use L-BFGS-B with bounds to keep points within the triangle's bounding box,
    # and rely on the penalty or a more rigorous constraint handling.
    # Actually, L-BFGS-B doesn't handle non-linear constraints well.
    # Let's use SLSQP with explicit constraints.
    
    # Constraints: each point must be inside the triangle.
    # This is 3 constraints per point (barycentric coords >= 0).
    # Total 33 constraints.
    
    # Alternatively, we can parameterize the points using barycentric coordinates.
    # Let x = [a1, b1, c1, a2, b2, c2, ..., a11, b11, c11]
    # where a_i + b_i + c_i = 1, a_i, b_i, c_i >= 0
    # This automatically ensures points are inside the triangle.
    # Number of variables: 33.
    # We need to handle the sum constraint a+b+c=1.
    # We can drop one variable, say c_i = 1 - a_i - b_i.
    # So variables are a_i, b_i for i=1..11. Total 22 variables.
    # Bounds: 0 <= a_i <= 1, 0 <= b_i <= 1, a_i + b_i <= 1.
    
    def objective_bary(x):
        # x has 22 elements: a1, b1, a2, b2, ..., a11, b11
        points = []
        for i in range(N):
            a = x[2*i]
            b = x[2*i+1]
            c = 1 - a - b
            # Point = a*A + b*B + c*C
            p = a * V[0] + b * V[1] + c * V[2]
            points.append(p)
        points = np.array(points)
        area = min_triangle_area(points)
        return -area
    
    # Bounds for a_i, b_i
    bounds = [(0, 1)] * (2 * N)
    
    # Constraints: a_i + b_i <= 1
    # Inequality constraint: a_i + b_i - 1 <= 0
    # For scipy minimize, inequality constraints are of the form g(x) >= 0
    # So we need 1 - a_i - b_i >= 0
    
    cons = []
    for i in range(N):
        def make_constraint(idx):
            def constraint(x):
                a = x[2*idx]
                b = x[2*idx+1]
                return 1 - a - b
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})
    
    # Optimize
    result = minimize(objective_bary, x0_bary, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract points
    x_opt = result.x
    points = []
    for i in range(N):
        a = x_opt[2*i]
        b = x_opt[2*i+1]
        c = 1 - a - b
        p = a * V[0] + b * V[1] + c * V[2]
        points.append(p)
    
    return np.array(points)

# Initial guess in barycentric coordinates
import random
rng = random.Random(42)
x0_bary = []
for i in range(N):
    a = rng.random()
    b = rng.random()
    if a + b > 1:
        a = rng.random()
        b = rng.random()
        while a + b > 1:
            a = rng.random()
            b = rng.random()
    x0_bary.append(a)
    x0_bary.append(b)
x0_bary = np.array(x0_bary)
# EVOLVE-BLOCK-END
