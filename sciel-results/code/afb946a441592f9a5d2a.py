# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

N = 11
h = np.sqrt(3) / 2
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: numpy array of shape (N, 2)
    """
    min_area = float('inf')
    n = len(points)
    # Precompute for efficiency? N=11 is small.
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

def objective_bary(x):
    """
    Objective function for differential evolution.
    x: array of length 2*N, containing [a1, b1, a2, b2, ..., aN, bN]
    Returns negative of the minimum triangle area.
    """
    points = np.empty((N, 2))
    for i in range(N):
        a = x[2*i]
        b = x[2*i+1]
        c = 1.0 - a - b
        # p = a*V[0] + b*V[1] + c*V[2]
        # V[0] = (0,0), V[1] = (1,0), V[2] = (0.5, h)
        # px = a*0 + b*1 + c*0.5 = b + 0.5*(1-a-b) = b + 0.5 - 0.5a - 0.5b = 0.5 + 0.5b - 0.5a
        # py = a*0 + b*0 + c*h = h*(1-a-b)
        px = 0.5 + 0.5 * b - 0.5 * a
        py = h * (1.0 - a - b)
        points[i] = [px, py]
    
    area = min_triangle_area(points)
    return -area

def construct():
    # Bounds for a and b: 0 <= a <= 1, 0 <= b <= 1, a+b <= 1
    # DE handles box constraints. The condition a+b<=1 is not a box constraint.
    # We can use a transformation or just rely on the fact that if a+b>1, the point is outside.
    # However, DE doesn't handle non-box constraints directly.
    # Option 1: Use penalty in objective if a+b > 1.
    # Option 2: Parameterize differently.
    # Let's stick to simple bounds [0,1] for a and b, and add a penalty in the objective
    # if the point falls outside the triangle (a+b > 1).
    
    def objective_with_penalty(x):
        # Check if any point is outside
        for i in range(N):
            a = x[2*i]
            b = x[2*i+1]
            if a + b > 1.0 + 1e-9:
                # Return a large penalty
                return 100.0 
        return objective_bary(x)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # popsize: 15 * dim = 15 * 22 = 330. This might be slow.
    # Let's try popsize=10, maxiter=100.
    # DE is stochastic. We want a good result.
    # Time limit 120s.
    
    # To improve success rate, we can run a few iterations or use a seed.
    # Let's try a standard run first.
    try:
        result = differential_evolution(
            objective_with_penalty, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9
        )
        x_opt = result.x
    except Exception:
        # Fallback to a random valid configuration if DE fails
        x_opt = np.random.rand(2 * N)
        # Ensure validity
        for i in range(N):
            a = x_opt[2*i]
            b = x_opt[2*i+1]
            if a + b > 1:
                # Resample
                a = np.random.rand()
                b = np.random.rand()
                while a + b > 1:
                    a = np.random.rand()
                    b = np.random.rand()
                x_opt[2*i] = a
                x_opt[2*i+1] = b

    # Extract points
    points = np.empty((N, 2))
    for i in range(N):
        a = x_opt[2*i]
        b = x_opt[2*i+1]
        c = 1.0 - a - b
        px = 0.5 + 0.5 * b - 0.5 * a
        py = h * (1.0 - a - b)
        points[i] = [px, py]
        
    return points
# EVOLVE-BLOCK-END
