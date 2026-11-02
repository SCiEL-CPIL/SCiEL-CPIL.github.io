"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Barycentric coordinates: each point has (l1, l2, l3) with l1+l2+l3=1, li>=0
    # We parameterize using l1, l2 >= 0, l1+l2 <= 1, and l3 = 1-l1-l2
    
    # Area of triangle with vertices in barycentric coords:
    # If points have barycentric coords (a1,a2,a3), (b1,b2,b3), (c1,c2,c3)
    # then the ratio of the triangle area to the reference triangle area is
    # |det([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])|
    # The reference triangle (0,0),(1,0),(0.5,sqrt(3)/2) has area sqrt(3)/4
    # So actual area = sqrt(3)/4 * |det|
    
    # We want to maximize the minimum |det| over all C(11,3)=165 triples
    
    # Parameterization: 11 points, each with 2 free barycentric coords (l1, l2)
    # l3 = 1 - l1 - l2
    
    def bary_to_cartesian(bary):
        """Convert barycentric coords (l1, l2) to Cartesian in the unit equilateral triangle.
        Reference triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
        Point = l1*A + l2*B + l3*C where l1+l2+l3=1
        So x = l2*1 + l3*0.5, y = l3*sqrt(3)/2
        With l3 = 1-l1-l2:
        x = l2 + (1-l1-l2)*0.5 = l2 + 0.5 - 0.5*l1 - 0.5*l2 = 0.5 - 0.5*l1 + 0.5*l2
        y = (1-l1-l2)*sqrt(3)/2
        """
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])
    
    def triangle_area_ratio(bary_coords, i, j, k):
        """Compute the ratio of triangle area to reference triangle area.
        This equals |det of the 3x3 matrix of barycentric coords|."""
        M = bary_coords[[i, j, k], :]
        # Need full 3x3 with l3
        l3 = 1 - M[:, 0] - M[:, 1]
        M_full = np.column_stack([M[:, 0], M[:, 1], l3])
        return abs(np.linalg.det(M_full))
    
    def objective(params):
        """Negative soft-min of triangle areas (we minimize negative to maximize)."""
        bary = params.reshape(N, 2)
        # Clip to valid barycentric range with small margin
        bary = np.clip(bary, 0.001, 0.999)
        # Ensure l1+l2 <= 1
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.998
        if np.any(mask):
            bary[mask, 0] *= 0.998 / s[mask]
            bary[mask, 1] *= 0.998 / s[mask]
        
        # Compute all 165 triangle area ratios
        areas = []
        for i, j, k in itertools.combinations(range(N), 3):
            areas.append(triangle_area_ratio(bary, i, j, k))
        areas = np.array(areas)
        
        # Soft minimum using log-sum-exp: -log(sum(exp(-a_i / t))) * t
        # As t -> 0, this approaches min(a_i)
        t = 0.01
        max_neg = np.max(-areas / t)
        soft_min = -t * (max_neg + np.log(np.sum(np.exp(-areas / t - max_neg))))
        
        return -soft_min  # Negate because we want to maximize
    
    def objective_grad(params):
        """Numerical gradient via finite differences (for simplicity and robustness)."""
        grad = np.zeros_like(params)
        h = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            params_minus = params.copy()
            params_minus[i] -= h
            grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * h)
        return grad
    
    # Initial guess: spread points somewhat evenly
    np.random.seed(42)
    # Start with points at various positions in the triangle
    # Use a grid-like initial placement
    initial_bary = np.zeros((N, 2))
    # Place points at vertices, edge midpoints, and center
    positions = [
        (0.05, 0.05),   # near A
        (0.05, 0.95),   # near B  
        (0.95, 0.05),   # near C
        (0.5, 0.5),     # center
        (0.33, 0.33),   # centroid
        (0.1, 0.4),
        (0.4, 0.1),
        (0.5, 0.3),
        (0.3, 0.5),
        (0.2, 0.2),
        (0.7, 0.7),
    ]
    for i, (l1, l2) in enumerate(positions[:N]):
        initial_bary[i] = [l1, l2]
    
    # Optimize using L-BFGS-B
    result = minimize(
        objective,
        initial_bary.flatten(),
        method='L-BFGS-B',
        jac=objective_grad,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    # Get final barycentric coords
    final_bary = result.x.reshape(N, 2)
    final_bary = np.clip(final_bary, 0.0, 1.0)
    s = final_bary[:, 0] + final_bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        final_bary[mask, 0] /= s[mask]
        final_bary[mask, 1] /= s[mask]
    
    # Convert to Cartesian
    cart = bary_to_cartesian(final_bary)
    
    # Ensure points are strictly inside (small margin from edges)
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    # For y, max is sqrt(3)/2
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
