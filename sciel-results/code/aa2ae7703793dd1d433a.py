"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _circles_valid(circles, tol=1e-6):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2 - tol:
                return False
    return True

def _compute_inradii_from_voronoi(points, n_points=N):
    """
    Given point locations, compute the inradius of each Voronoi cell.
    This is an approximation: for each point, find the closest other points
    and estimate the inradius based on distances to neighbors.
    
    A more accurate approach: for each point i, the inradius of its Voronoi cell
    is the minimum distance from i to any edge of the cell. The edges are determined
    by the perpendicular bisectors between i and its neighbors.
    
    We use a simplified but effective heuristic: for each point, the inradius is
    approximately half the distance to the nearest other point, but we also need
    to account for boundary constraints.
    
    Actually, let's use a proper approach: for each point i, the Voronoi cell inradius
    is the minimum over all j≠i of the distance from i to the bisector plane of i and j,
    but only if j is a Voronoi neighbor. For a robust estimate without full Voronoi,
    we use: inradius_i = min(0.5 * min_j dist(i,j), x_i, 1-x_i, y_i, 1-y_i)
    But this is not quite right either.
    
    Let me think more carefully. The Voronoi cell of point i is the set of points closer
    to i than to any other point. The inradius is the radius of the largest circle centered
    at i that fits inside this cell. This is the minimum distance from i to any edge of the cell.
    
    The edges of the Vororonoi cell are segments of the perpendicular bisectors between i and
    its Voronoi neighbors. The distance from i to the bisector between i and j is 0.5*dist(i,j).
    But the edge is only part of that bisector. The inradius is the minimum over all Voronoi
    neighbors j of the distance from i to the edge segment on the bisector.
    
    For a practical approach without computing the full Voronoi diagram, we can use:
    inradius_i = min( min_{j neighbor} 0.5*dist(i,j), x_i, 1-x_i, y_i, 1-y_i )
    
    But we need to know which j are Voronoi neighbors. A common approximation is to use
    the k-nearest neighbors. Let's use all other points and take the minimum of 0.5*dist(i,j)
    over all j, plus the boundary constraints. This gives a lower bound on the true inradius.
    
    Actually, for the purpose of optimization, we can define a smooth surrogate:
    For each point i, define f_i = min( x_i, 1-x_i, y_i, 1-y_i, min_{j≠i} 0.5*||p_i - p_j|| )
    This is the maximum radius of a circle centered at p_i that doesn't overlap with any other
    circle centered at p_j (with the same radius constraint) and stays in the square.
    
    But wait - if all circles have the same radius r, then we need r <= 0.5*dist(i,j) for all i,j
    and r <= min(x_i, 1-x_i, y_i, 1-y_i). So the maximum common radius is:
    r_max = min( min_i min(x_i, 1-x_i, y_i, 1-y_i), min_{i<j} 0.5*||p_i-p_j|| )
    
    But we want to maximize the SUM of radii, not the minimum radius. So we can have different radii.
    
    Let me reconsider. If we place circles at points p_i with radii r_i, the constraints are:
    1. r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i (inside square)
    2. r_i + r_j <= ||p_i - p_j|| for all i ≠ j (disjoint)
    
    To maximize sum(r_i), we can use a smooth surrogate. Let's define for each i:
    g_i = min( x_i, 1-x_i, y_i, 1-y_i, min_{j≠i} (||p_i - p_j|| - r_j) )
    And set r_i = g_i. But this is circular.
    
    Better approach: use an iterative method. Start with equal radii, then adjust.
    
    Let me use a different strategy: 
    1. Optimize the positions p_i to maximize the minimum distance between any two points,
       and also maximize the minimum distance to the boundary. This gives a good initial packing.
    2. Then, given the positions, compute optimal radii by solving a linear program or using
       a greedy/water-filling approach.
    
    For step 2, given positions, the optimal radii can be found by:
    r_i = min( b_i, min_{j≠i} (d_ij - r_j) )
    where b_i = min(x_i, 1-x_i, y_i, 1-y_i) and d_ij = ||p_i - p_j||.
    This is a system that can be solved iteratively.
    
    Let me implement a practical algorithm:
    1. Use a greedy placement with local optimization.
    2. Use scipy.optimize to maximize a smooth objective.
    
    Given the time constraint, let me use a simpler but effective approach:
    - Start with a grid or random initial positions.
    - Use gradient ascent on a smooth surrogate of the sum of radii.
    - The surrogate: for each i, r_i = softmin of the constraints.
    """
    pass

def _softmin(values, tau=0.01):
    """Soft minimum using log-sum-exp trick."""
    values = np.array(values)
    return -tau * np.log(np.sum(np.exp(-values / tau)))

def _compute_radii_from_positions(positions, tau=0.01):
    """
    Given positions, compute approximate optimal radii using a smooth approach.
    r_i is limited by boundary and by distances to other circles.
    """
    n = len(positions)
    positions = np.array(positions)
    
    # Boundary limits
    b = np.minimum(np.minimum(positions[:, 0], 1 - positions[:, 0]),
                   np.minimum(positions[:, 1], 1 - positions[:, 1]))
    
    # Distance limits: for each i, the limit from j is 0.5 * dist(i,j) if equal radii
    # But for variable radii, it's more complex. Let's use an iterative approach.
    
    # Start with equal radii estimate
    r = np.zeros(n)
    for i in range(n):
        # Distance to other points
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        r[i] = min(b[i], 0.5 * min_dist)
    
    # Iterate to refine
    for _ in range(50):
        new_r = np.zeros(n)
        for i in range(n):
            limits = [b[i]]
            for j in range(n):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                limits.append(dist - r[j])
            new_r[i] = max(0, min(limits))
        if np.allclose(r, new_r, atol=1e-8):
            break
        r = new_r
    
    return r

def _objective_and_grad(params, tau=0.01):
    """
    Objective: maximize sum of radii.
    params: 2N vector of positions.
    Returns negative of sum of radii (for minimization).
    """
    n = N
    positions = params.reshape(n, 2)
    
    # Boundary limits
    b = np.minimum(np.minimum(positions[:, 0], 1 - positions[:, 0]),
                   np.minimum(positions[:, 1], 1 - positions[:, 1]))
    
    # Compute radii iteratively
    r = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        r[i] = min(b[i], 0.5 * min_dist)
    
    for _ in range(20):
        new_r = np.zeros(n)
        for i in range(n):
            limits = [b[i]]
            for j in range(n):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                limits.append(dist - r[j])
            new_r[i] = max(0, min(limits))
        r = new_r
    
    total_r = np.sum(r)
    
    # Compute gradient numerically if needed, but let's use finite differences for now
    # For efficiency, let's return the negative objective
    return -total_r

def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
    
    positions = np.array(cells[:N])
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            pos = positions.copy()
        else:
            # Random initialization with some structure
            pos = np.random.uniform(0.1, 0.9, size=(N, 2))
        
        # Local optimization using scipy
        try:
            result = minimize(
                _objective_and_grad,
                pos.flatten(),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 50, 'gtol': 1e-4}
            )
            
            if result.success or result.fun < best_score:
                pos_opt = result.x.reshape(N, 2)
                radii = _compute_radii_from_positions(pos_opt)
                
                # Check validity
                circles = np.column_stack([pos_opt, radii])
                if _circles_valid(circles, tol=1e-5):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_result = circles
        except Exception as e:
            pass
    
    # If no valid result found, use the initial grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
