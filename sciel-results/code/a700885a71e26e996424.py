"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import fsolve, minimize
import time
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _generate_graph(n):
    """
    Generate a random contact graph for n circles.
    We assume a roughly hexagonal packing, so each circle has ~6 neighbors.
    We use a k-NN graph plus random additions to ensure connectivity and sufficient constraints.
    """
    rng = np.random.default_rng(12345)
    # Initial positions for graph generation (grid)
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    for i in range(n):
        pos[i, 0] = (i % side + 0.5) / side
        pos[i, 1] = (i // side + 0.5) / side
        
    # Calculate distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Find k-NN for each node
    k = 8 # Slightly more than 6 for redundancy
    graph_edges = set()
    for i in range(n):
        # Sort neighbors by distance
        order = np.argsort(dists[i])
        order = order[order != i]
        for j in order[:k]:
            if i < j:
                graph_edges.add((i, j))
            else:
                graph_edges.add((j, i))
                
    # Add some random long-range connections to ensure global connectivity
    for _ in range(n):
        i, j = rng.choice(n, 2, replace=False)
        if i < j:
            graph_edges.add((i, j))
        else:
            graph_edges.add((j, i))
            
    edges = list(graph_edges)
    # Ensure we have enough edges to determine the system (3n variables, need ~3n constraints)
    # Walls provide some constraints, but let's aim for a robust graph.
    return edges

def _solve_packing_homotopy(edges, deadline, n=26):
    """
    Solve the packing using homotopy (continuation method).
    We scale the 'target' radii from 0 to 1.
    Variables: x_i, y_i, r_i
    Constraints:
    1. Wall contacts (softened via penalty or active set? 
       For fsolve, we need exactly 3n equations.
       Strategy: 
       - Assume all circles touch at least one wall or another circle.
       - Actually, a simpler approach for homotopy:
         Define a function F(params, scale) = 0.
         We want to maximize sum(r).
         Let's define the system such that circles are as large as possible.
         
         Alternative: 
         Use a penalty method inside fsolve? No, fsolve needs 0.
         
         Better approach for Homotopy:
         1. Start with small radii r_i = epsilon.
         2. Increase a parameter lambda.
         3. Solve for positions and radii such that:
            - Circles are tangent to neighbors in the graph: dist(i,j) = r_i + r_j
            - Circles are tangent to walls: x_i = r_i, x_i = 1-r_i, etc.
            
         This is under-determined if not all circles are constrained.
         To make it determined, we can fix the center of mass or assume specific wall contacts.
         
         Let's use a different formulation:
         Maximize S = sum(r_i)
         Subject to:
           r_i + r_j <= dist(i,j)
           r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
           
         This is an optimization problem. Homotopy applies to finding the solution path as we change a parameter.
         Let's define a parameter alpha in [0, 1].
         Target radii R_i(alpha) = alpha * R_i_max_est
         
         We solve for positions X(alpha) and actual radii r(alpha) such that:
         1. The circles are in equilibrium (force balance?) No, that's physics.
         2. The constraints are tight.
         
         Let's try a specific system of equations:
         For each circle i:
         - It touches its k-neighbors in the graph.
         - It touches the wall if it's on the boundary.
         
         This is tricky to set up exactly 3n equations.
         
         Let's use a simpler, robust homotopy:
         1. Initialize with a grid.
         2. Define a "compression" factor c.
         3. At each step, we want to maximize the minimum clearance.
         
         Actually, let's stick to the standard "Tight Packing" equations:
         For each edge (i,j) in the graph:
           (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 = 0
         For each circle i, we need 3 equations.
         We can assume each circle touches 3 neighbors/walls.
         
         Let's use `scipy.optimize.minimize` with a very specific objective that mimics the homotopy path,
         but since the prompt asks for a homotopy method, let's implement a simple path-following scheme.
         
         Path Following:
         1. Define a sequence of target configurations.
         2. Start with a loose grid (large gaps).
         3. Gradually reduce the "repulsion" or increase the "size" requirement.
         
         Let's implement a simpler, effective variant: 
         **Gradient Flow with Homotopy-like Scaling**
         Start with small radii. 
         Iteratively:
         1. Increase radii by a small step.
         2. Run a few steps of gradient descent to resolve overlaps (move centers).
         3. Repeat.
         
         This is effectively a "soft" homotopy.
    """
    
    # Initialize
    side = int(np.ceil(np.sqrt(n)))
    xy = np.zeros((n, 2))
    r = np.zeros(n)
    for i in range(n):
        xy[i, 0] = (i % side + 0.5) / side
        xy[i, 1] = (i // side + 0.5) / side
        r[i] = 0.01
        
    # Precompute edges
    # Using the generated graph
    # We will use a force-based relaxation that respects the graph structure.
    
    # Homotopy steps
    steps = 50
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Increase target radius
        # Target radius scales with 1/sqrt(n) roughly, but we let it grow.
        # We want to push them to be as big as possible.
        # Let's add a "pressure" term.
        
        # Gradient step to maximize sum(r) while resolving overlaps
        # Objective: -sum(r) + Penalty(Overlaps) + Penalty(Walls)
        
        # Calculate gradients
        grad_xy = np.zeros((n, 2))
        grad_r = np.zeros(n)
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                    # Random push
                    angle = np.random.rand() * 2 * np.pi
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Force to separate
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    grad_xy[i, 0] -= fx
                    grad_xy[i, 1] -= fy
                    grad_xy[j, 0] += fx
                    grad_xy[j, 1] += fy
                    # Force to shrink
                    grad_r[i] += overlap
                    grad_r[j] += overlap
                    # We want to maximize r, so gradient of -r is -1.
                    # But we have a penalty.
                    # Let's just use a simple iterative solver.
        
        # Penalty for walls
        for i in range(n):
            # Left
            if xy[i, 0] < r[i]:
                grad_xy[i, 0] += (r[i] - xy[i, 0])
                grad_r[i] += (r[i] - xy[i, 0])
            # Right
            if 1.0 - xy[i, 0] < r[i]:
                grad_xy[i, 0] -= (r[i] - (1.0 - xy[i, 0]))
                grad_r[i] += (r[i] - (1.0 - xy[i, 0]))
            # Bottom
            if xy[i, 1] < r[i]:
                grad_xy[i, 1] += (r[i] - xy[i, 1])
                grad_r[i] += (r[i] - xy[i, 1])
            # Top
            if 1.0 - xy[i, 1] < r[i]:
                grad_xy[i, 1] -= (r[i] - (1.0 - xy[i, 1]))
                grad_r[i] += (r[i] - (1.0 - xy[i, 1]))
                
        # Update
        # We want to increase r, so we move r in the direction of +grad_r?
        # No, grad_r is the gradient of the penalty. We want to reduce penalty.
        # So we should move r to reduce the penalty, i.e., decrease r if there's overlap.
        # But we want to maximize r.
        
        # Let's use a simple "expand and fix" strategy.
        # 1. Expand all radii by a small amount.
        # 2. Resolve overlaps by moving centers.
        
        expansion = 0.005
        r += expansion
        
        # Resolve overlaps via iterative projection (like Lloyd's algorithm for circles)
        for _ in range(10):
            for i in range(n):
                # Check walls
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                if 1.0 - xy[i, 0] < r[i]:
                    xy[i, 0] = 1.0 - r[i]
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                if 1.0 - xy[i, 1] < r[i]:
                    xy[i, 1] = 1.0 - r[i]
                    
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j]:
                        # Move apart
                        if dist < 1e-6:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-6
                        overlap = r[i] + r[j] - dist
                        # Move each by half the overlap
                        shift_x = (dx / dist) * (overlap / 2)
                        shift_y = (dy / dist) * (overlap / 2)
                        xy[i, 0] += shift_x
                        xy[i, 1] += shift_y
                        xy[j, 0] -= shift_x
                        xy[j, 1] -= shift_y
                        
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # 1. Homotopy-like expansion
    if time.time() < deadline - 20.0:
        xy, r = _solve_packing_homotopy(None, deadline - 20.0, N)
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        for i in range(N):
            best_xy[i, 0] = (i % side + 0.5) / side
            best_xy[i, 1] = (i // side + 0.5) / side
            best_r[i] = 0.4 / side
        best_total, best_valid = _eval_packing(best_xy, best_r)
        
    # 2. Polish with L-BFGS-B to refine the homotopy result
    if best_valid and time.time() < deadline - 5.0:
        def objective(p):
            x = p[0::2]
            y = p[1::2]
            # Radii are fixed from the homotopy step? 
            # No, we should optimize radii too.
            # But L-BFGS-B needs bounds.
            # Let's just polish positions for fixed radii first, then try to expand radii.
            # Actually, let's use a combined optimization.
            pass
            
        # Let's use a simpler polish: Local Search on the homotopy result
        # We can reuse the logic from the previous program's local search but simplified.
        
        # For now, just return the homotopy result if it's good.
        # If we have time, we can add a gradient polish.
        
        # Gradient Polish
        def obj(p):
            xy = p[:2*N].reshape(N, 2)
            r = p[2*N:]
            # Penalty for overlaps
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy[i,0] - xy[j,0]
                    dy = xy[i,1] - xy[j,1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r[i] + r[j]:
                        pen += (r[i] + r[j] - d)**2
            # Penalty for walls
            for i in range(N):
                if xy[i,0] < r[i]: pen += (r[i] - xy[i,0])**2
                if 1-xy[i,0] < r[i]: pen += (r[i] - (1-xy[i,0]))**2
                if xy[i,1] < r[i]: pen += (r[i] - xy[i,1])**2
                if 1-xy[i,1] < r[i]: pen += (r[i] - (1-xy[i,1]))**2
            
            # Objective: Maximize sum(r) - Penalty
            return -np.sum(r) + 100.0 * pen
            
        p0 = np.concatenate([best_xy.ravel(), best_r])
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.001, 0.4)) # r
            
        try:
            res = minimize(obj, p0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            p_opt = res.x
            xy_opt = p_opt[:2*N].reshape(N, 2)
            r_opt = p_opt[2*N:]
            total_opt, valid_opt = _eval_packing(xy_opt, r_opt)
            if valid_opt and total_opt > best_total:
                best_total = total_opt
                best_valid = True
                best_xy = xy_opt
                best_r = r_opt
        except Exception:
            pass

    # Final Validation
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
