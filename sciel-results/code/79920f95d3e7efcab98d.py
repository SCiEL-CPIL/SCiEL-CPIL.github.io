import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap_depth_grad(A, B):
    """
    Calculate overlap depth and gradient for two convex polygons A and B.
    A, B are arrays of shape (6, 2).
    Returns (depth, grad_A, grad_B) where grad is the direction to move A and B to separate.
    If disjoint, returns (0, 0, 0).
    """
    min_depth = float('inf')
    best_ax = None
    
    axes = []
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            length = np.linalg.norm(edge)
            if length < 1e-9:
                continue
            normal = np.array([-edge[1], edge[0]]) / length
            axes.append(normal)
            
    for ax in axes:
        proj_a = np.dot(A, ax)
        min_a, max_a = np.min(proj_a), np.max(proj_a)
        proj_b = np.dot(B, ax)
        min_b, max_b = np.min(proj_b), np.max(proj_b)
        
        if max_a <= min_b or max_b <= min_a:
            return 0.0, np.zeros(2), np.zeros(2)
        
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
            best_ax = ax
            
    if best_ax is None:
        return 0.0, np.zeros(2), np.zeros(2)
        
    # Determine direction to separate
    # If A is to the left of B on the axis, A should move left (-ax) and B right (+ax)
    center_a = np.mean(A, axis=0)
    center_b = np.mean(B, axis=0)
    diff_center = center_b - center_a
    if np.dot(diff_center, best_ax) > 0:
        # B is in +ax direction relative to A
        grad_a = -best_ax
        grad_b = best_ax
    else:
        grad_a = best_ax
        grad_b = -best_ax
        
    return min_depth, grad_a, grad_b

def _containment_penalty_grad(inner_verts, outer_verts):
    """
    Calculate sum of distances of inner vertices outside the outer hexagon and the gradient.
    Returns (penalty, grad_inner) where grad_inner is the direction to move the inner hex to reduce penalty.
    """
    penalty = 0.0
    grad = np.zeros(2)
    n_out = len(outer_verts)
    
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-9:
                continue
            # Normal pointing inward (assuming CCW outer)
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                dist = -sd
                penalty += dist
                # To reduce penalty, move iv in direction nrm (inward)
                grad += nrm * (1.0 / 6.0) # Average over vertices
    return penalty, grad

def _make_grid_init(nr, nc):
    """Create initial positions for a grid."""
    pos = []
    for r in range(nr):
        for c in range(nc):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _cluster_init():
    """1 center, 6 ring 1, 5 ring 2."""
    pos = [[0.0, 0.0]]
    for i in range(6):
        a = np.radians(60 * i)
        pos.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    indices = [0, 2, 4, 6, 8]
    for idx in indices:
        a = np.radians(60 * idx)
        pos.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _physics_solve(S, init_params, max_iters=2000, time_limit=10.0):
    """
    Solve for positions of N inner hexagons inside outer hexagon of side S.
    init_params: shape (N, 3) [x, y, angle]
    Returns: (success, final_params)
    """
    t0 = time.time()
    params = init_params.copy()
    lr = 0.1
    min_lr = 1e-5
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
            
        # Compute vertices
        inner_verts_list = []
        for i in range(N):
            inner_verts_list.append(_hex_verts(params[i, 0], params[i, 1], 1.0, params[i, 2]))
        
        outer_verts = _hex_verts(0.0, 0.0, S, 0.0)
        
        forces = np.zeros((N, 2))
        max_overlap = 0.0
        
        # Inner-Inner Repulsion
        for i in range(N):
            for j in range(i + 1, N):
                depth, grad_i, grad_j = _sat_overlap_depth_grad(inner_verts_list[i], inner_verts_list[j])
                if depth > 0:
                    max_overlap = max(max_overlap, depth)
                    # Force proportional to depth
                    f = depth * 10.0
                    forces[i] += f * grad_i
                    forces[j] += f * grad_j
        
        # Containment
        for i in range(N):
            pen, grad = _containment_penalty_grad(inner_verts_list[i], outer_verts)
            if pen > 0:
                max_overlap = max(max_overlap, pen)
                f = pen * 10.0
                forces[i] += f * grad
        
        if max_overlap < 1e-6:
            return True, params
            
        # Update positions
        # Simple gradient descent with damping
        params[:, :2] += lr * forces
        
        # Simple adaptive learning rate
        if it % 100 == 0:
            if max_overlap < 0.01:
                lr = max(min_lr, lr * 0.9)
            else:
                lr = min(0.5, lr * 1.1)
                
    # Final check
    inner_verts_list = []
    for i in range(N):
        inner_verts_list.append(_hex_verts(params[i, 0], params[i, 1], 1.0, params[i, 2]))
    outer_verts = _hex_verts(0.0, 0.0, S, 0.0)
    
    for i in range(N):
        for j in range(i + 1, N):
            depth, _, _ = _sat_overlap_depth_grad(inner_verts_list[i], inner_verts_list[j])
            if depth > 1e-4:
                return False, params
        pen, _ = _containment_penalty_grad(inner_verts_list[i], outer_verts)
        if pen > 1e-4:
            return False, params
            
    return True, params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0 # Safety margin
    
    best_S = float('inf')
    best_params = None
    
    inits = [
        _make_grid_init(3, 4),
        _make_grid_init(4, 3),
        _cluster_init()
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(3):
        r = rng.uniform(-2, 2, size=(N, 2))
        inits.append(r)
        
    for init_pos in inits:
        if time.time() > DEADLINE - 20:
            break
            
        # Start with a large S that is definitely valid
        S_start = 6.0
        # Initialize angles to 0
        init_params = np.zeros((N, 3))
        init_params[:, :2] = init_pos
        init_params[:, 2] = 0.0
        
        # Squeeze: Decrease S until it fails
        S = S_start
        step = 0.2
        current_params = init_params.copy()
        
        # First, ensure we have a valid config at S_start
        success, current_params = _physics_solve(S_start, current_params, max_iters=500, time_limit=5.0)
        if not success:
            continue
            
        while S > 2.0: # Lower bound guess
            if time.time() > DEADLINE - 5:
                break
                
            new_S = S - step
            # Try to solve for new_S using current_params as init
            success, new_params = _physics_solve(new_S, current_params, max_iters=1000, time_limit=15.0)
            
            if success:
                S = new_S
                current_params = new_params
                if S < best_S:
                    best_S = S
                    best_params = current_params.copy()
                # Reduce step size as we get closer to the limit
                step = max(0.05, step * 0.9)
            else:
                # Failed to fit in new_S, try smaller step?
                # For simplicity, just stop this init's squeeze
                break
                
    if best_params is None:
        # Fallback: Return a known valid grid with large S
        grid = _make_grid_init(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 6.0, 0.0
        
    inner = []
    for i in range(N):
        inner.append([float(best_params[i,0]), float(best_params[i,1]), float(best_params[i,2])])
        
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
