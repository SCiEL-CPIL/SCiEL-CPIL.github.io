import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _hexagon_vertices(x, y, angle, side=1.0):
    """Return the 6 vertices of a regular hexagon."""
    angles = angle + np.arange(6) * math.pi / 3.0
    r = side
    return np.array([
        [x + r * math.cos(a), y + r * math.sin(a)] for a in angles
    ])

def _compute_container_side(pts, beta):
    """
    Compute the minimum side length S of a regular hexagon with orientation beta
    that encloses the set of points pts. The center of the hexagon is chosen
    optimally (center of the bounding box).
    """
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    
    # Normals of the hexagon sides
    # The hexagon is defined by 6 half-planes.
    # If beta is the angle of the first normal, the normals are at beta, beta+pi/3, ...
    # The width in direction n is max(p.n) - min(p.n).
    # The container must satisfy: max(p.n) - min(p.n) <= S * sqrt(3)
    # So S >= (max - min) / sqrt(3)
    
    angles = beta + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Project points onto normals
    # pts: (M, 2)
    # normals: (6, 2)
    projs = pts @ normals.T  # (M, 6)
    
    max_projs = np.max(projs, axis=0)
    min_projs = np.min(projs, axis=0)
    
    widths = max_projs - min_projs
    max_width = np.max(widths)
    
    S = max_width / math.sqrt(3)
    
    # Optimal center is the midpoint of the max/min projections in the "tightest" direction?
    # Actually, to minimize S, we should center the container such that the max deviation is minimized.
    # If we fix S, the center (cx, cy) must satisfy:
    # cx*nx + cy*ny <= max_proj - S*sqrt(3)/2
    # cx*nx + cy*ny >= min_proj + S*sqrt(3)/2
    # This is a linear feasibility problem.
    # For the purpose of the objective function in optimization, we can just return S.
    # The center can be calculated later for the final output.
    # However, to be safe, we can estimate the center as the average of the midpoints.
    mid_projs = (max_projs + min_projs) / 2.0
    # We need cx, cy such that cx*nx + cy*ny = mid_projs for all k?
    # This is overdetermined. Least squares solution:
    # [nx ny] * [cx cy]^T = mid_projs
    A = normals.T # (2, 6)
    b = mid_projs # (6,)
    try:
        cx, cy = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        cx, cy = np.mean(pts, axis=0)
        
    return S, cx, cy

def _overlap_penalty(positions, rotations, threshold=2.0):
    """
    Calculate a penalty for overlapping hexagons.
    Uses a conservative distance threshold of 2.0 (vertex-to-vertex contact for same orientation).
    If rotations differ, 2.0 is still a safe lower bound for non-overlap.
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < threshold:
                # Quadratic penalty
                penalty += (threshold - d) ** 2
    return penalty

def _objective(params, N_hexes=N):
    """
    Objective function for the optimizer.
    params: [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, beta]
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e6
        
    positions = np.zeros((N_hexes, 2))
    rotations = np.zeros(N_hexes)
    
    for i in range(N_hexes):
        positions[i, 0] = params[3*i]
        positions[i, 1] = params[3*i + 1]
        rotations[i] = params[3*i + 2]
        
    beta = params[3*N_hexes]
    
    # Compute all vertices
    all_vertices = []
    for i in range(N_hexes):
        verts = _hexagon_vertices(positions[i, 0], positions[i, 1], rotations[i])
        all_vertices.append(verts)
        
    pts = np.vstack(all_vertices)
    
    # Compute container side length
    S, cx, cy = _compute_container_side(pts, beta)
    
    # Add overlap penalty
    pen = _overlap_penalty(positions, rotations)
    
    # Weight the penalty heavily
    return S + 10.0 * pen

def _generate_initial_guesses():
    """
    Generate a list of initial guesses for the optimizer.
    """
    guesses = []
    
    # Guess 1: 3x3 Grid + 2
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    axial = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (-1,1), (3,1) # Two extra
    ]
    pos = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pos.append([x, y, 0.0])
    guesses.append(np.array(pos + [[0.0]]).flatten())
    
    # Guess 2: Flower (7) + 4 corners
    # Center (0,0)
    # Ring 1: 6 hexes
    # Corners: 4 hexes placed further out
    pos = [[0,0,0.0]]
    for k in range(6):
        angle = k * math.pi / 3.0
        # Distance to touch is 2.0? No, for flat top, horizontal distance is sqrt(3)?
        # Let's just use the axial grid for the flower part.
        # Axial for flower:
        pass # Skip complex generation, use a simple radial guess
    # Simple radial guess for 11 points
    center = [0,0,0.0]
    ring1 = []
    for k in range(6):
        a = k * math.pi / 3.0
        x = 2.0 * math.cos(a) # Approx
        y = 2.0 * math.sin(a)
        ring1.append([x, y, 0.0])
    ring2 = []
    for k in range(4):
        a = (k + 0.5) * math.pi / 2.0
        x = 4.0 * math.cos(a)
        y = 4.0 * math.sin(a)
        ring2.append([x, y, 0.0])
    guesses.append(np.array([center] + ring1 + ring2 + [[0.0]]).flatten())

    # Guess 3: Random perturbations of Guess 1
    np.random.seed(42)
    base = guesses[0]
    for _ in range(5):
        perturbed = base.copy()
        # Perturb positions slightly
        for i in range(N):
            perturbed[3*i] += np.random.normal(0, 0.2)
            perturbed[3*i+1] += np.random.normal(0, 0.2)
            perturbed[3*i+2] += np.random.normal(0, 0.1)
        guesses.append(perturbed)

    return guesses

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    guesses = _generate_initial_guesses()
    
    best_S = float('inf')
    best_params = None
    best_beta = 0.0
    
    for guess in guesses:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            # Use Nelder-Mead for robustness with non-smooth penalty
            res = minimize(_objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_S:
                # Check if penalty is low (no overlaps)
                positions = np.zeros((N, 2))
                rotations = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = res.x[3*i]
                    positions[i, 1] = res.x[3*i + 1]
                    rotations[i] = res.x[3*i + 2]
                pen = _overlap_penalty(positions, rotations)
                if pen < 1e-3:
                    best_S = res.fun
                    best_params = res.x
                    best_beta = res.x[3*N]
        except Exception as e:
            pass
            
    # If no good result, fallback to a simple grid
    if best_params is None:
        best_params = guesses[0]
        best_beta = best_params[3*N]
        # Recalculate S
        positions = np.zeros((N, 2))
        rotations = np.zeros(N)
        for i in range(N):
            positions[i, 0] = best_params[3*i]
            positions[i, 1] = best_params[3*i + 1]
            rotations[i] = best_params[3*i + 2]
        all_vertices = []
        for i in range(N):
            verts = _hexagon_vertices(positions[i, 0], positions[i, 1], rotations[i])
            all_vertices.append(verts)
        pts = np.vstack(all_vertices)
        best_S, _, _ = _compute_container_side(pts, best_beta)

    # Final Polish: Refine the center and S exactly
    positions = np.zeros((N, 2))
    rotations = np.zeros(N)
    for i in range(N):
        positions[i, 0] = best_params[3*i]
        positions[i, 1] = best_params[3*i + 1]
        rotations[i] = best_params[3*i + 2]
        
    all_vertices = []
    for i in range(N):
        verts = _hexagon_vertices(positions[i, 0], positions[i, 1], rotations[i])
        all_vertices.append(verts)
    pts = np.vstack(all_vertices)
    
    # Optimize beta and center for this fixed configuration
    def final_obj(params):
        beta = params[0]
        cx = params[1]
        cy = params[2]
        angles = beta + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        widths = max_projs - min_projs
        S = np.max(widths) / math.sqrt(3)
        return S

    start_params = [best_beta, 0, 0]
    # Try a few starting betas
    for b in [best_beta, 0.0, math.pi/6]:
        try:
            res = minimize(final_obj, [b, 0, 0], method='Nelder-Mead', options={'maxiter': 200})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x[0]
                best_cx = res.x[1]
                best_cy = res.x[2]
        except:
            pass

    # Ensure we have a center
    if 'best_cx' not in locals():
        _, best_cx, best_cy = _compute_container_side(pts, best_beta)

    # Prepare output
    inner = []
    for i in range(N):
        x = float(positions[i, 0])
        y = float(positions[i, 1])
        angle_deg = math.degrees(float(rotations[i]))
        inner.append([x, y, angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
