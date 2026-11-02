import numpy as np
from scipy.spatial.distance import pdist, squareform

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Initialize centers in a hexagonal grid
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    points = []
    for i in range(4):
        for j in range(3):
            p = i * u + j * v
            points.append(p)
            
    P = np.array(points)
    centroid = np.mean(P, axis=0)
    P = P - centroid
    
    # Normals for outer hexagon (flat-top)
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Vertices of unit hexagon
    vert_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    vertices = np.array([[np.cos(a), np.sin(a)] for a in vert_angles])
    
    V_N = vertices @ normals.T
    max_V_N = np.max(V_N, axis=0)
    
    def get_required_side(P):
        P_N = P @ normals.T
        proj = P_N + max_V_N[None, :]
        A_req = np.max(proj)
        return 2.0 * A_req / np.sqrt(3)

    best_P = P.copy()
    best_S = get_required_side(best_P)
    
    step_size = 0.05
    iterations = 1000
    
    for _ in range(iterations):
        S_curr = get_required_side(P)
        
        # Check overlap
        d = pdist(P)
        min_d = np.min(d)
        
        valid = min_d >= np.sqrt(3) - 1e-6
        
        if valid and S_curr < best_S:
            best_S = S_curr
            best_P = P.copy()
            step_size = max(0.01, step_size * 0.99)
        else:
            if not valid:
                step_size = max(0.01, step_size * 0.95)

        # Gradient
        P_N = P @ normals.T
        proj = P_N + max_V_N[None, :]
        max_proj_val = np.max(proj)
        
        grad = np.zeros_like(P)
        for i in range(N):
            # Active constraints
            active = np.where(proj[i, :] >= max_proj_val - 1e-3)[0]
            if len(active) > 0:
                g = np.sum(normals[active, :], axis=0)
                grad[i] = g / len(active)
            else:
                grad[i] = P[i] * 0.1

        # Repulsion
        d_sq = squareform(d)
        np.fill_diagonal(d_sq, 1)
        
        repulsion = np.zeros_like(P)
        # Vectorized repulsion
        # Force direction: P[i] - P[j]
        # Magnitude: 1/d^2 if d < sqrt(3)
        
        # Simple loop is fine for N=12
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = d_sq[i, j]
                    if dist < np.sqrt(3):
                        vec = P[i] - P[j]
                        repulsion[i] += vec / (dist * dist) * 0.5

        P = P - step_size * (grad * 0.5 + repulsion)

    # Final validation
    d_best = pdist(best_P)
    if np.min(d_best) < np.sqrt(3) - 1e-5:
        # Fallback
        best_P = P.copy()
        # Re-center
        centroid = np.mean(best_P, axis=0)
        best_P = best_P - centroid
        best_S = get_required_side(best_P)

    inner = []
    for p in best_P:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
