import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 55

def lj_energy(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    inv_r6 = 1.0 / (dist_sq ** 3)
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6)

def lj_grad(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    r_inv_8 = dist_sq_safe ** -4
    r_inv_14 = dist_sq_safe ** -7
    coeff = 24.0 * (r_inv_8 - 2.0 * r_inv_14)
    coeff = coeff * mask
    return np.sum(coeff[:, :, np.newaxis] * diff, axis=1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    hull = ConvexHull(verts)
    faces = hull.simplices
    face_centers = np.array([verts[f].mean(axis=0) for f in faces])
    face_centers /= np.linalg.norm(face_centers[0])
    
    pos_center = np.array([[0.0, 0.0, 0.0]])
    
    # Try a few initial radius configurations
    configs = [
        (1.12, 1.5, 1.8),
        (1.10, 1.45, 1.75),
        (1.15, 1.55, 1.85)
    ]
    
    best_pos = None
    best_energy = float('inf')
    
    for R1, R2, R3 in configs:
        if time.time() > deadline:
            break
            
        pos_inner = verts * R1
        pos_outer_fc = face_centers * R2
        pos_outer_v = verts * R3
        
        pos0 = np.vstack([pos_center, pos_inner, pos_outer_fc, pos_outer_v])
        
        try:
            res = minimize(lj_energy, pos0, jac=lj_grad, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.success:
                e = res.fun
                if e < best_energy:
                    best_energy = e
                    best_pos = res.x
        except Exception:
            continue
            
    if best_pos is not None:
        return best_pos
    else:
        # Fallback to seed if all fail
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12
