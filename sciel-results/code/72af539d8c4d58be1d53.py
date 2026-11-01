import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _halton_sphere(n):
    """Generate points using Halton sequence projected to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        theta = 2.0 * math.pi * x
        phi = math.acos(2.0 * y - 1.0)
        px = math.sin(phi) * math.cos(theta)
        py = math.sin(phi) * math.sin(theta)
        pz = math.cos(phi)
        points.append([px, py, pz])
    
    pts = np.array(points)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def _stereo_to_sphere(u):
    """Map 2D stereographic coords (stored as Nx3 with z=0) to unit sphere."""
    u0 = u[:, 0]
    u1 = u[:, 1]
    u_sq = u0**2 + u1**2
    factor = 2.0 / (1.0 + u_sq)
    xyz = np.empty((u.shape[0], 3))
    xyz[:, 0] = u0 * factor
    xyz[:, 1] = u1 * factor
    xyz[:, 2] = (1.0 - u_sq) * factor
    return xyz

def _sphere_to_stereo(xyz):
    """Map unit sphere points to 2D stereographic coords (Nx3 with z=0)."""
    z = xyz[:, 2]
    denom = 1.0 - z
    near_south = np.abs(denom) < 1e-10
    u = np.zeros((xyz.shape[0], 3))
    regular = ~near_south
    u[regular, 0] = xyz[regular, 0] / denom[regular]
    u[regular, 1] = xyz[regular, 1] / denom[regular]
    if np.any(near_south):
        denom_s = 1.0 + z[near_south]
        u[near_south, 0] = -xyz[near_south, 0] / denom_s
        u[near_south, 1] = -xyz[near_south, 1] / denom_s
    return u

def _energy_grad_stereo(u):
    """Compute log energy and gradient w.r.t. stereographic coords."""
    xyz = _stereo_to_sphere(u)
    diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    np.fill_diagonal(dist_sq, 1.0)
    
    log_d2 = np.log(dist_sq)
    np.fill_diagonal(log_d2, 0.0)
    energy = -0.25 * np.sum(log_d2)
    
    forces_xyz = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    u0 = u[:, 0]
    u1 = u[:, 1]
    u_sq = u0**2 + u1**2
    denom2 = (1.0 + u_sq)**2
    
    # Chain rule: grad_u_j = sum_k dxyz_k/du_j * forces_xyz_k
    grad_u = np.zeros((u.shape[0], 3))
    grad_u[:, 0] = (2.0 * (1.0 + u1**2) * forces_xyz[:, 0] - 4.0 * u0 * u1 * forces_xyz[:, 1] - 4.0 * u0 * forces_xyz[:, 2]) / denom2
    grad_u[:, 1] = (-4.0 * u0 * u1 * forces_xyz[:, 0] + 2.0 * (1.0 + u0**2) * forces_xyz[:, 1] - 4.0 * u1 * forces_xyz[:, 2]) / denom2
    grad_u[:, 2] = 0.0
    
    return energy, grad_u

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    pts = _halton_sphere(N)
    
    def calc_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        log_d2 = np.log(dist_sq)
        np.fill_diagonal(log_d2, 0.0)
        return -0.25 * np.sum(log_d2)

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Phase 1: Gradient descent in stereographic coords
    u = _sphere_to_stereo(pts)
    step_size = 0.03
    momentum = 0.85
    v = np.zeros_like(u)
    
    for it in range(15000):
        if time.time() > deadline - 45.0:
            break
        
        _, grad_u = _energy_grad_stereo(u)
        v = momentum * v + (1 - momentum) * grad_u
        new_u = u - step_size * v
        new_pts = _stereo_to_sphere(new_u)
        new_energy = calc_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            u = new_u
            step_size = min(step_size * 1.003, 0.12)
        else:
            u = _sphere_to_stereo(best_pts)
            step_size *= 0.85
            v *= 0.5
            if step_size < 1e-9:
                break

    # Phase 2: L-BFGS-B in stereographic coords
    if time.time() < deadline - 20.0:
        u0 = _sphere_to_stereo(best_pts)
        
        def obj(u_flat):
            u = u_flat.reshape(N, 3)
            return calc_energy(_stereo_to_sphere(u))
        
        def grad(u_flat):
            u = u_flat.reshape(N, 3)
            _, g = _energy_grad_stereo(u)
            return g.flatten()
        
        try:
            res = minimize(obj, u0.flatten(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11})
            if res.fun < best_energy:
                best_energy = res.fun
                best_pts = _stereo_to_sphere(res.x.reshape(N, 3))
        except Exception:
            pass

    # Phase 3: Final polish
    if time.time() < deadline - 5.0:
        u0 = _sphere_to_stereo(best_pts)
        
        def obj2(u_flat):
            u = u_flat.reshape(N, 3)
            return calc_energy(_stereo_to_sphere(u))
        
        def grad2(u_flat):
            u = u_flat.reshape(N, 3)
            _, g = _energy_grad_stereo(u)
            return g.flatten()
        
        try:
            res2 = minimize(obj2, u0.flatten(), jac=grad2, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13})
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_pts = _stereo_to_sphere(res2.x.reshape(N, 3))
        except Exception:
            pass

    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    return best_pts
