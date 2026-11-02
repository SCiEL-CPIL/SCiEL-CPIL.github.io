"""Improved packing of 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices_local(angle_deg):
    """Return the 6 vertices of a unit regular hexagon centered at origin, rotated by angle_deg."""
    a = np.radians(angle_deg)
    base_angles = np.array([0.0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    angles = base_angles + a
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _outer_side_length(vertices, center, angle_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at center, rotated by angle_deg)
    that contains all given vertices.
    """
    a = np.radians(angle_deg)
    base_normals = np.array([np.pi/6, np.pi/6 + np.pi/3, np.pi/6 + 2*np.pi/3,
                             np.pi/6 + np.pi, np.pi/6 + 4*np.pi/3, np.pi/6 + 5*np.pi/3])
    normals = np.column_stack([np.cos(base_normals + a), np.sin(base_normals + a)])
    rel = vertices - np.array(center)
    projections = rel @ normals.T
    max_proj = np.max(projections)
    if max_proj < 0:
        max_proj = 0.0
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _all_inner_vertices(inner):
    verts = []
    for x, y, ang in inner:
        v = _hex_vertices_local(ang) + np.array([x, y])
        verts.append(v)
    return np.vstack(verts)

def _objective(params, N_inner):
    inner = params.reshape(N_inner, 3)
    verts = _all_inner_vertices(inner)
    S = _outer_side_length(verts, (0.0, 0.0), 0.0)
    return S

def _separation_penalty(params, N_inner):
    """Penalty for overlapping inner hexagons using separating axis theorem."""
    penalty = 0.0
    inner = params.reshape(N_inner, 3)
    for i in range(N_inner):
        vi = _hex_vertices_local(inner[i, 2]) + np.array([inner[i, 0], inner[i, 1]])
        ni = np.column_stack([np.cos(np.radians(30 + 60*k) + np.radians(inner[i,2])), 
                               np.sin(np.radians(30 + 60*k) + np.radians(inner[i,2]))] for k in range(6))
        for j in range(i + 1, N_inner):
            vj = _hex_vertices_local(inner[j, 2]) + np.array([inner[j, 0], inner[j, 1]])
            nj = np.column_stack([np.cos(np.radians(30 + 60*k) + np.radians(inner[j,2])), 
                                  np.sin(np.radians(30 + 60*k) + np.radians(inner[j,2]))] for k in range(6))
            all_normals = np.vstack([ni, nj])
            min_sep = np.inf
            for n in all_normals:
                proj_i = vi @ n
                proj_j = vj @ n
                sep = np.max(proj_i) - np.min(proj_j)
                if sep < min_sep:
                    min_sep = sep
            if min_sep < 0:
                penalty += (-min_sep) ** 2
    return penalty

def _objective_with_penalty(params, N_inner, penalty_weight=100.0):
    S = _objective(params, N_inner)
    pen = _separation_penalty(params, N_inner)
    return S + penalty_weight * pen

def construct():
    deadline = time.time() + 140.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Hexagonal cluster initial guess
    positions = [[0.0, 0.0]]
    angles = [0.0]
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([sqrt3 * np.cos(a), sqrt3 * np.sin(a)])
        angles.append(0.0)
    for k in [1, 3, 5, 7]:
        a = np.radians(30 * k)
        positions.append([2 * sqrt3 * np.cos(a), 2 * sqrt3 * np.sin(a)])
        angles.append(0.0)
    
    initial_params = np.array([[p[0], p[1], a] for p, a in zip(positions, angles)]).flatten()
    
    best_params = initial_params.copy()
    best_score = _objective_with_penalty(initial_params, N)
    
    try:
        result = minimize(
            _objective_with_penalty,
            initial_params,
            args=(N,),
            method='Powell',
            options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False}
        )
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except:
        pass
    
    restarts = 0
    while time.time() < deadline - 30 and restarts < 8:
        np.random.seed(100 + restarts)
        perturbed = best_params + np.random.normal(0, 0.2, size=best_params.shape)
        try:
            res = minimize(
                _objective_with_penalty,
                perturbed,
                args=(N,),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except:
            pass
        restarts += 1
    
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _objective_with_penalty,
                best_params,
                args=(N,),
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12, 'disp': False}
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except:
            pass
    
    inner = best_params.reshape(N, 3)
    inner = [[float(x), float(y), float(a)] for x, y, a in inner]
    
    return inner, (0.0, 0.0), 40.0, 0.0
