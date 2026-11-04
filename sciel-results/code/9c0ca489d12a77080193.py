"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Strategy: Use the known optimal topological structure (1 center + 6 first-ring + 4 second-ring
at alternating positions), parameterize the 4 second-ring radial distances, and optimize
together with the global rotation angle via differential evolution + fine local refinement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = math.sqrt(3.0)
S32 = S3 / 2.0

# EVOLVE-BLOCK-START

def _hex_vertices(center, angle_rad):
    """Vertices of unit regular hexagon centered at center, rotated by angle_rad."""
    cx, cy = center
    out = np.empty((6, 2))
    for k in range(6):
        th = angle_rad + k * math.pi / 3.0
        out[k, 0] = cx + math.cos(th)
        out[k, 1] = cy + math.sin(th)
    return out

def _outer_hex_verts(S, oa):
    """Vertices of outer hexagon with side S, rotated by oa, centered at origin."""
    out = np.empty((6, 2))
    for k in range(6):
        th = oa + k * math.pi / 3.0
        out[k, 0] = S * math.cos(th)
        out[k, 1] = S * math.sin(th)
    return out

def _poly_poly_dist(A, B):
    """Minimum distance between two convex polygons (both CCW). 0 if overlapping."""
    # SAT-based
    min_gap = float('inf')
    for P in (A, B):
        n = len(P)
        for i in range(n):
            e = P[(i + 1) % n] - P[i]
            el = np.linalg.norm(e)
            if el < 1e-14:
                continue
            nrm = np.array([-e[1], e[0]]) / el
            pa = A @ nrm
            pb = B @ nrm
            a_min, a_max = pa.min(), pa.max()
            b_min, b_max = pb.min(), pb.max()
            if a_max < b_min:
                gap = b_min - a_max
            elif b_max < a_min:
                gap = a_min - b_max
            else:
                continue
            if gap < min_gap:
                min_gap = gap
                if min_gap <= 0.0:
                    return 0.0
    return min_gap

def _build_config(r2s, oa):
    """Build centers and angles for the 11 hexagons.
    r2s: 4 radial distances for the second ring (positions at 30, 150, 210, 330 deg)
    oa: outer hexagon rotation angle (radians)
    All inner hexagons are axis-aligned (angle=0).
    """
    d = S3  # first ring distance
    centers = np.empty((N, 2))
    angles = np.zeros(N)
    centers[0] = [0.0, 0.0]
    for i in range(6):
        th = i * math.pi / 3.0
        centers[1 + i] = [d * math.cos(th), d * math.sin(th)]
    # Second ring at 30, 150, 210, 330 degrees (i.e., (2k+1)*30 deg for k=0..3)
    r2_thetas = [math.pi / 6.0, 5.0 * math.pi / 6.0, 7.0 * math.pi / 6.0, 11.0 * math.pi / 6.0]
    for k in range(4):
        centers[7 + k] = [r2s[k] * math.cos(r2_thetas[k]), r2s[k] * math.sin(r2_thetas[k])]
    return centers, angles

def _objective(params):
    """Objective: outer side length S plus overlap penalty.
    params: [r2_0, r2_1, r2_2, r2_3, oa]
    """
    r2s = params[:4]
    oa = params[4]
    # Clamp r2s to reasonable range
    r2s = np.clip(r2s, 1.0, 4.0)
    centers, angles = _build_config(r2s, oa)
    
    # Compute required S for containment
    outer_normals = np.array([[math.cos(oa + k * math.pi / 3.0),
                                math.sin(oa + k * math.pi / 3.0)] for k in range(6)])
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i], angles[i])
        projs = np.abs(verts @ outer_normals.T)
        m = projs.max()
        if m > max_proj:
            max_proj = m
    S = 2.0 * max_proj / S3
    
    # Overlap penalty
    polys = [_hex_vertices(centers[i], angles[i]) for i in range(N)]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_poly_dist(polys[i], polys[j])
            if d < 0.005:
                pen += (0.005 - d) ** 2
    return S + 50.0 * pen

def _validate(centers, angles, S, oa, tol=1e-5):
    """Check no overlaps and all inside outer hexagon of side S."""
    polys = [_hex_vertices(centers[i], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_poly_dist(polys[i], polys[j]) < tol:
                return False
    outer_normals = np.array([[math.cos(oa + k * math.pi / 3.0),
                                math.sin(oa + k * math.pi / 3.0)] for k in range(6)])
    limit = S * S32
    for i in range(N):
        verts = _hex_vertices(centers[i], angles[i])
        projs = np.abs(verts @ outer_normals.T)
        if projs.max() > limit + tol:
            return False
    return True

def _tighten_S(centers, angles, oa):
    """Compute exact minimum S for containment."""
    outer_normals = np.array([[math.cos(oa + k * math.pi / 3.0),
                                math.sin(oa + k * math.pi / 3.0)] for k in range(6)])
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i], angles[i])
        projs = np.abs(verts @ outer_normals.T)
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / S3

def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Initial guess: r2 = 2*sqrt(3) (regular hex lattice), oa = 0
    r2_init = np.array([2.0 * S3, 2.0 * S3, 2.0 * S3, 2.0 * S3])
    oa_init = 0.0
    centers, angles = _build_config(r2_init, oa_init)
    best_S = _tighten_S(centers, angles, oa_init)
    best_params = np.concatenate([r2_init, [oa_init]])
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_oa = oa_init
    
    if not _validate(best_centers, best_angles, best_S, best_oa):
        # Fallback: spread out more
        r2_init = np.array([2.5 * S3, 2.5 * S3, 2.5 * S3, 2.5 * S3])
        centers, angles = _build_config(r2_init, 0.0)
        best_S = _tighten_S(centers, angles, 0.0)
        best_params = np.concatenate([r2_init, [0.0]])
        best_centers = centers.copy()
        best_angles = angles.copy()
        best_oa = 0.0
    
    # Phase 1: Differential evolution over 5 params
    if time.time() < deadline - 60.0:
        bounds = [(1.5, 4.0)] * 4 + [(-math.pi / 12.0, math.pi / 12.0)]
        try:
            de_res = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=80, popsize=15, tol=1e-8,
                polish=True, mutation=(0.3, 1.0), recombination=0.7,
                init='latinhypercube'
            )
            if de_res.fun < best_S + 1e-6:
                cand_params = de_res.x
                r2s = np.clip(cand_params[:4], 1.0, 4.5)
                oa_c = cand_params[4]
                c, a = _build_config(r2s, oa_c)
                S_c = _tighten_S(c, a, oa_c)
                if _validate(c, a, S_c, oa_c):
                    best_S = S_c
                    best_params = np.concatenate([r2s, [oa_c]])
                    best_centers = c
                    best_angles = a
                    best_oa = oa_c
        except Exception:
            pass
    
    # Phase 2: Fine local refinement with Nelder-Mead from best
    if time.time() < deadline - 30.0:
        x0 = best_params.copy()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-10,
                                    'maxiter': 20000, 'maxfev': 30000,
                                    'adaptive': True})
            if res.fun < best_S + 1e-8:
                cand_params = res.x
                r2s = np.clip(cand_params[:4], 1.0, 4.5)
                oa_c = cand_params[4]
                c, a = _build_config(r2s, oa_c)
                S_c = _tighten_S(c, a, oa_c)
                if _validate(c, a, S_c, oa_c):
                    best_S = S_c
                    best_params = np.concatenate([r2s, [oa_c]])
                    best_centers = c
                    best_angles = a
                    best_oa = oa_c
        except Exception:
            pass
    
    # Phase 3: Try small perturbations of r2s independently (coordinate descent)
    if time.time() < deadline - 15.0:
        for _ in range(200):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for k in range(4):
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    trial = best_params.copy()
                    trial[k] = np.clip(trial[k] + delta, 1.0, 4.5)
                    r2s = trial[:4]
                    oa_c = trial[4]
                    c, a = _build_config(r2s, oa_c)
                    S_c = _tighten_S(c, a, oa_c)
                    if S_c < best_S - 1e-9 and _validate(c, a, S_c, oa_c):
                        best_S = S_c
                        best_params = trial.copy()
                        best_centers = c
                        best_angles = a
                        best_oa = oa_c
                        improved = True
                        break
                if improved:
                    break
            # Also try small oa perturbations
            for delta in [0.001, -0.001, 0.0005, -0.0005]:
                trial = best_params.copy()
                trial[4] = trial[4] + delta
                r2s = trial[:4]
                oa_c = trial[4]
                c, a = _build_config(r2s, oa_c)
                S_c = _tighten_S(c, a, oa_c)
                if S_c < best_S - 1e-9 and _validate(c, a, S_c, oa_c):
                    best_S = S_c
                    best_params = trial.copy()
                    best_centers = c
                    best_angles = a
                    best_oa = oa_c
                    break
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 8.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-11, 'fatol': 1e-12,
                                    'maxiter': 30000, 'maxfev': 50000,
                                    'adaptive': True})
            if res.fun < best_S + 1e-10:
                cand_params = res.x
                r2s = np.clip(cand_params[:4], 1.0, 4.5)
                oa_c = cand_params[4]
                c, a = _build_config(r2s, oa_c)
                S_c = _tighten_S(c, a, oa_c)
                if _validate(c, a, S_c, oa_c):
                    best_S = S_c
                    best_params = np.concatenate([r2s, [oa_c]])
                    best_centers = c
                    best_angles = a
                    best_oa = oa_c
        except Exception:
            pass
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]),
                      float(math.degrees(best_angles[i]))])
    
    return inner, (0.0, 0.0), float(best_S), float(math.degrees(best_oa))

# EVOLVE-BLOCK-END
