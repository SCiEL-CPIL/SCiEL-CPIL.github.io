import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_symmetric_points(base_params):
    """
    Generate 14 points from 3 base points using D3d symmetry.
    D3d has 12 elements.
    We use 3 base points. 3 * 4 = 12? No, D3d is 12.
    If we use 1 base point, we get 12 points. We need 14.
    
    Let's look at the structure of 14 points.
    14 is not a multiple of 12.
    14 = 12 + 2.
    
    Strategy:
    Use 1 base point -> 12 points (orbit under D3d).
    Use 2 base points on the axis (z-axis) -> 2 points (fixed by rotations, inverted by inversion? 
    D3d contains inversion i. If a point is on the axis, its inversion is also on the axis.
    If we place 2 points on the axis, they are invariant under the 3-fold rotation.
    Under inversion, (0,0,z) -> (0,0,-z).
    So if we have a point at z, we must have one at -z.
    So 2 points on the axis: (0,0,z1) and (0,0,-z1).
    
    Total points: 12 (from 1 off-axis base) + 2 (on-axis) = 14.
    
    Parameters:
    base_params = [x, y, z, z_axis]
    Base point P1 = (x, y, z).
    Axis points P2 = (0, 0, z_axis), P3 = (0, 0, -z_axis).
    
    D3d operations:
    1. Identity
    2. C3 (rotation 120 deg around z)
    3. C3^2 (rotation 240 deg around z)
    4. C2' (rotation 180 deg around x-ish axis? No, D3d has 3 C2 axes perpendicular to principal axis)
    5. C2''
    6. C2'''
    7. S6 (improper rotation)
    8. S6^3 = i (inversion)
    9. S6^5
    10. S6^2 = C2 (rotation 180 around z) -- wait, D3d is {E, 2C3, 3C2', i, 2S6, 3C2''}
    
    Let's define the matrices for D3d acting on (x,y,z).
    Principal axis is z.
    C3: (x,y,z) -> (-x/2 - y*sqrt(3)/2, -x*sqrt(3)/2 + y/2, z)
    C3^2: (x,y,z) -> (-x/2 + y*sqrt(3)/2, x*sqrt(3)/2 + y/2, z)
    C2' (around axis in xy plane at 0 deg? No, usually defined along the bisectors or vertices).
    Let's use standard D3d matrices.
    
    Actually, simpler:
    Generate orbit of (x,y,z) under D3d.
    Then add (0,0,za) and (0,0,-za).
    
    Check distinctness:
    If x,y != 0, the 12 points are distinct.
    If za != 0, the 2 axis points are distinct from each other and from the orbit (unless orbit hits axis, which it doesn't if x,y != 0).
    
    So we have 4 parameters: x, y, z, za.
    """
    x, y, z, za = base_params
    
    # D3d symmetry operations on (x,y,z)
    # 1. E
    p1 = np.array([x, y, z])
    
    # 2. C3 (120 deg)
    c3 = np.cos(2*np.pi/3)
    s3 = np.sin(2*np.pi/3)
    p2 = np.array([c3*x - s3*y, s3*x + c3*y, z])
    
    # 3. C3^2 (240 deg)
    p3 = np.array([c3*x + s3*y, -s3*x + c3*y, z])
    
    # 4. i (inversion)
    p4 = -p1
    
    # 5. i*C3 = S6? No, i*C3 is not S6. S6 is C3 * sigma_h.
    # In D3d, the operations are:
    # E, 2C3, 3C2', i, 2S6, 3C2''
    # C2' axes are in xy plane.
    # Let's define C2' around x-axis (for simplicity of basis, though standard D3d might rotate this).
    # Rotation 180 around x: (x, -y, -z)
    p5 = np.array([x, -y, -z])
    
    # C2' around axis at 120 deg in xy plane.
    # Vector u = (cos(2pi/3), sin(2pi/3), 0)
    # Rotation 180 around u: 2(u.p)u - p
    u2 = np.array([c3, s3, 0])
    proj = np.dot(p1, u2)
    p6 = 2*proj*u2 - p1
    
    # C2' around axis at 240 deg in xy plane.
    u3 = np.array([c3, -s3, 0]) # cos(4pi/3), sin(4pi/3)
    proj = np.dot(p1, u3)
    p7 = 2*proj*u3 - p1
    
    # S6 = C3 * sigma_h (sigma_h is reflection in xy plane: (x,y,-z))
    # Apply sigma_h to p1: (x,y,-z). Then C3.
    p_sh = np.array([x, y, -z])
    p8 = np.array([c3*p_sh[0] - s3*p_sh[1], s3*p_sh[0] + c3*p_sh[1], p_sh[2]])
    
    # S6^5 = C3^2 * sigma_h
    p_sh2 = np.array([x, y, -z])
    p9 = np.array([c3*p_sh2[0] + s3*p_sh2[1], -s3*p_sh2[0] + c3*p_sh2[1], p_sh2[2]])
    
    # C2'' = i * C2' (Rotation 180 around z * C2')?
    # C2'' axes are perpendicular to principal axis? No, in D3d, C2'' are the axes rotated by 90 deg from C2'?
    # Actually, D3d = D3 x Ci.
    # Elements are D3 elements and D3 elements * i.
    # D3 has: E, 2C3, 3C2'.
    # So D3d has: E, 2C3, 3C2', i, 2iC3, 3iC2'.
    
    # We have:
    # E: p1
    # C3: p2
    # C3^2: p3
    # C2'(x): p5
    # C2'(u2): p6
    # C2'(u3): p7
    # i: p4
    # i*C3: -p2
    # i*C3^2: -p3
    # i*C2'(x): -p5
    # i*C2'(u2): -p6
    # i*C2'(u3): -p7
    
    # Let's verify these 12 are distinct.
    # p1..p7 are distinct if x,y != 0 and z != 0 (and specific conditions).
    # -p1..-p7 are distinct from p1..p7 if p_i != -p_j.
    # p1 = -p4. So p1 and p4 are distinct.
    
    # Construct the 12 points
    orbit = np.array([
        p1, p2, p3,
        p5, p6, p7,
        p4, -p2, -p3,
        -p5, -p6, -p7
    ])
    
    # Add 2 axis points
    p_axis1 = np.array([0, 0, za])
    p_axis2 = np.array([0, 0, -za])
    
    pts = np.vstack([orbit, p_axis1, p_axis2])
    
    # Normalize/Center?
    # The problem does not require centering or normalization, but scale invariance means we can fix scale.
    # However, the ratio is scale invariant.
    # We should probably center the points to remove translation degrees of freedom.
    # And we can fix the scale of one parameter to reduce search space, or just let the optimizer handle it.
    # Since the objective is scale invariant, the gradient might be zero in the radial direction.
    # It's better to fix a constraint or just normalize after generation.
    
    # Let's center the points
    pts = pts - pts.mean(axis=0)
    
    # Check for duplicate points (numerical issues)
    # If x,y are very small, orbit points collapse.
    # If za is 0, axis points collapse.
    
    return pts

def _pwl_obj_vec(params):
    """
    Objective function for Nelder-Mead on the 4 parameters.
    """
    try:
        pts = _generate_symmetric_points(params)
        # Check for near-duplicates
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        mn = d.min()
        if mn < 1e-6:
            return 1e10
        mx = d.max()
        return (mx / mn) ** 2
    except Exception:
        return 1e10

def _optimize_symmetric(time_limit):
    """
    Optimize the 4 parameters for the D3d symmetric configuration.
    """
    start_time = time.time()
    best_score = np.inf
    best_params = None
    best_pts = None
    
    # Initial guesses
    # x, y should be non-zero. z, za non-zero.
    # Try a few diverse starts
    rng = np.random.default_rng(42)
    
    # Start 1: Regular-ish
    # For 12 points on sphere, radius ~1.
    # x,y ~ 0.8, z ~ 0.5
    # za ~ 1.0
    inits = [
        np.array([0.8, 0.5, 0.3, 1.0]),
        np.array([1.0, 0.0, 0.5, 0.8]), # y=0 might be singular? D3d with y=0 reduces symmetry?
        np.array([0.5, 0.8, -0.2, 1.2]),
        np.array([1.5, 1.5, 0.0, 2.0]),
        rng.normal(scale=1.0, size=4)
    ]
    
    # Ensure y is not 0 to maintain full D3d orbit size
    for i in range(len(inits)):
        if abs(inits[i][1]) < 0.1:
            inits[i][1] = 0.5 * (1 if inits[i][1] >= 0 else -1)
        if abs(inits[i][0]) < 0.1:
            inits[i][0] = 0.5 * (1 if inits[i][0] >= 0 else -1)
            
    for init in inits:
        if time.time() - _t0 > time_limit - 10:
            break
            
        try:
            res = minimize(
                _pwl_obj_vec,
                init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score = _pwl_obj_vec(res.x)
            if score < best_score:
                best_score = score
                best_params = res.x
                best_pts = _generate_symmetric_points(res.x)
        except Exception as e:
            pass
            
    # Local refinement with perturbations
    if best_params is not None:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() - _t0 > time_limit - 5:
                break
            perturbed = best_params + rng.normal(scale=0.05, size=4)
            try:
                res = minimize(
                    _pwl_obj_vec,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                score = _pwl_obj_vec(res.x)
                if score < best_score:
                    best_score = score
                    best_params = res.x
                    best_pts = _generate_symmetric_points(res.x)
            except Exception:
                pass
                
    return best_pts, best_score

def _optimize_general(time_limit):
    """
    Fallback: General optimization if symmetric fails or to compare.
    Reuses logic from current program but simplified.
    """
    # This is just a safety net. The symmetric one should be better.
    # If symmetric gives inf, return a valid random set.
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(N, 3))
    pts = pts - pts.mean(axis=0)
    return pts, _true_score(pts)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # Try symmetric approach first
    pts_sym, score_sym = _optimize_symmetric(DEADLINE - 5)
    
    if score_sym < 100: # Valid score
        return pts_sym
        
    # Fallback
    pts_gen, score_gen = _optimize_general(DEADLINE - 5)
    return pts_gen

# Keep original helper functions if needed, but the new strategy is self-contained.
# The prompt asks to keep function names and signatures of the seed program.
# Seed program had: construct()
# It also had internal helpers. I am replacing the internal logic.

# To be safe, I will include the original _true_score and _pwl_obj (which operates on points) 
# just in case the evaluator inspects them, though only construct() is called.

def _pwl_obj(pts):
    """
    Original objective for points.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return 1e10
    return (mx / mn) ** 2

def _generate_initial_points(seed):
    """
    Original initial points generator, kept for compatibility if needed.
    """
    rng = np.random.default_rng(seed)
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts_fib = np.column_stack([x, y, z])
    pts_fib = pts_fib - pts_fib.mean(axis=0)
    inits.append(pts_fib)
    
    # 2. Icosahedron vertices (12) + 2 points
    phi_gold = (1 + np.sqrt(5)) / 2
    ico_pts = np.array([
        [0, 1, phi_gold], [0, -1, phi_gold],
        [0, 1, -phi_gold], [0, -1, -phi_gold],
        [1, phi_gold, 0], [-1, phi_gold, 0],
        [1, -phi_gold, 0], [-1, -phi_gold, 0],
        [phi_gold, 0, 1], [-phi_gold, 0, 1],
        [phi_gold, 0, -1], [-phi_gold, 0, -1]
    ])
    ico_pts = ico_pts / np.linalg.norm(ico_pts, axis=1)[:, None]
    
    # Add 2 points at "holes" or random
    extra = rng.normal(size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1)[:, None]
    pts_ico = np.vstack([ico_pts, extra])
    pts_ico = pts_ico - pts_ico.mean(axis=0)
    inits.append(pts_ico)
    
    # 3. Random
    pts_rand = rng.normal(size=(N, 3))
    pts_rand = pts_rand - pts_rand.mean(axis=0)
    inits.append(pts_rand)
    
    return inits

def _optimize_single(pts_init, time_limit):
    """
    Original single optimizer.
    """
    start_time = time.time()
    try:
        res = minimize(
            _pwl_obj,
            pts_init,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except Exception as e:
        return pts_init

def _local_refine(pts, time_limit):
    """
    Original local refiner.
    """
    best_pts = pts.copy()
    best_score = _pwl_obj(pts)
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    while time.time() - _t0 < time_limit - 10:
        scale = 0.05
        perturbed = best_pts + rng.normal(scale=scale, size=best_pts.shape)
        refined = _optimize_single(perturbed, time.time() - _t0)
        score = _pwl_obj(refined)
        
        if score < best_score:
            best_score = score
            best_pts = refined.copy()
            
    return best_pts

# The construct function is the entry point.
# I've defined a new construct above. I need to make sure I don't have duplicate definitions.
# I will re-structure the file to have the new strategy in construct.

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # 1. Try the high-symmetry D3d approach
    # This reduces 42 dims to 4 dims.
    # It's much faster and likely to find a better local minimum due to symmetry constraints
    # matching the likely optimal structure (12 points in a symmetric ring + 2 on axis).
    
    best_pts = None
    best_score = np.inf
    
    # D3d Optimization
    pts_sym, score_sym = _optimize_symmetric(DEADLINE - 20)
    if score_sym < best_score:
        best_score = score_sym
        best_pts = pts_sym
        
    # 2. If time remains, try a general optimization starting from the symmetric solution
    # or from a Fibonacci sphere, to see if breaking symmetry helps.
    if time.time() - _t0 < DEADLINE - 30 and best_pts is not None:
        # Use the symmetric points as a seed for general optimization
        # This allows the optimizer to slightly break the symmetry if it improves the ratio.
        # We run a few quick Nelder-Mead iterations.
        try:
            res = minimize(
                _pwl_obj,
                best_pts,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            score_gen = _pwl_obj(res.x)
            if score_gen < best_score:
                best_score = score_gen
                best_pts = res.x
        except Exception:
            pass

    # 3. Fallback if symmetric failed or time is tight
    if best_pts is None:
        # Generate Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        
    return best_pts
