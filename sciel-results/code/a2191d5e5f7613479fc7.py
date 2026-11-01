"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin in seconds


def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = len(points)
    if n < 2:
        return float('inf')
    # Compute all pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    # Zero out diagonal
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return float('inf')
    ratio = max_sq / min_sq
    return ratio ** 2


def _tetra_vertices():
    """Return the 4 vertices of a regular tetrahedron centered at origin with edge length 2."""
    # Regular tetrahedron vertices (centered, edge length 2)
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    # Edge length is 2*sqrt(2), normalize to edge length 2
    edge = 2 * np.sqrt(2)
    v = v * (2.0 / edge)
    return v


def _rotation_matrix_from_params(p):
    """Create rotation matrix from 3 parameters (Euler angles or axis-angle).
    Using axis-angle: p[0:3] is the axis direction, normalized.
    Actually let's use a simpler approach: 3 parameters as rotation angles about x, y, z.
    """
    # Use ZYX Euler angles for simplicity
    ax, ay, az = p[0], p[1], p[2]
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _build_points(params):
    """Build 14 points from parameters.
    params: 9-dimensional vector
      [0:3] - scale/shape parameter (single value repeated? or 3 scales?)
      Let's use: [0] = scale factor for first tetra
      [1] = scale factor for second tetra  
      [2:5] = translation of second tetra relative to first (3 params)
      [5:8] - rotation angles for second tetra (3 params)
      [8] - additional scale or offset
    Actually, let me rethink. Two tetrahedra:
      Tetra 1: 4 points, fixed orientation, centered at origin, scale s1
      Tetra 2: 4 points, rotated by R, translated by t, scaled by s2
    That's 4 + 4 = 8 points. Need 14.
    
    Let me use a different approach: stellated octahedron has 8 vertices total.
    For 14 points, maybe use a different symmetric structure.
    
    Alternative: Use a combination of a regular octahedron (6 vertices) and 
    an inner cube or another shape.
    
    Actually, let me try: 14 points as two interlocking structures.
    Perhaps: a regular icosahedron has 12 vertices. Add 2 more?
    
    Let me try a different strategy: use a known good configuration and optimize.
    For 14 points in 3D, one good approach is to use a grid-like structure or 
    a perturbed lattice.
    
    Let me instead do: 14 points parameterized as a perturbed regular structure.
    Use a 2x2x2 cube (8 points) plus 6 additional points arranged symmetrically.
    
    Actually, let me just use differential evolution on all 42 coordinates with 
    a good initial guess, plus a local refinement step.
    """
    # This function will be replaced by the direct optimization approach below
    raise NotImplementedError


def construct():
    """Construct 14 points in 3D to minimize (max/min pairwise distance)^2."""
    global _START, _DEADLINE, _MARGIN
    
    deadline = _START + _DEADLINE - _MARGIN
    
    # Initial guess: use a grid-based configuration
    # 14 points: take a 3x3x3 grid and pick 14 points
    best_points = None
    best_score = float('inf')
    
    # Strategy 1: Differential evolution on 42 coordinates
    # Use a good initial population based on symmetric structures
    
    def _make_initial_points():
        """Create initial points from a symmetric structure."""
        # Start with a 2x2x2 cube plus 6 more points
        # Or use a perturbed icosahedron-like structure
        
        # Option: 14 points on a sphere with some structure
        # Let's use a combination of:
        # - 6 points of a regular octahedron
        # - 8 points of a cube (slightly smaller or larger)
        
        # Regular octahedron (edge length 2):
        oct_pts = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        
        # Cube vertices (edge length 2/sqrt(3) to have similar circumradius)
        # Actually, let's use a cube with the same circumradius
        # Octahedron circumradius = 1
        # Cube with circumradius 1: half-edge = 1/sqrt(3)
        h = 1.0 / np.sqrt(3)
        cube_pts = np.array([
            [h, h, h], [h, h, -h], [h, -h, h], [h, -h, -h],
            [-h, h, h], [-h, h, -h], [-h, -h, h], [-h, -h, -h]
        ], dtype=float)
        
        pts = np.vstack([oct_pts, cube_pts])
        return pts
    
    # Objective function for optimization
    def _objective(x):
        pts = x.reshape(-1, 3)
        return _score(pts)
    
    # Differential evolution
    initial_pts = _make_initial_points()
    x0 = initial_pts.flatten()
    
    # Define bounds: allow points to move within [-5, 5]
    bounds = [(-5, 5)] * (N * D)
    
    try:
        # Use differential evolution with limited evaluations
        max_iter = 200
        popsize = 10
        result = differential_evolution(
            _objective, bounds, 
            x0=x0,
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            init='latinhypercube',
            updating='deferred',
            polish=False,  # We'll do our own polish
            seed=42
        )
        
        if time.time() > deadline:
            if best_points is not None:
                return best_points
            # Return current best from DE
            best_points = result.x.reshape(-1, 3)
            best_score = _score(best_points)
            return best_points
        
        if result.success or result.fun < best_score:
            best_points = result.x.reshape(-1, 3)
            best_score = _score(best_points)
        
        # Local refinement with L-BFGS-B
        if time.time() < deadline:
            # Use a smooth surrogate: minimize max/min ratio
            # Use the auxiliary variable approach:
            # Minimize t subject to: d_ij^2 <= t * d_min^2 for all i,j
            # This is complex. Instead, use direct minimization with BFGS.
            
            def _smooth_objective(x):
                """Smooth approximation using log-sum-exp."""
                pts = x.reshape(-1, 3)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)
                np.fill_diagonal(sq_dists, 1e-10)
                
                # Use soft-max and soft-min via log-sum-exp
                beta = 50  # temperature parameter
                log_max = np.log(np.sum(np.exp(beta * sq_dists), axis=(0, 1)) / (N * (N - 1))) / beta
                log_min = -np.log(np.sum(np.exp(-beta * sq_dists), axis=(0, 1)) / (N * (N - 1))) / beta
                
                # Ratio approximation
                ratio = np.exp(log_max - log_min)
                return ratio ** 2
            
            # Try local optimization
            try:
                res_local = minimize(
                    _smooth_objective, 
                    best_points.flatten(), 
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_local.fun < best_score:
                    best_points = res_local.x.reshape(-1, 3)
                    best_score = _score(best_points)
            except Exception:
                pass
            
            # Another local refinement with direct objective
            if time.time() < deadline:
                try:
                    res_local2 = minimize(
                        _objective, 
                        best_points.flatten(), 
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res_local2.fun < best_score:
                        best_points = res_local2.x.reshape(-1, 3)
                        best_score = _score(best_points)
                except Exception:
                    pass
    except Exception:
        pass
    
    # If DE didn't find anything good, try random restarts
    if best_points is None or best_score > 100:
        for trial in range(20):
            if time.time() > deadline:
                break
            # Random points on a sphere with radius 1
            pts = np.random.randn(N, D)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Quick local optimization
            try:
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_points = res.x.reshape(-1, 3)
            except Exception:
                pass
    
    # Final check: ensure points are distinct
    if best_points is not None:
        # Verify distinctness
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, 1)
        if np.min(sq_dists) < 1e-12:
            # Points too close, add small perturbation
            best_points += np.random.randn(N, D) * 1e-6
    
    if best_points is None:
        # Fallback: return the initial symmetric structure
        best_points = _make_initial_points()
    
    return best_points


# EVOLVE-BLOCK-END

