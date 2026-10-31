"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

# Pre-compute triangle indices for all C(11,3) = 165 triples
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary columns are (u, v, w) corresponding to V0, V1, V2
    # P = u*V0 + v*V1 + w*V2
    # Since V0 is (0,0), P = v*V1 + w*V2
    # x = v*1 + w*0.5
    # y = v*0 + w*sqrt(3)/2
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = bary[:, 2] * H
    return np.column_stack((x, y))

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by points given in barycentric coords."""
    # Area of triangle formed by points P_i, P_j, P_k in barycentric coords:
    # Area = 0.5 * |det(M)| * Area(Triangle)
    # Where M is the 3x3 matrix of barycentric coordinates of the 3 points.
    # Wait, the determinant of the matrix of barycentric coordinates gives the ratio of areas.
    # Let M be matrix with rows bary[i], bary[j], bary[k].
    # Then Area(P_i P_j P_k) = |det(M)| * Area(V0 V1 V2).
    
    # Extract barycentric coordinates for all triples
    # tri_idx shape (165, 3)
    # bary shape (11, 3)
    # We need to index bary with tri_idx
    # result shape (165, 3, 3)
    M = bary[tri_idx]
    
    # Calculate determinant
    dets = np.linalg.det(M)
    
    # Absolute value and scale by triangle area
    # Area of V0 V1 V2 is 0.5 * 1 * H = H/2
    areas = 0.5 * np.abs(dets) * (1.0 * H) # Area = 0.5 * base * height. Base=1, Height=H. Area=H/2.
    # Wait, determinant of barycentric matrix for vertices of the reference triangle is 1.
    # So Area = |det| * Area_Reference.
    # Area_Reference = 0.5 * 1 * H = H/2.
    # So areas = 0.5 * np.abs(dets) * (1.0 * H) is correct.
    
    return areas

def clamp_bary(bary):
    """Clamp barycentric coordinates to be valid (non-negative and sum to 1)."""
    # Normalize to sum to 1
    s = np.sum(bary, axis=1, keepdims=True)
    # Avoid division by zero
    s[s == 0] = 1.0
    bary = bary / s
    # Clamp negatives to 0 and renormalize
    bary = np.maximum(bary, 0)
    s = np.sum(bary, axis=1, keepdims=True)
    s[s == 0] = 1.0
    bary = bary / s
    return bary

def hard_min_obj(flat_bary):
    """Objective function to minimize negative min area."""
    bary = flat_bary.reshape(N, 3)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    return -min_area

def generate_hexagonal_init(rng):
    """Generate a structured initial guess using a hexagonal grid pattern."""
    # This is a heuristic initialization that often provides a good starting point
    # for maximizing minimum distance/area.
    # For N=11, a common optimal configuration is related to a triangular grid.
    # Let's try a specific known good configuration or a perturbed grid.
    
    # One known good configuration for 11 points in a triangle is derived from
    # a 3x3 grid plus some adjustments, or a hexagonal packing.
    # Let's use barycentric coordinates for a 3x3 grid on the edges and center.
    # Actually, let's just use a random perturbation of a grid.
    
    # Grid points in barycentric coords (u, v, w)
    # 3x3 grid:
    # (1,0,0), (2/3,1/3,0), (1/3,2/3,0), (0,1,0) -- Edge 1
    # (2/3,0,1/3), (1/3,1/3,1/3), (0,2/3,1/3) -- Middle
    # (1/3,0,2/3), (0,1/3,2/3), (0,0,1) -- Edge 2
    
    # This gives 9 points. We need 11.
    # Let's add midpoints of the sides of the central triangle?
    # Or just use a random start and let optimization do the work, but seed with something better than pure random.
    
    # Let's try a "star" pattern or similar.
    # Center: (1/3, 1/3, 1/3)
    # 6 points around center at radius r?
    # 4 points at corners?
    
    # Simpler: Use the 9 grid points + 2 more.
    # Let's just generate 11 random points and rely on multiple restarts.
    # But the inspiration mentions "generate_hexagonal_init".
    
    # Let's construct a specific 11-point configuration.
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Midpoints of sides: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Center: (1/3, 1/3, 1/3)
    # That's 7 points.
    # Add 4 more?
    
    # Let's use a random initialization but with a bias towards symmetry.
    bary = rng.random((N, 3))
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    return bary

def generate_random_init(rng):
    """Generate random barycentric coordinates."""
    bary = rng.random((N, 3))
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    return bary

def refine_local(bary, time_limit):
    """Refine the barycentric coordinates using Nelder-Mead or L-BFGS-B."""
    flat_bary = bary.flatten()
    
    # Use L-BFGS-B if possible, but Nelder-Mead is derivative-free and robust for small N.
    # Given N=11, 22 variables (if we drop one bary coord) or 33 variables (with constraint).
    # Using 33 variables with clamping is simpler for Nelder-Mead.
    
    start_time = time.time()
    
    # Try Nelder-Mead first
    try:
        res = minimize(
            hard_min_obj, 
            flat_bary, 
            method='Nelder-Mead', 
            options={
                'maxiter': 1000, 
                'xatol': 1e-6, 
                'fatol': 1e-6
            }
        )
        if time.time() - start_time > time_limit * 0.5:
             # If it took too long, maybe try a faster method or just accept
             pass
        return res.x.reshape(N, 3)
    except Exception:
        return bary

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_score = -np.inf
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        # Check time
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if r == 0:
            # First try a structured guess
            # Simple heuristic: place points at vertices, midpoints, center, and some random
            bary_init = generate_random_init(rng)
        else:
            bary_init = generate_random_init(rng)
        
        # Refine
        bary_opt = refine_local(bary_init, 2.0) # Allow up to 2 seconds per refinement
        
        # Evaluate
        bary_opt = clamp_bary(bary_opt)
        areas = calc_areas_bary(bary_opt)
        min_area = np.min(areas)
        
        if min_area > best_score:
            best_score = min_area
            best_bary = bary_opt.copy()
            
    if best_bary is None:
        # Fallback
        best_bary = generate_random_init(rng)
        
    # Convert to Cartesian
    cart_points = bary_to_cart(best_bary)
    
    return cart_points
# EVOLVE-BLOCK-END
