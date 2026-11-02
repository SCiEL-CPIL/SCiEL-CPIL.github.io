"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # seconds, safety margin before 150s limit

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2
    # V0 is (0,0), so term 0 vanishes
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack((x, y))

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # Solve for b1, b2 given x = b1 + 0.5*b2, y = S3H*b2
    b2 = y / S3H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack((b0, b1, b2))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (Cartesian)."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        area = 0.5 * cross
        if area < min_area:
            min_area = area
    return min_area

def _min_triangle_area_bary(bary):
    """Compute the minimum area of any triangle formed by three points (Barycentric)."""
    # Area in Cartesian = 0.5 * |cross(pj-pi, pk-pi)|
    # In Barycentric, the area of triangle formed by points with barycentric coords 
    # b_i, b_j, b_k is proportional to the absolute value of the determinant of 
    # the matrix formed by their barycentric coordinates, multiplied by the area of the reference triangle.
    # Reference triangle area = S3H / 2.
    # Det = b_i0(b_j1*b_k2 - b_j2*b_k1) - b_i1(b_j0*b_k2 - b_j2*b_k0) + b_i2(b_j0*b_k1 - b_j1*b_k0)
    # Area = 0.5 * |Det| * Area_Ref? No, the formula is Area = |Det| * Area_Ref.
    # Let's verify.
    # Vertices: (1,0,0), (0,1,0), (0,0,1). Det = 1. Area = S3H/2. Correct.
    # Midpoints: (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5).
    # Det = 0.5(0.25-0) - 0.5(0-0.25) + 0 = 0.125 + 0.125 = 0.25.
    # Area = 0.25 * (S3H/2) = S3H/8.
    # Midpoint triangle area is 1/4 of total area. Total = S3H/2. 1/4 * S3H/2 = S3H/8. Correct.
    
    n = len(bary)
    min_area = float('inf')
    area_ref = S3H / 2.0
    for i, j, k in combinations(range(n), 3):
        bi = bary[i]
        bj = bary[j]
        bk = bary[k]
        det = bi[0] * (bj[1] * bk[2] - bj[2] * bk[1]) - \
              bi[1] * (bj[0] * bk[2] - bj[2] * bk[0]) + \
              bi[2] * (bj[0] * bk[1] - bj[1] * bk[0])
        area = abs(det) * area_ref
        if area < min_area:
            min_area = area
    return min_area

def _check_bary_valid(bary, eps=1e-9):
    """Check if barycentric coordinates are valid (sum to 1, all >= 0)."""
    if not np.allclose(np.sum(bary, axis=1), 1.0, atol=1e-5):
        return False
    if np.any(bary < -eps):
        return False
    return True

def _project_bary(bary):
    """Project barycentric coordinates to the valid simplex (sum=1, >=0)."""
    bary = np.array(bary, dtype=float)
    # Normalize sum to 1
    sums = np.sum(bary, axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    bary = bary / sums
    # Clip negative values
    bary = np.clip(bary, 0.0, 1.0)
    # Re-normalize after clipping to ensure sum is exactly 1
    sums = np.sum(bary, axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    bary = bary / sums
    return bary

def _objective_bary(x_flat):
    """Objective function for Nelder-Mead: negative of min triangle area."""
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    # If projection changed things significantly, it might be outside, but Nelder-Mead handles this.
    # We just want to maximize the area, so minimize -area.
    area = _min_triangle_area_bary(bary)
    return -area

def _structured_init_bary(rng):
    """Generate a structured initial guess in barycentric coordinates."""
    # Vertices
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    
    # Edge midpoints
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    
    # Centroid
    centroid = (v0 + v1 + v2) / 3
    
    pts = [v0, v1, v2, m01, m12, m20, centroid]
    
    # Add 4 random points
    for _ in range(4):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        b0 = 1 - u - v
        b1 = u
        b2 = v
        pts.append(np.array([b0, b1, b2]))
        
    return np.array(pts)

def _random_init_bary(rng):
    """Generate a random initial guess in barycentric coordinates."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        b0 = 1 - u - v
        b1 = u
        b2 = v
        pts.append(np.array([b0, b1, b2]))
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_bary = None
    best_area = 0.0
    
    # Phase 1: Multiple restarts with Nelder-Mead
    # Nelder-Mead is derivative-free and works well with non-smooth objectives like min-area.
    # It operates in the 33-dimensional space of barycentric coords (3 per point).
    
    max_restarts = 20
    
    for trial in range(max_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Alternate between structured and random starts
        if trial % 3 == 0:
            bary_init = _structured_init_bary(rng)
        else:
            bary_init = _random_init_bary(rng)
        
        x0 = bary_init.flatten()
        
        try:
            # Use Nelder-Mead
            # maxiter needs to be high enough for 33D
            result = minimize(
                _objective_bary,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
        except Exception:
            continue
        
        opt_bary = result.x.reshape(N, 3)
        opt_bary = _project_bary(opt_bary)
        
        area = _min_triangle_area_bary(opt_bary)
        if area > best_area:
            best_area = area
            best_bary = opt_bary.copy()
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break

    # Phase 2: Polish with a finer Nelder-Mead run on the best result
    if best_bary is not None and time.time() - start_time < DEADLINE - 10:
        x0 = best_bary.flatten()
        try:
            result = minimize(
                _objective_bary,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            opt_bary = result.x.reshape(N, 3)
            opt_bary = _project_bary(opt_bary)
            area = _min_triangle_area_bary(opt_bary)
            if area > best_area:
                best_area = area
                best_bary = opt_bary.copy()
        except Exception:
            pass

    if best_bary is None:
        # Fallback to random
        best_bary = _random_init_bary(rng)
    
    # Convert to Cartesian for output
    cart_pts = _bary_to_cart(best_bary)
    
    # Final sanity check: ensure points are inside
    # The barycentric projection ensures this, but let's double check with Cartesian bounds
    # and project if necessary (though it shouldn't be needed)
    for i in range(N):
        x, y = cart_pts[i]
        if y < 0:
            cart_pts[i, 1] = 0
        max_y = min(S3H * x, S3H * (1 - x))
        if cart_pts[i, 1] > max_y:
            cart_pts[i, 1] = max_y
        if x < 0:
            cart_pts[i, 0] = 0
        if x > 1:
            cart_pts[i, 0] = 1
            
    return cart_pts

if __name__ == "__main__":
    pts = construct()
    print(pts)
