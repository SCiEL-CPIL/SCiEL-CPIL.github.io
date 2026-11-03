import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian coordinates.
    Point P = (1-u-v)*A + u*B + v*C
    Since A is (0,0), P = u*B + v*C
    """
    x = u * B[0] + v * C[0]
    y = u * B[1] + v * C[1]
    return np.array([x, y])

def cart_to_bary(x, y):
    """Convert Cartesian coordinates to barycentric coordinates (u, v).
    x = u*1 + v*0.5
    y = u*0 + v*(sqrt(3)/2)
    => v = 2*y/sqrt(3)
    => u = x - 0.5*v
    """
    v = 2.0 * y / np.sqrt(3)
    u = x - 0.5 * v
    return np.array([u, v])

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    """
    # Vectorized area calculation for all triples is O(N^3), which is 11^3 = 1331 operations.
    # This is very fast.
    min_area = np.inf
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                vec1 = pj - pi
                vec2 = pk - pi
                area = 0.5 * abs(vec1[0]*vec2[1] - vec1[1]*vec2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """
    params: array of shape (2*N,) representing u_i, v_i for each point.
    Returns negative of the minimum area.
    """
    u = params[0:N]
    v = params[N:2*N]
    
    # Ensure barycentric coordinates are valid (u>=0, v>=0, u+v<=1)
    # Powell doesn't handle bounds well by default, but we can clip.
    # Clipping might create non-differentiable boundaries, but Powell is derivative-free.
    # A better way is to use penalty or just let it optimize and check validity at the end.
    # However, if points go outside, the area calculation is still valid geometrically,
    # but the constraint "inside triangle" is violated.
    # The problem requires points "on or inside".
    # We will clip u, v to valid range [0, 1] and u+v <= 1.
    # If u+v > 1, we scale down u and v proportionally to bring them to the boundary.
    
    u_clipped = np.clip(u, 0, 1)
    v_clipped = np.clip(v, 0, 1)
    
    # Check u+v <= 1
    s = u_clipped + v_clipped
    mask = s > 1.0
    if np.any(mask):
        u_clipped[mask] = u_clipped[mask] / s[mask]
        v_clipped[mask] = v_clipped[mask] / s[mask]
        
    points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
    
    min_area = compute_min_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_params = None
    best_score = -np.inf
    
    # Initial guesses
    # 1. Random points
    # 2. Grid-like points
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Random initialization
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        initializations.append(np.concatenate([u, v]))
        
    # Structured initialization: approximate grid
    # We want to spread points out.
    # Try a few regular patterns.
    # Pattern 1: Center + corners + midpoints
    # But we need 11 points.
    # Let's just use a few random restarts as they are effective for this size.
    
    for init in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='Powell', options={'maxiter': 1000, 'disp': False})
            
            if res.success or res.fun < best_score:
                # Verify the solution is valid
                u = res.x[0:N]
                v = res.x[N:2*N]
                
                # Re-apply clipping just in case
                u_clipped = np.clip(u, 0, 1)
                v_clipped = np.clip(v, 0, 1)
                s = u_clipped + v_clipped
                mask = s > 1.0
                if np.any(mask):
                    u_clipped[mask] = u_clipped[mask] / s[mask]
                    v_clipped[mask] = v_clipped[mask] / s[mask]
                    
                points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
                
                # Check if points are actually inside
                # Barycentric coords u,v >= 0 and u+v <= 1 ensures inside.
                
                min_area = compute_min_area(points)
                if min_area > best_score:
                    best_score = min_area
                    best_params = res.x.copy()
                    
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to random if optimization failed completely
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
        return points

    u = best_params[0:N]
    v = best_params[N:2*N]
    
    # Final clipping
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        
    points = np.array([bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
    return points
