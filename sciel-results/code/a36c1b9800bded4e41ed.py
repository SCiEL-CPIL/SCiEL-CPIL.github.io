import numpy as np
from scipy.optimize import minimize
import time

N = 11
V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2]
])

# Pre-compute triangle indices
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            indices.append((i, j, k))
I = np.array([t[0] for t in indices])
J = np.array([t[1] for t in indices])
K = np.array([t[2] for t in indices])

def soft_min(x, beta):
    m = np.min(x)
    # Handle potential numerical issues if x is very large or beta very large
    # exp(-beta * (x - m)) is always <= 1, so no overflow
    return m - np.log(np.sum(np.exp(-beta * (x - m)))) / beta

def points_from_logits(u):
    """Convert logits to Cartesian points inside the triangle."""
    u = u.reshape(N, 3)
    # Subtract max for numerical stability
    u_shifted = u - np.max(u, axis=1, keepdims=True)
    exp_u = np.exp(u_shifted)
    lambda_mat = exp_u / np.sum(exp_u, axis=1, keepdims=True)
    return lambda_mat @ V

def compute_areas(points):
    PI = points[I]
    PJ = points[J]
    PK = points[K]
    V1 = PJ - PI
    V2 = PK - PI
    cross = V1[:, 0] * V2[:, 1] - V1[:, 1] * V2[:, 0]
    return 0.5 * np.abs(cross)

def objective(u, beta):
    points = points_from_logits(u)
    areas = compute_areas(points)
    return soft_min(areas, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -np.inf
    best_u = None
    
    # Initial guesses
    # 1. Random
    # 2. Structured (grid-like in barycentric)
    
    rng = np.random.default_rng(42)
    
    # Create a set of initial logits
    inits = []
    # Random inits
    for _ in range(5):
        inits.append(rng.uniform(-2, 2, size=(N, 3)))
        
    # A structured init: distribute points roughly uniformly in barycentric coords
    # Then convert to logits (log(lambda))
    # This is a bit tricky, just use random for now, or specific patterns.
    # Let's add one init where points are placed on a grid in barycentric space
    # and then converted to logits.
    # Actually, just random is fine with multiple betas.
    
    betas = [5, 10, 20, 50, 100, 200, 400]
    
    current_u = inits[0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Try optimizing from the current best solution
        # Also try a few random restarts if time permits
        candidates = [current_u]
        
        # Add some random restarts if we have time
        if time.time() < deadline - 30:
             for i in range(1, min(len(inits), 3)):
                 candidates.append(inits[i])
        
        for u0 in candidates:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(objective, u0, args=(beta,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                
                if res.success or res.fun < objective(current_u, beta):
                    # Evaluate the hard min score for this solution
                    points = points_from_logits(res.x)
                    areas = compute_areas(points)
                    score = np.min(areas)
                    
                    if score > best_score:
                        best_score = score
                        best_u = res.x
                        current_u = res.x
                        
            except Exception:
                pass
                
    # Final check: if best_u is still None (shouldn't happen), use first init
    if best_u is None:
        best_u = inits[0]
        
    points = points_from_logits(best_u)
    return points

# Test locally? No, just provide the function.
