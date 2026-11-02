import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

def bary_to_cartesian(bary):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, SQRT3 / 2.0])
    x = bary[:, 1] * v1[0] + bary[:, 2] * v2[0]
    y = bary[:, 1] * v1[1] + bary[:, 2] * v2[1]
    return np.column_stack((x, y))

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def objective(bary_flat):
    """Objective function for minimization. We want to maximize min area, so minimize -min area."""
    bary = bary_flat.reshape(N, 3)
    # Ensure valid barycentric coordinates
    bary = np.maximum(bary, 0.0)
    sums = bary.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    bary = bary / sums
    
    points = bary_to_cartesian(bary)
    area = min_triangle_area(points)
    return -area

def generate_sobol_initializations(num_starts):
    """Generate initial barycentric coordinates using Sobol sequence."""
    # We need 3*N dimensions for barycentric coords, but they are constrained.
    # It's better to generate points in the triangle directly or generate random barycentric coords.
    # Sobol works in [0,1]^d. We can map [0,1]^(3N) to barycentric coords.
    # A simple mapping: generate 3N uniform randoms, normalize each triple.
    
    sampler = Sobol(d=3*N, scramble=True)
    # Number of points must be a power of 2 for Sobol to be efficient, but we can just take the first num_starts
    # Actually, Sobol requires power of 2 for the sequence length if we want low discrepancy properties strictly, 
    # but for initialization, we can just generate a block.
    # Let's generate 2**10 points and take the first num_starts.
    n_samples = 1024 
    samples = sampler.random(n_samples)
    
    inits = []
    for i in range(min(num_starts, n_samples)):
        bary = samples[i].reshape(N, 3)
        # Normalize to be valid barycentric
        bary = np.maximum(bary, 0.0)
        sums = bary.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        bary = bary / sums
        inits.append(bary)
    return inits

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    best_bary = None
    best_area = -float('inf')
    
    # Generate multiple initial points
    num_starts = 20
    inits = generate_sobol_initializations(num_starts)
    
    # Add one symmetric "center" start as well
    center_bary = np.zeros((N, 3))
    center_bary[:, 0] = 1.0/N
    center_bary[:, 1] = 1.0/N
    center_bary[:, 2] = 1.0/N
    inits.append(center_bary)
    
    for init_bary in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
            
        # Run Nelder-Mead
        # We flatten the barycentric coordinates
        x0 = init_bary.flatten()
        
        # Define bounds? Nelder-Mead doesn't support bounds directly.
        # We rely on the objective function to handle invalid coords by projecting/clipping.
        # However, clipping in the objective function makes the gradient discontinuous.
        # For Nelder-Mead, this is often acceptable.
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            # Check if it improved
            if res.fun < best_area: # Note: objective is -area, so lower is better
                best_area = -res.fun
                best_bary = res.x.reshape(N, 3)
                
                # Clean up best_bary
                best_bary = np.maximum(best_bary, 0.0)
                sums = best_bary.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1.0
                best_bary = best_bary / sums
                
        except Exception:
            pass
            
    # If no result found (should not happen), fallback to center
    if best_bary is None:
        best_bary = inits[-1]
        
    points = bary_to_cartesian(best_bary)
    
    # Final safety check: ensure points are strictly inside.
    # If a point is on the boundary, move it slightly inwards.
    # Check barycentric coords again.
    # If any barycentric coord is 0, it's on the edge.
    # The evaluator might reject boundary points.
    # Let's enforce a small epsilon inside.
    epsilon = 1e-5
    for i in range(N):
        if best_bary[i, 0] < epsilon or best_bary[i, 1] < epsilon or best_bary[i, 2] < epsilon:
             # Push it towards the center
             center = np.array([1.0/3, 1.0/3, 1.0/3])
             # Interpolate slightly
             best_bary[i] = (1 - 0.1) * best_bary[i] + 0.1 * center
             # Re-normalize
             s = best_bary[i].sum()
             best_bary[i] /= s
             
    points = bary_to_cartesian(best_bary)
    return points
