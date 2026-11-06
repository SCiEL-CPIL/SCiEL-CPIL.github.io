"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Compute energy and gradient for points on the sphere.
    x is a flattened array of shape (N*3,).
    We treat the problem as minimizing on the manifold S^2 x ... x S^2.
    The gradient returned is the Euclidean gradient, which we will project in the optimizer.
    """
    pts = x.reshape(N, 3)
    
    # Compute pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: E = - sum_{i<j} log(||x_i - x_j||)
    iu = np.triu_indices(N, k=1)
    dists = np.sqrt(dists_sq[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Gradient:
    # d/dx_i [-log(||x_i - x_j||)] = -(x_i - x_j) / ||x_i - x_j||^2
    # grad_i = sum_{j != i} [ -(x_i - x_j) / ||x_i - x_j||^2 ]
    # Note: The term for pair (i,j) contributes to both grad_i and grad_j.
    # In the sum over j != i, we include all j.
    
    # inv_dists_sq is (N, N)
    inv_dists_sq = 1.0 / dists_sq
    
    # grad[i] = - sum_j inv_dists_sq[i,j] * (pts[i] - pts[j])
    #         = - pts[i] * sum_j inv_dists_sq[i,j] + sum_j inv_dists_sq[i,j] * pts[j]
    
    row_sums = np.sum(inv_dists_sq, axis=1)  # (N,)
    # weighted_pts[i] = sum_j inv_dists_sq[i,j] * pts[j]
    weighted_pts = np.sum(inv_dists_sq[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_pts
    
    return energy, grad.ravel()

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _riemannian_minimize(points, maxiter=2000, tol=1e-12):
    """
    Minimize energy on the sphere using L-BFGS-B on the Euclidean embedding,
    but with a projection step or simply relying on the fact that L-BFGS-B
    will try to stay close if we start on the sphere? 
    Actually, standard L-BFGS-B does not respect the manifold.
    We need to either:
    1. Use a manifold-aware optimizer (not directly in scipy).
    2. Use a penalty method.
    3. Use a Riemannian gradient descent manually.
    
    Let's use a manual Riemannian gradient descent with line search, 
    or better, use L-BFGS-B but project back to the sphere after every step?
    No, that breaks the line search.
    
    Better approach: Use `scipy.optimize.minimize` with method 'L-BFGS-B' 
    is bad for manifolds. 
    Let's implement a simple Riemannian Gradient Descent with Barzilai-Borwein step size 
    or just fixed step size with backtracking.
    
    Alternatively, we can use the fact that for small steps, the projection
    is a good approximation. But L-BFGS takes variable steps.
    
    Let's stick to a robust manual Riemannian GD with adaptive step size.
    """
    pts = _normalize(points.copy())
    current_energy = _log_energy(pts)
    
    # Precompute some constants
    step_size = 0.01
    max_steps = maxiter
    
    for step in range(max_steps):
        # Check time
        if time.time() > TIME_LIMIT - 5:
            break
            
        e, g = _energy_and_grad(pts.ravel())
        
        # Check gradient norm (Riemannian)
        # Project gradient to tangent space: g_tan = g - (g . x) x
        g_proj = g.reshape(N, 3)
        dot = np.sum(g_proj * pts, axis=1, keepdims=True)
        g_tan = g_proj - dot * pts
        
        grad_norm = np.linalg.norm(g_tan)
        
        if grad_norm < tol:
            break
            
        # Barzilai-Borwein or just simple adaptive
        # Let's use a simple backtracking line search on the sphere
        
        # Direction is -g_tan
        direction = -g_tan
        
        # Normalize direction to unit length for consistent step size interpretation?
        # No, keep it as is, step_size scales it.
        
        # Line search
        alpha = step_size
        for _ in range(20):
            # Move in tangent space
            new_pts = pts + alpha * direction
            # Project back to sphere
            new_pts = _normalize(new_pts)
            
            new_energy = _log_energy(new_pts)
            
            if new_energy < current_energy - 1e-12 * abs(current_energy):
                pts = new_pts
                current_energy = new_energy
                break
            else:
                alpha *= 0.5
        
        # If we didn't improve, maybe reduce step size permanently?
        # For now, just break if alpha is too small
        if alpha < 1e-8:
            break

    return pts, current_energy

def _kick(points, strength, rng):
    """
    Apply a random perturbation (kick) to escape local minima.
    """
    pts = points.copy()
    # Random directions
    random_dirs = rng.standard_normal((N, 3))
    # Project to tangent space
    dot = np.sum(random_dirs * pts, axis=1, keepdims=True)
    tangent_dirs = random_dirs - dot * pts
    # Normalize tangent directions
    norms = np.linalg.norm(tangent_dirs, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    tangent_dirs = tangent_dirs / norms
    
    pts = pts + strength * tangent_dirs
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)
    
    # 1. Initial refinement
    refined, e = _riemannian_minimize(best_points, maxiter=1000)
    if e < best_energy:
        best_energy = e
        best_points = refined
    
    # 2. Simulated Annealing / Kick loop
    rng = np.random.default_rng(42)
    current_temp = 0.1  # Initial kick strength
    min_temp = 0.001
    cooling_rate = 0.95
    
    # We will try to improve from the best found so far
    # But we should also keep a portfolio of "recent" good solutions?
    # For simplicity, we kick the current best and re-optimize.
    
    kick_count = 0
    max_kicks = 200
    
    while kick_count < max_kicks:
        if time.time() > deadline - 10:
            break
            
        # Generate a kicked version
        kicked_pts = _kick(best_points, current_temp, rng)
        
        # Optimize the kicked version
        # Use a limited number of iterations to save time
        # If it's good, we might spend more time?
        # Let's use a moderate number of iterations.
        refined_kicked, e_kicked = _riemannian_minimize(kicked_pts, maxiter=500)
        
        # Metropolis criterion?
        # If it's better, accept.
        # If it's worse, accept with probability exp(-(e_kicked - best_energy) / temp_energy)
        # But we don't have a separate energy temperature. 
        # Simple hill climbing with kicks is often enough for this scale if kicks are large enough.
        # Let's use a simple acceptance: if better, accept. 
        # To allow escaping, we should sometimes accept worse solutions.
        # Let's use a simple simulated annealing scheme.
        
        delta_e = e_kicked - best_energy
        if delta_e < 0:
            # Better
            best_points = refined_kicked
            best_energy = e_kicked
            # Reset cooling? Or just continue.
            # Maybe increase temp slightly if we found a new global min?
            # No, just cool down.
        else:
            # Worse
            # Accept with probability exp(-delta_e / current_temp)
            # But current_temp is kick strength, not energy temp.
            # Let's define an energy temperature for acceptance.
            # T_energy = current_temp * 10? 
            # Let's just use a fixed acceptance probability for small worsenings.
            # If delta_e < 0.1, accept with prob 0.5.
            # This is ad-hoc.
            # Let's stick to strict improvement for stability, but rely on larger kicks 
            # to jump over barriers.
            # Actually, for N=150, the landscape is complex. 
            # Let's try to accept if the kick was large enough to potentially jump a barrier.
            # A simpler heuristic: if e_kicked < best_energy + 1.0, accept it as the new "current" 
            # but keep the global best. Then kick from the "current".
            
            # Let's implement a proper SA loop.
            # We need a separate temperature for the energy.
            # Let's redefine the loop.
            pass
            
        # Cool down the kick strength
        current_temp *= cooling_rate
        kick_count += 1
        
        # If we are stuck (no improvement for many kicks), increase kick strength
        # We didn't track improvements here properly because of the 'pass'.
        # Let's restart the logic inside the loop properly.
        
    # The above loop was a bit messy. Let's implement a cleaner SA in the next block?
    # No, I need to output the complete program. 
    # Let's refine the SA logic in the code below.
    
    # Re-do the SA loop properly
    current_pts = best_points.copy()
    current_energy = best_energy
    temp_kick = 0.1
    temp_energy = 1.0  # Energy temperature for Metropolis
    
    kicks_since_improve = 0
    
    for kick_count in range(200):
        if time.time() > deadline - 10:
            break
            
        kicked_pts = _kick(current_pts, temp_kick, rng)
        refined_kicked, e_kicked = _riemannian_minimize(kicked_pts, maxiter=500)
        
        delta_e = e_kicked - current_energy
        
        if delta_e < 0:
            # Accept
            current_pts = refined_kicked
            current_energy = e_kicked
            kicks_since_improve = 0
            
            if e_kicked < best_energy:
                best_energy = e_kicked
                best_points = refined_kicked
        else:
            # Metropolis acceptance
            prob = math.exp(-delta_e / temp_energy)
            if rng.random() < prob:
                current_pts = refined_kicked
                current_energy = e_kicked
                kicks_since_improve += 1
            else:
                kicks_since_improve += 1
        
        # Cooling
        temp_kick *= 0.98
        temp_energy *= 0.98
        
        # If stuck, reheat
        if kicks_since_improve > 20:
            temp_kick = 0.1
            temp_energy = 1.0
            kicks_since_improve = 0
            
    # Final polish
    if time.time() < deadline - 5:
        final_pts, e_final = _riemannian_minimize(best_points, maxiter=2000, tol=1e-14)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_pts

    return _normalize(best_points)

# EVOLVE-BLOCK-END
