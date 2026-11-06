"""
Strategy: Riemannian Optimization on Stiefel Manifold + Simulated Annealing.
Instead of optimizing in R^3 and projecting, we treat the 100 points as rows of a 
matrix P in R^{100x3} such that P P^T = I (Stiefel manifold V_100(R^3)).
We use a Riemannian Trust-Region method (approximated via L-BFGS on the manifold 
with polar retraction) combined with a Simulated Annealing phase to escape 
shallow local minima.
"""

import numpy as np
import math
import time
from scipy.linalg import polar

N = 100

def _compute_energy_and_grad(P):
    """
    Compute energy and Riemannian gradient for P (N x 3) on Stiefel manifold.
    Energy: sum_{i<j} log(1/d_{ij})
    Euclidean Gradient G: dE/dP_{ik} = - sum_{j!=i} (P_{ik} - P_{jk}) / d_{ij}^2
    Riemannian Gradient: grad_E = G - P (P^T G)
    """
    n = P.shape[0]
    # Pairwise distances
    # D2[i,j] = ||P_i - P_j||^2 = 2 - 2 P_i . P_j
    PdotP = P @ P.T
    D2 = 2.0 - 2.0 * PdotP
    np.fill_diagonal(D2, 1.0) # Avoid division by zero, though we mask later
    
    # Distances
    D = np.sqrt(D2)
    
    # Mask for upper triangle i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    D_upper = D[mask]
    
    if np.any(D_upper < 1e-15):
        return -1e18, np.zeros_like(P)
        
    energy = math.fsum(np.log(1.0 / D_upper).tolist())
    
    # Compute Euclidean Gradient
    # G[i,k] = - sum_{j!=i} (P[i,k] - P[j,k]) / D2[i,j]
    # Let W be the matrix of weights: W[i,j] = 1/D2[i,j] for i!=j, 0 otherwise.
    # G = -P * (row_sum(W)) + W @ P
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    
    row_sum_W = np.sum(W, axis=1)
    # G[i,k] = -P[i,k] * sum_j W[i,j] + sum_j W[i,j] P[j,k]
    G = -P * row_sum_W[:, np.newaxis] + W @ P
    
    # Riemannian Gradient Projection
    # grad = G - P (P^T G)
    P_T_G = P.T @ G
    grad = G - P @ P_T_G
    
    return energy, grad

def _retract(P, X):
    """
    Riemannian Retraction on Stiefel Manifold using Polar Decomposition.
    X is a tangent vector (P^T X = 0).
    Retraction: R_P(X) = P + X - 1/2 P (P^T X) ... wait, for Stiefel, 
    a common retraction is the polar factor of (P + X).
    However, if X is a valid tangent vector, P+X is close to the manifold.
    Polar decomposition: A = UP. We take U.
    """
    A = P + X
    # Polar decomposition A = U H
    U, H = polar(A)
    return U

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_rtr(P0, deadline, max_iter=2000):
    """
    Riemannian Trust-Region style optimization.
    We implement a simplified RTR loop:
    1. Compute Riemannian Gradient.
    2. Approximate Hessian or use L-BFGS memory on manifold.
       For simplicity and robustness, we use a gradient descent step with 
       adaptive step size (barzilai-borwein style) on the manifold, 
       which is often as effective as RTR for this problem if well-tuned.
       Alternatively, we can use scipy's L-BFGS-B on the flattened manifold 
       with a custom Jacobian that maps to tangent space? No, scipy doesn't 
       handle manifolds.
       
       Let's implement a simple Riemannian Gradient Descent with Barzilai-Borwein 
       step sizes, which is robust and fast.
    """
    P = P0.copy()
    e, g = _compute_energy_and_grad(P)
    
    # Initial step size
    step = 0.01
    prev_P = P.copy()
    prev_g = g.copy()
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check convergence
        if np.linalg.norm(g) < 1e-8:
            break
            
        # Barzilai-Borwein Step Size
        # s = P - prev_P
        # y = g - prev_g
        # step = <s, s> / <s, y> (BB1) or <s, y> / <y, y> (BB2)
        s = P - prev_P
        y = g - prev_g
        
        s_norm_sq = np.sum(s * s)
        y_norm_sq = np.sum(y * y)
        s_dot_y = np.sum(s * y)
        
        if s_dot_y > 1e-12:
            step_bb1 = s_norm_sq / s_dot_y
            step_bb2 = s_dot_y / y_norm_sq
            # Alternate or average
            step = 0.5 * (step_bb1 + step_bb2)
            
        # Clip step size to prevent divergence
        step = np.clip(step, 1e-6, 0.1)
        
        # Update
        P_new = _retract(P, -step * g)
        
        # Check for improvement
        e_new, g_new = _compute_energy_and_grad(P_new)
        
        if e_new < e:
            P = P_new
            e = e_new
            g = g_new
            prev_P = P.copy()
            prev_g = g.copy()
        else:
            # Backtrack
            step *= 0.5
            # If step becomes too small, break or accept
            if step < 1e-8:
                break

    return P, e

def _simulated_annealing(P0, deadline, T0=1.0, T1=0.01, alpha=0.99):
    """
    Simulated Annealing to escape local minima.
    """
    P = P0.copy()
    e, g = _compute_energy_and_grad(P)
    T = T0
    
    start_time = time.time()
    
    while T > T1:
        if time.time() > deadline:
            break
            
        # Generate random perturbation in tangent space
        # Z is random N x 3
        Z = np.random.randn(N, 3)
        # Project to tangent space: Z_tan = Z - P (P^T Z)
        P_T_Z = P.T @ Z
        Z_tan = Z - P @ P_T_Z
        
        # Normalize perturbation
        Z_norm = np.linalg.norm(Z_tan)
        if Z_norm > 1e-10:
            Z_tan = Z_tan / Z_norm
            
        # Step size proportional to T
        step = 0.1 * T
        
        P_new = _retract(P, step * Z_tan)
        e_new, _ = _compute_energy_and_grad(P_new)
        
        # Accept if better or with probability exp(-delta/T)
        delta = e_new - e
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            P = P_new
            e = e_new
            
        T *= alpha
        
    return P, e

def construct():
    """
    Construct 100 points on unit sphere minimizing logarithmic energy.
    Strategy:
    1. Start with diverse initializations (Fibonacci, Random).
    2. Apply Simulated Annealing to explore the landscape and escape local minima.
    3. Polish with Riemannian Gradient Descent (BB step) for high precision.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_P = None
    best_energy = np.inf
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _random_sphere(N, seed=4)
    ]
    
    # We have ~145s.
    # Allocate time for SA and Polish for each init.
    # SA is expensive. Let's do 2-3 SA runs and polish the best.
    
    # Phase 1: Simulated Annealing from diverse starts
    # We'll run SA for ~20s each on 3 best starts, then polish.
    
    sa_results = []
    
    for i, P0 in enumerate(inits[:3]): # Top 3 starts
        if time.time() > deadline - 30.0:
            break
            
        # Run SA for ~20s
        P_sa, e_sa = _simulated_annealing(P0, deadline - 30.0 - i*5.0)
        sa_results.append((P_sa, e_sa))
        
    if not sa_results:
        # Fallback if time ran out
        P_sa, e_sa = _fibonacci_sphere(N), _compute_energy_and_grad(_fibonacci_sphere(N))[0]
        sa_results.append((P_sa, e_sa))
        
    # Find best from SA
    best_P_sa, best_e_sa = min(sa_results, key=lambda x: x[1])
    best_P = best_P_sa
    best_energy = best_e_sa
    
    # Phase 2: Polish with RTR/GD
    # Run multiple polish rounds from the best SA result, and maybe from other SA results if time permits.
    
    polish_candidates = [best_P_sa]
    # Add other SA results if they are close
    for P_sa, e_sa in sa_results[1:]:
        if e_sa < best_e_sa * 1.01: # Within 1%
            polish_candidates.append(P_sa)
            
    for P0 in polish_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Run RTR/GD polish
        P_pol, e_pol = _optimize_rtr(P0, deadline - 5.0, max_iter=5000)
        
        if e_pol < best_energy:
            best_energy = e_pol
            best_P = P_pol
            
    # Final check
    if best_P is None:
        best_P = _fibonacci_sphere(N)
        
    return best_P

# EVOLVE-BLOCK-END
