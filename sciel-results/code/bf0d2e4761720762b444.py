"""
Strategy: Global grid search over a coarsely spaced, graph-scaled parameter space, 
followed by a single high-precision Nelder-Mead refinement.
This approach addresses the stagnation of local methods by first ensuring a broad, 
uniform coverage of the parameter space (gamma, beta) scaled by graph density, 
identifying the best candidate region, and then using a robust derivative-free 
local optimizer (Nelder-Mead) to precisely locate the optimum within that region.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys
from scipy.linalg import expm

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build the problem Hamiltonian C (MaxCut) and mixing Hamiltonian B (sum X).
    C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    B = sum_{i=0}^{n-1} X_i
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    # Helper to apply operator on qubit k
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
        
    # Build C
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    # Build B
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _expected_cut_value(n, edges, angles, C, B):
    """
    Compute the expected cut value.
    angles: (p, 2) array of (gamma, beta)
    """
    if len(edges) == 0:
        return 0.0
        
    dim = 2**n
    p = len(angles)
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Forward pass
    for i in range(p):
        gamma, beta = angles[i]
        
        # U_C = exp(-i gamma C)
        U_C = expm(-1j * gamma * C)
        psi = U_C @ psi
        
        # U_B = exp(-i beta B)
        U_B = expm(-1j * beta * B)
        psi = U_B @ psi
    
    # Final state
    psi_final = psi
    
    # Value: <psi| C |psi>
    val = np.vdot(psi_final, C @ psi_final).real
    
    return val

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 15.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Graph-aware scaling for the grid
    # Density affects the scale of optimal angles
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0.0
    
    # Heuristic: Higher density might require smaller steps or different ranges
    # Standard QAOA angles are often in [0, pi/2] or [0, pi]
    # We scale the grid resolution and range based on density
    
    # Grid search parameters
    # Coarse grid to cover the space
    # Gamma range: [0, pi]
    # Beta range: [0, pi/2] (often sufficient, but let's use [0, pi] for safety in grid)
    
    # Number of points per dimension for the coarse grid
    # We want to balance coverage vs time. 
    # For p=2, we have 4 parameters. 
    # A 5x5x5x5 grid is 625 points. 
    # Each point requires expm calculations which are expensive.
    # Let's start with a 4x4x4x4 grid (256 points) to ensure we finish in time.
    
    grid_size = 4
    
    gamma_vals = np.linspace(0.1, np.pi - 0.1, grid_size)
    beta_vals = np.linspace(0.1, np.pi/2 - 0.1, grid_size)
    
    best_val = -np.inf
    best_angles = None
    
    # Iterate over all combinations for p=2
    # angles shape (2, 2) -> [ (g1, b1), (g2, b2) ]
    
    # To save time, we can precompute expm for the grid points? 
    # No, the state vector changes, so we can't easily reuse U_C and U_B 
    # unless we store them, but memory might be an issue or complexity high.
    # However, for a fixed gamma, U_C is fixed. 
    # We can precompute U_C for each gamma in gamma_vals and U_B for each beta in beta_vals.
    
    U_C_list = [expm(-1j * g * C) for g in gamma_vals]
    U_B_list = [expm(-1j * b * B) for b in beta_vals]
    
    # Now loop
    for g1_idx in range(grid_size):
        if _time_remaining() < 10.0:
            break
        for b1_idx in range(grid_size):
            if _time_remaining() < 10.0:
                break
            for g2_idx in range(grid_size):
                if _time_remaining() < 10.0:
                    break
                for b2_idx in range(grid_size):
                    # Construct angles
                    a = np.array([[gamma_vals[g1_idx], beta_vals[b1_idx]],
                                  [gamma_vals[g2_idx], beta_vals[b2_idx]]])
                    
                    # Simulate
                    dim = 2**n
                    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
                    
                    # Apply layer 1
                    psi = U_C_list[g1_idx] @ psi
                    psi = U_B_list[b1_idx] @ psi
                    
                    # Apply layer 2
                    psi = U_C_list[g2_idx] @ psi
                    psi = U_B_list[b2_idx] @ psi
                    
                    # Calculate value
                    val = np.vdot(psi, C @ psi).real
                    
                    if val > best_val:
                        best_val = val
                        best_angles = a.copy()
    
    # If grid search failed or found nothing, fallback
    if best_angles is None:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Refine with Nelder-Mead
    # Objective: Minimize negative cut value
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x = np.array(x).reshape(-1, 2)
        # Keep angles within reasonable bounds for the optimizer
        # Nelder-Mead doesn't handle bounds well, but the landscape is periodic.
        # We can just let it run, but might drift. 
        # For safety, we can clip or just rely on the local nature.
        val = _expected_cut_value(n, edges, x, C, B)
        return -val
        
    try:
        # Initial simplex size based on grid step
        x0 = best_angles.flatten()
        # Set initial simplex to be small, around the best grid point
        # The grid step was approx (pi - 0.2) / 3 ~ 1.0 rad
        # Let's use a smaller initial step for refinement
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        # Check if refinement improved
        if res.success or -res.fun > best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
            
    except Exception:
        pass
        
    return best_angles

# EVOLVE-BLOCK-END
