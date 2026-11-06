"""
Strategy: Multi-Start L-BFGS-B with Analytical Gradients via State-Vector Chain Rule.
This approach abandons the coarse grid search and derivative-free Nelder-Mead refinement.
Instead, it leverages the low dimensionality (4 parameters for p=2) to perform high-precision
gradient-based optimization. We compute the expectation value and its exact analytical
gradient by propagating the state vector and tracking the conjugate states.
Multiple diverse starting points are used to mitigate the risk of local minima, 
and the best result is returned.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time
import sys
import itertools

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
    if n == 0:
        return np.zeros((1,1), dtype=complex), np.zeros((1,1), dtype=complex)
        
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

    C = np.zeros((dim, dim), dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (np.eye(dim, dtype=complex) - ZZ)
        
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _compute_value_and_grad(n, edges, angles, C, B):
    """
    Compute the expected cut value and its gradient with respect to the angles.
    angles: (p, 2) array of (gamma, beta)
    
    We use the state vector chain rule.
    |psi_k> = U_B(beta_k) U_C(gamma_k) ... U_B(beta_1) U_C(gamma_1) |psi_0>
    
    Let |psi> be the final state.
    Value V = <psi| C |psi>
    
    Gradient dV/dtheta_j = -2 Im( <psi| (dU/dtheta_j) U^-1 |psi> C |psi> ) 
    Or more simply, propagate forward and backward.
    
    Forward:
    |psi_0> = |+>^n
    |psi_k> = U_k |psi_{k-1}>
    
    Backward (conjugate state <phi_k|):
    <phi_p| = <psi_p| C
    <phi_{k-1}| = <phi_k| U_k^†
    
    Then dV/dtheta_k = -2 Im( <phi_k| (dU_k/dtheta_k) |psi_{k-1}> )
    
    U_k = exp(-i beta_k B) exp(-i gamma_k C)
    
    dU_k/dgamma_k = -i C exp(-i beta_k B) exp(-i gamma_k C)
    dU_k/dbeta_k = exp(-i beta_k B) (-i B) exp(-i gamma_k C)
    
    Let A = exp(-i gamma_k C) and Bk = exp(-i beta_k B).
    U_k = Bk A.
    
    dU_k/dgamma_k = Bk (-i C A)
    dU_k/dbeta_k = (-i B Bk) A
    
    So:
    grad_gamma_k = -2 Im( <phi_k| Bk (-i C A) |psi_{k-1}> )
    grad_beta_k = -2 Im( <phi_k| (-i B Bk) A |psi_{k-1}> )
    
    Note: <phi_k| is the row vector.
    """
    p = len(angles)
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Store intermediate states and unitaries for backward pass
    psis = [psi]
    U_Cs = []
    U_Bs = []
    
    for k in range(p):
        gamma, beta = angles[k]
        U_C = expm(-1j * gamma * C)
        U_B = expm(-1j * beta * B)
        
        # Apply U_C then U_B
        psi = U_B @ (U_C @ psi)
        psis.append(psi)
        U_Cs.append(U_C)
        U_Bs.append(U_B)
        
    # Backward pass
    # phi_p = <psi_p| C
    phi = (C @ psis[p]).conj() # This is C^† |psi> = C |psi> since C is Hermitian.
    # Wait, <phi| = <psi| C. So phi vector is (C^† |psi>)^†? No.
    # Let's work with ket vectors for the backward state.
    # Let |chi_k> be the state such that <chi_k| = <phi_k|.
    # <phi_p| = <psi_p| C  => |chi_p> = C |psi_p> (since C is Hermitian)
    
    chi = C @ psis[p]
    
    grads = np.zeros((p, 2), dtype=float)
    
    for k in range(p - 1, -1, -1):
        gamma, beta = angles[k]
        U_C = U_Cs[k]
        U_B = U_Bs[k]
        
        # Current chi is |chi_k>
        # We need terms for grad_gamma_k and grad_beta_k
        
        # Term for gamma: <chi_k| U_B (-i C U_C) |psi_{k-1}>
        # Let v_gamma = U_B (-i C U_C) |psi_{k-1}>
        # grad_gamma_k = -2 Im( <chi_k| v_gamma > )
        
        # Term for beta: <chi_k| (-i B U_B) U_C |psi_{k-1}>
        # Let v_beta = (-i B U_B) U_C |psi_{k-1}>
        # grad_beta_k = -2 Im( <chi_k| v_beta > )
        
        psi_prev = psis[k]
        
        # Calculate v_gamma
        # C @ psi_prev
        C_psi_prev = C @ psi_prev
        # -i C U_C psi_prev = -i C_psi_prev? No.
        # U_C psi_prev is part of the forward step.
        # Actually, we have psis[k] = U_B U_C psi_prev.
        # Let's compute the terms directly.
        
        # v_gamma = U_B @ (-1j * C @ (U_C @ psi_prev))
        # Note: U_C @ psi_prev is the state after cost update but before mixing.
        # Let's call it psi_mid.
        psi_mid = U_C @ psi_prev
        C_psi_mid = C @ psi_mid
        v_gamma = U_B @ (-1j * C_psi_mid)
        
        # v_beta = (-1j * B @ U_B) @ (U_C @ psi_prev)
        # B @ U_B @ psi_mid
        B_U_B_psi_mid = B @ (U_B @ psi_mid)
        v_beta = -1j * B_U_B_psi_mid
        
        # Inner products
        # <chi_k| v_gamma> = vdot(chi, v_gamma)
        ip_gamma = np.vdot(chi, v_gamma)
        ip_beta = np.vdot(chi, v_beta)
        
        grads[k, 0] = -2.0 * ip_gamma.imag
        grads[k, 1] = -2.0 * ip_beta.imag
        
        # Update chi for next iteration (k-1)
        # <chi_{k-1}| = <chi_k| U_k^†
        # |chi_{k-1}> = (U_k^†)^† |chi_k> = U_k |chi_k>
        # U_k = U_B U_C
        chi = U_B @ (U_C @ chi)
        
    value = np.vdot(psis[p], C @ psis[p]).real
    return value, grads

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Precompute Hamiltonians
    C, B = _build_hamiltonians(n, edges)
    
    # Define objective function for L-BFGS-B
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9, np.zeros_like(x)
        x_reshaped = x.reshape(p, 2)
        val, grad = _compute_value_and_grad(n, edges, x_reshaped, C, B)
        return -val, -grad.flatten()
        
    # Multi-start strategy
    # We define several initial guesses.
    # 1. Adiabatic intuition: small gamma, beta ~ pi/4
    # 2. Random uniform in [0, pi] x [0, pi/2]
    # 3. Known good points for p=2 from literature (approx)
    
    starts = []
    
    # Start 1: Adiabatic-like
    starts.append(np.array([[0.1, 0.5], [0.1, 0.5]]))
    
    # Start 2: Symmetric mid-range
    starts.append(np.array([[0.5, 0.5], [0.5, 0.5]]))
    
    # Start 3: Random
    np.random.seed(42)
    for _ in range(5):
        g1 = np.random.uniform(0, np.pi)
        b1 = np.random.uniform(0, np.pi/2)
        g2 = np.random.uniform(0, np.pi)
        b2 = np.random.uniform(0, np.pi/2)
        starts.append(np.array([[g1, b1], [g2, b2]]))
        
    best_val = -1e9
    best_angles = None
    
    for start in starts:
        if _time_remaining() < 15.0:
            break
            
        try:
            res = minimize(objective, start.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False, 'ftol': 1e-8})
            
            if res.success or res.fun < -best_val:
                # res.fun is -val
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_angles = res.x.reshape(p, 2)
        except Exception as e:
            pass # Skip failed start
            
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles

# EVOLVE-BLOCK-END
