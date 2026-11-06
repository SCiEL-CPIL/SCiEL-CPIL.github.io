<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """Strang splitting repeated over stages/2 substeps: a = (1/2, 1/2) per substep, b = (1, 0)."""
    substeps = stages // 2
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.linalg as sla
import time
import itertools

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    # Pauli matrices
    I2 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def site_op(op, site):
        """Apply op at given site (0-indexed from left)."""
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def bond_op(op1, op2, site):
        """Apply op1 at site, op2 at site+1."""
        ops = [I2] * n
        ops[site] = op1
        ops[site + 1] = op2
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    # XXZ Hamiltonian: sum_j [Jx(Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] + h * sum_j Sz_j
    # Jx = 1, Jz = 0.8, h = 0.3
    Jx = 1.0
    Jz = 0.8
    h = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(n - 1):
        sxj = bond_op(sx, sx, j)
        syj = bond_op(sy, sy, j)
        szj = bond_op(sz, sz, j)
        bond_term = Jx * (sxj + syj) + Jz * szj
        
        if j % 2 == 0:
            # Even bond (0,2,4) -> B
            B += bond_term
        else:
            # Odd bond (1,3) -> A
            A += bond_term
    
    # Field: split half to A, half to B
    field = np.zeros((dim, dim), dtype=complex)
    for j in range(n):
        field += site_op(sz, j)
    
    A += (h / 2.0) * field
    B += (h / 2.0) * field
    
    return A, B

def _compute_score(a, b, A, B, exact_propagators, times):
    """Compute mean spectral norm error over the three times."""
    errors = []
    for i, t in enumerate(times):
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            # Apply exp(-i a_k t A) first, then exp(-i b_k t B)
            # The product is: ... * exp(-i b_k t B) * exp(-i a_k t A) * ...
            # So we multiply U on the right by exp(-i a_k t A) then exp(-i b_k t B)
            U = U @ sla.expm(-1j * a[k] * t * A)
            U = U @ sla.expm(-1j * b[k] * t * B)
        
        error = np.linalg.norm(U - exact_propagators[i], 2)
        errors.append(error)
    return np.mean(errors)

def _surrogate_objective(coeffs, A, B, exact_propagators, times, n_stages):
    """Smooth surrogate: sum of squared errors across all times."""
    a = coeffs[:n_stages]
    b = coeffs[n_stages:2*n_stages]
    
    # Enforce constraints softly: sum(a) = 1, sum(b) = 1
    # We'll handle this by projecting after optimization
    
    errors = []
    for t in times:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(n_stages):
            U = U @ sla.expm(-1j * a[k] * t * A)
            U = U @ sla.expm(-1j * b[k] * t * B)
        exact = sla.expm(-1j * t * (A + B))
        diff = U - exact
        # Use Frobenius norm squared as smooth surrogate
        errors.append(np.vdot(diff.flatten(), diff.flatten()).real)
    return np.sum(errors)

def _project_constraints(coeffs, n_stages):
    """Project coefficients to satisfy sum(a)=1, sum(b)=1, and bounds [-50,50]."""
    a = coeffs[:n_stages].copy()
    b = coeffs[n_stages:2*n_stages].copy()
    
    # Project to sum = 1
    a_sum = np.sum(a)
    if a_sum != 0:
        a = a - a + (a - np.mean(a)) + 1.0 / n_stages
    else:
        a = np.ones(n_stages) / n_stages
    
    b_sum = np.sum(b)
    if b_sum != 0:
        b = b - b + (b - np.mean(b)) + 1.0 / n_stages
    else:
        b = np.ones(n_stages) / n_stages
    
    # Clip to bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-project after clipping
    a = a / np.sum(a) * 1.0 if np.sum(a) != 0 else np.ones(n_stages) / n_stages
    b = b / np.sum(b) * 1.0 if np.sum(b) != 0 else np.ones(n_stages) / n_stages
    
    return np.concatenate([a, b])

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

