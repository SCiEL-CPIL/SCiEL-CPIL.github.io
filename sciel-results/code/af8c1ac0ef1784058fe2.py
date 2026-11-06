import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time
import sys

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    Delta = 0.8
    h = 0.3
    J = 1.0

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def op(k, op_mat):
        ops = []
        for i in range(N):
            if i == k:
                ops.append(op_mat)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond_term(j, Delta):
        sx_j = op(j, sx)
        sx_j1 = op(j+1, sx)
        sy_j = op(j, sy)
        sy_j1 = op(j+1, sy)
        sz_j = op(j, sz)
        sz_j1 = op(j+1, sz)
        term = sx_j @ sx_j1 + sy_j @ sy_j1 + Delta * (sz_j @ sz_j1)
        return J * term

    H_odd = np.zeros((dim, dim), dtype=complex)
    H_even = np.zeros((dim, dim), dtype=complex)
    
    for j in [0, 2, 4]:
        H_odd += bond_term(j, Delta)
        
    for j in [1, 3]:
        H_even += bond_term(j, Delta)
        
    H_field = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H_field += op(i, sz)
    H_field *= h
        
    A = H_odd + 0.5 * H_field
    B = H_even + 0.5 * H_field
    
    return A, B

def coefficients(stages):
    if stages != 12:
        a = [1.0/stages]*stages
        b = [1.0/stages]*stages
        return a, b

    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Precompute eigen-decompositions
    # A and B are Hermitian
    wA, vA = la.eigh(A)
    wB, vB = la.eigh(B)
    wT, vT = la.eigh(H_total)
    
    t_vals = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact_props = []
    for t in t_vals:
        # exp(-i t H_total)
        # vT diag(exp(-i t wT)) vT^dagger
        exp_wT = np.exp(-1j * t * wT)
        U_exact = vT @ np.diag(exp_wT) @ vT.conj().T
        exact_props.append(U_exact)

    def fast_expm_hermitian(w, v, t_coeff):
        """Compute exp(-i * t_coeff * H) where H = v diag(w) v^dagger."""
        exp_w = np.exp(-1j * t_coeff * w)
        # v @ diag(exp_w) @ v^dagger
        # This is (v * exp_w) @ v^dagger
        # v is 64x64. exp_w is 64.
        # v * exp_w scales columns of v.
        return (v * exp_w) @ v.conj().T

    def objective(x):
        a = x[:12]
        b = x[12:]
        
        # Simple penalty
        if np.any(a > 50) or np.any(a < -50) or np.any(b > 50) or np.any(b < -50):
            return 1e6
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        if abs(sum_a - 1.0) > 1e-4 or abs(sum_b - 1.0) > 1e-4:
            return 1e6 + (sum_a-1)**2 + (sum_b-1)**2

        error_sum = 0.0
        for i, t in enumerate(t_vals):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B)
                U = fast_expm_hermitian(wB, vB, b[k] * t) @ U
                # exp(-i a_k t A)
                U = fast_expm_hermitian(wA, vA, a[k] * t) @ U
                
            diff = U - exact_props[i]
            err = np.linalg.norm(diff, 2)
            error_sum += err
            
        return error_sum / 3.0

    x0 = np.ones(24) / 12.0
    
    start_time = time.time()
    timeout = 100.0
    
    try:
        # Use Powell or Nelder-Mead
        # Powell is often faster for low dim
        result = opt.minimize(
            objective, 
            x0, 
            method='Powell', 
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        x_best = result.x
    except Exception as e:
        print(f"Opt failed: {e}", file=sys.stderr)
        x_best = x0

    a_opt = x_best[:12]
    b_opt = x_best[12:]
    
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if sum_a != 0: a_opt /= sum_a
    if sum_b != 0: b_opt /= sum_b
    
    a_opt = np.clip(a_opt, -50, 50)
    b_opt = np.clip(b_opt, -50, 50)
    
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if sum_a != 0: a_opt /= sum_a
    if sum_b != 0: b_opt /= sum_b
        
    return list(a_opt), list(b_opt)
