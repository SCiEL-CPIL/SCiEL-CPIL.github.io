import numpy as np
from scipy.linalg import expm
import time

# Build the 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3
# Split into even and odd bonds (each including half of the uniform field)

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ Heisenberg chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Single-site operators
    I2 = np.eye(2, dtype=complex)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Tensor product helpers
    def tensor(*ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** N
    zero = np.zeros((dim, dim), dtype=complex)
    
    # Build operators on specific sites
    def site_op(site, op, N=N):
        """Apply operator to a specific site."""
        ops = [I2] * N
        ops[site] = op
        return tensor(*ops)
    
    # XXZ interaction: J_x * (Sx_i Sx_j + Sy_i Sy_j) + J_z * Sz_i Sz_j
    # With anisotropy delta = J_z / J_x, so J_x = 1, J_z = delta
    # Actually, let's use the standard form:
    # H = sum_k [ Sx_k Sx_{k+1} + Sy_k Sy_{k+1} + delta * Sz_k Sz_{k+1} ] - h * sum_k Sz_k
    # Split: even bonds (0-1, 2-3, 4-5) go to one half, odd bonds (1-2, 3-4, 5-6... wait)
    # For 6 sites: bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Even bonds: (0,1), (2,3), (4,5) -> these are "even" in some convention
    # Odd bonds: (1,2), (3,4) -> these are "odd"
    # Actually, for a chain of 6 sites, bonds are indexed 0..4
    # Even-indexed bonds: 0, 2, 4 -> (0,1), (2,3), (4,5)
    # Odd-indexed bonds: 1, 3 -> (1,2), (3,4)
    
    # Let's define:
    # A = odd bonds + half field
    # B = even bonds + half field
    
    bonds_even = [(0, 1), (2, 3), (4, 5)]  # even-indexed bonds
    bonds_odd = [(1, 2), (3, 4)]  # odd-indexed bonds
    
    delta = anisotropy
    h = field
    
    # Build A (odd bonds + half field)
    A = zero.copy()
    for (i, j) in bonds_odd:
        Sx_i_Sx_j = np.kron(site_op(i, Sx), site_op(j, Sx))
        Sy_i_Sy_j = np.kron(site_op(i, Sy), site_op(j, Sy))
        Sz_i_Sz_j = np.kron(site_op(i, Sz), site_op(j, Sz))
        A += Sx_i_Sx_j + Sy_i_Sy_j + delta * Sz_i_Sz_j
    
    # Add half the uniform field to A
    for k in range(N):
        A -= 0.5 * h * site_op(k, Sz)
    
    # Build B (even bonds + half field)
    B = zero.copy()
    for (i, j) in bonds_even:
        Sx_i_Sx_j = np.kron(site_op(i, Sx), site_op(j, Sx))
        Sy_i_Sy_j = np.kron(site_op(i, Sy), site_op(j, Sy))
        Sz_i_Sz_j = np.kron(site_op(i, Sz), site_op(j, Sz))
        B += Sx_i_Sx_j + Sy_i_Sy_j + delta * Sz_i_Sz_j
    
    # Add half the uniform field to B
    for k in range(N):
        B -= 0.5 * h * site_op(k, Sz)
    
    return A, B


def _spectral_norm_error(A, B, a, b, t_values=(0.25, 0.5, 1.0)):
    """Compute mean spectral norm error over the given time values."""
    U_exact_full = A + B
    errors = []
    for t in t_values:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            # First apply exp(-i a_k t A), then exp(-i b_k t B)
            # Wait, the task says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
            # This means: U = exp(-i b_12 t B) exp(-i a_12 t A) ... exp(-i b_1 t B) exp(-i a_1 t A)
            # No wait, "in order k=1..12" means the product is applied left to right as k goes 1 to 12
            # So U = (exp(-i b_12 t B) exp(-i a_12 t A)) * ... * (exp(-i b_1 t B) exp(-i a_1 t A))
            # Actually in matrix multiplication, the rightmost factor is applied first.
            # Let me re-read: "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
            # This typically means: U = [exp(-i b_12 t B) exp(-i a_12 t A)] ... [exp(-i b_1 t B) exp(-i a_1 t A)]
            # where the product goes from k=1 to k=12, so the k=1 term is rightmost.
            
            # Let's build it by applying factors from k=1 to k=12, with each new factor on the left
            pass
        
        # Let me redo this more carefully
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a) - 1, -1, -1):
            # Apply exp(-i b_k t B) then exp(-i a_k t A)
            # In the product, for each k: exp(-i b_k t B) exp(-i a_k t A)
            # The full product for k=1 to 12 is: F_12 F_11 ... F_1 where F_k = exp(-i b_k t B) exp(-i a_k t A)
            # So U = F_12 * F_11 * ... * F_1
            # To compute: start with U = I, then U = F_1 * U, then U = F_2 * U, ..., U = F_12 * U
            F = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A)
            U = F @ U
        
        U_exact = expm(-1j * t * U_exact_full)
        diff = U - U_exact
        # Spectral norm = largest singular value
        norm_val = np.linalg.norm(diff, 2)
        errors.append(norm_val)
    
    return np.mean(errors)


def coefficients(stages):
    """
    Find coefficients for a 12-stage product formula approximating time evolution.
    
    Uses optimization to minimize mean spectral norm error at t = 0.25, 0.5, 1.0.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    A, B = _build_hamiltonians()
    
    # Initial guess: use 4th-order Suzuki-Trotter coefficients replicated
    # 4th order: c1 = 1/(2*(1-2^(1/3))), c2 = 2^(1/3)/(2*(1-2^(1/3)))
    # The 4th order symmetric scheme for exp(A+B) is:
    # exp(c1*A) exp(c1*B) exp(c2*A) exp(c2*B) exp(c1*A) exp(c1*B)
    # where c1 = 1/(2*(1-2^(1/3))), c2 = 1 - 2*c1
    # But our product is exp(-i b_k t B) exp(-i a_k t A), so we need to match this form.
    
    # Let's just use a simple initial guess and optimize
    # Symmetric initialization
    n = stages
    a = [1.0/n] * n
    b = [1.0/n] * n
    
    # Ensure sums are exactly 1
    a_sum = sum(a)
    b_sum = sum(b)
    a = [x / a_sum for x in a]
    b = [x / b_sum for x in b]
    
    def objective(params):
        """Compute objective from flattened parameters."""
        a_params = params[:n]
        b_params = params[n:2*n]
        
        # Enforce sum constraints by projecting
        a_sum = np.sum(a_params)
        b_sum = np.sum(b_params)
        
        if abs(a_sum) < 1e-15:
            return 1e10
        if abs(b_sum) < 1e-15:
            return 1e10
        
        a_corrected = a_params / a_sum
        b_corrected = b_params / b_sum
        
        # Clip to valid range
        a_corrected = np.clip(a_corrected, -50, 50)
        b_corrected = np.clip(b_corrected, -50, 50)
        
        # Re-normalize after clipping
        a_sum2 = np.sum(a_corrected)
        b_sum2 = np.sum(b_corrected)
        if abs(a_sum2) < 1e-15 or abs(b_sum2) < 1e-15:
            return 1e10
        a_corrected = a_corrected / a_sum2
        b_corrected = b_corrected / b_sum2
        
        return _spectral_norm_error(A, B, a_corrected.tolist(), b_corrected.tolist())
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    
    x0 = np.array(a + b, dtype=float)
    
    best_obj = objective(x0)
    best_x = x0.copy()
    
    # Run optimization with time budget
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10})
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        
        # If we have more time, try another optimization from a different starting point
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            # Try a perturbed starting point
            rng = np.random.RandomState(42)
            for trial in range(5):
                elapsed = time.time() - start_time
                if elapsed > deadline - 20:
                    break
                
                x0_trial = best_x + rng.normal(0, 0.1, size=2*n)
                x0_trial[:n] /= np.sum(x0_trial[:n]) if abs(np.sum(x0_trial[:n])) > 1e-15 else 1
                x0_trial[n:] /= np.sum(x0_trial[n:]) if abs(np.sum(x0_trial[n:])) > 1e-15 else 1
                
                result_trial = minimize(objective, x0_trial, method='Nelder-Mead',
                                       options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-10})
                
                if result_trial.fun < best_obj:
                    best_obj = result_trial.fun
                    best_x = result_trial.x
                    
    except Exception:
        pass
    
    # Extract final coefficients
    a_final = best_x[:n].tolist()
    b_final = best_x[n:].tolist()
    
    # Ensure sums are exactly 1
    a_sum = sum(a_final)
    b_sum = sum(b_final)
    if abs(a_sum) > 1e-15:
        a_final = [x / a_sum for x in a_final]
    if abs(b_sum) > 1e-15:
        b_final = [x / b_sum for x in b_final]
    
    # Clip to [-50, 50]
    a_final = [max(-50, min(50, x)) for x in a_final]
    b_final = [max(-50, min(50, x)) for x in b_final]
    
    # Final sum correction
    a_sum = sum(a_final)
    b_sum = sum(b_final)
    if abs(a_sum) > 1e-15:
        a_final = [x / a_sum for x in a_final]
    if abs(b_sum) > 1e-15:
        b_final = [x / b_sum for x in b_final]
    
    return a_final, b_final
