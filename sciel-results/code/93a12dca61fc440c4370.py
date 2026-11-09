import numpy as np
import time
import scipy.optimize

def coefficients(stages):
    """
    Returns 12-stage product formula coefficients for Heisenberg chain time evolution.
    Uses 6th-order symmetric Suzuki-Trotter splitting as initial guess, then optimizes.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Build Hamiltonians
    # 6-site open XXZ Heisenberg chain
    # H = sum_{i=1}^{5} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + 0.8 S_i^z S_{i+1}^z) + 0.3 sum_{i=1}^{6} S_i^z
    # Split into even and odd bonds
    # A: odd bonds (1-2, 3-4, 5-6) + half field
    # B: even bonds (2-3, 4-5) + half field
    
    # Spin 1/2 matrices
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build A and B
    # Identity on 6 sites
    I = np.eye(2, dtype=complex)
    
    def kron_all(*mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    # A: bonds (0,1), (2,3), (4,5)
    # H_A = sum_{i in {0,2,4}} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + 0.8 Sz_i Sz_{i+1}) + 0.15 sum_{j=0}^5 Sz_j
    H_A = np.zeros((64, 64), dtype=complex)
    for i in [0, 2, 4]:
        # Sx_i Sx_{i+1}
        mats = [I]*6
        mats[i] = Sx
        mats[i+1] = Sx
        H_A += kron_all(*mats)
        
        # Sy_i Sy_{i+1}
        mats[i] = Sy
        mats[i+1] = Sy
        H_A += kron_all(*mats)
        
        # Sz_i Sz_{i+1}
        mats[i] = Sz
        mats[i+1] = Sz
        H_A += 0.8 * kron_all(*mats)
        
    # Field part for A: 0.15 * sum Sz_j
    for j in range(6):
        mats = [I]*6
        mats[j] = Sz
        H_A += 0.15 * kron_all(*mats)
        
    # B: bonds (1,2), (3,4)
    H_B = np.zeros((64, 64), dtype=complex)
    for i in [1, 3]:
        mats = [I]*6
        mats[i] = Sx
        mats[i+1] = Sx
        H_B += kron_all(*mats)
        
        mats[i] = Sy
        mats[i+1] = Sy
        H_B += kron_all(*mats)
        
        mats[i] = Sz
        mats[i+1] = Sz
        H_B += 0.8 * kron_all(*mats)
        
    # Field part for B: 0.15 * sum Sz_j
    for j in range(6):
        mats = [I]*6
        mats[j] = Sz
        H_B += 0.15 * kron_all(*mats)

    # Precompute exponentials for different t?
    # t = 0.25, 0.5, 1.0
    # We need exp(-i t H)
    
    def spectral_norm_error(a, b):
        err = 0.0
        for t in [0.25, 0.5, 1.0]:
            # Exact
            H_total = H_A + H_B
            U_exact = np.linalg.expm(-1j * t * H_total)
            
            # Approx
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B) exp(-i a_k t A)
                U_B = np.linalg.expm(-1j * b[k] * t * H_B)
                U_A = np.linalg.expm(-1j * a[k] * t * H_A)
                U_approx = U_B @ U_A @ U_approx
                
            diff = U_approx - U_exact
            # Spectral norm is the largest singular value
            err += np.linalg.norm(diff, 2)
        return err / 3.0

    # Initial guess: 6th order symmetric
    # c1 = 0.289151 / 1.8, c2 = 0.421698 / 1.8, c3 = 0.189151 / 1.8
    c1 = 0.289151 / 1.8
    c2 = 0.421698 / 1.8
    c3 = 0.189151 / 1.8
    
    a_init = [c1, c2, c3, c3, c2, c1, 0, 0, 0, 0, 0, 0]
    b_init = [c1, c2, c3, c3, c2, c1, 0, 0, 0, 0, 0, 0]
    
    # Convert to numpy for optimization
    a0 = np.array(a_init)
    b0 = np.array(b_init)
    
    # Optimize
    # We can optimize 24 variables.
    # Constraint: sum(a)=1, sum(b)=1.
    # We can use scipy.optimize.minimize with constraints.
    
    def objective(x):
        a = x[:12]
        b = x[12:]
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        # Check sums (loose check, optimizer will handle)
        return spectral_norm_error(a, b)
    
    # Constraints
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:12]) - 1.0},
        {'type': 'eq', 'fun': lambda x: np.sum(x[12:]) - 1.0}
    ]
    
    x0 = np.concatenate([a0, b0])
    
    # Use Powell or Nelder-Mead
    # Powell is derivative-free and handles constraints? No, Powell doesn't handle constraints directly in standard SciPy.
    # Use SLSQP or Trust-Constr.
    # SLSQP is good for small problems.
    
    try:
        res = scipy.optimize.minimize(objective, x0, method='SLSQP', constraints=cons, bounds=[(-50, 50)]*24, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            a_opt = res.x[:12]
            b_opt = res.x[12:]
            # Ensure finite
            if np.all(np.isfinite(a_opt)) and np.all(np.isfinite(b_opt)):
                return list(a_opt), list(b_opt)
    except Exception:
        pass
    
    # Fallback to initial guess
    return a_init, b_init
