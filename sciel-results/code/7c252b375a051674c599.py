import numpy as np
import time
import scipy.linalg
import scipy.optimize

# Constants
DELTA = 0.8
FIELD = 0.3
N_SITES = 6
DIM = 2 ** N_SITES
TIMES = [0.25, 0.5, 1.0]

def build_matrices():
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    def kron_chain(ops):
        """
        ops: list of 2x2 matrices, one for each site.
        Returns the Kronecker product.
        """
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Build H_odd and H_even
    # Bonds: 1-2, 2-3, 3-4, 4-5, 5-6 (0-indexed: 0-1, 1-2, 2-3, 3-4, 4-5)
    # Odd bonds (1st, 3rd, 5th): 0-1, 2-3, 4-5
    # Even bonds (2nd, 4th): 1-2, 3-4
    
    H_odd = np.zeros((DIM, DIM), dtype=complex)
    H_even = np.zeros((DIM, DIM), dtype=complex)
    H_field = np.zeros((DIM, DIM), dtype=complex)
    
    for i in range(N_SITES - 1):
        # Construct operators for bond i, i+1
        ops_x = [id2] * N_SITES
        ops_y = [id2] * N_SITES
        ops_z = [id2] * N_SITES
        
        ops_x[i] = sx
        ops_x[i+1] = sx
        ops_y[i] = sy
        ops_y[i+1] = sy
        ops_z[i] = sz
        ops_z[i+1] = sz
        
        xx = kron_chain(ops_x)
        yy = kron_chain(ops_y)
        zz = kron_chain(ops_z)
        
        bond_h = 0.5 * (xx + yy) + DELTA * zz
        
        if i % 2 == 0: # 0, 2, 4 -> Odd bonds (1st, 3rd, 5th)
            H_odd += bond_h
        else: # 1, 3 -> Even bonds (2nd, 4th)
            H_even += bond_h
            
    # Field term
    for i in range(N_SITES):
        ops_z = [id2] * N_SITES
        ops_z[i] = sz
        H_field += kron_chain(ops_z)
        
    A = H_odd + 0.5 * FIELD * H_field
    B = H_even + 0.5 * FIELD * H_field
    
    return A, B

def compute_error(a, b, A, B, exact_propagators):
    """
    a, b: lists of 12 coefficients.
    A, B: matrices.
    exact_propagators: dict {t: U_exact(t)}
    """
    total_err = 0.0
    for t in TIMES:
        U_approx = np.eye(DIM, dtype=complex)
        # Order k=1..12: exp(-i b_k t B) exp(-i a_k t A)
        for k in range(12):
            # We multiply from right to left?
            # U(t) = prod_{k=1}^{12} E_k
            # E_k = exp(-i b_k t B) exp(-i a_k t A)
            # U = E_1 E_2 ... E_12
            # So U = U @ E_k
            U_approx = U_approx @ scipy.linalg.expm(-1j * b[k] * t * B) @ scipy.linalg.expm(-1j * a[k] * t * A)
        
        U_exact = exact_propagators[t]
        diff = U_approx - U_exact
        err = np.linalg.norm(diff, 2)
        total_err += err
        
    return total_err / len(TIMES)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    if stages != 12:
        # Fallback to Strang if stages != 12
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    A, B = build_matrices()
    
    # Precompute exact propagators
    H_total = A + B
    exact_propagators = {}
    for t in TIMES:
        exact_propagators[t] = scipy.linalg.expm(-1j * t * H_total)
    
    # Initial guess: 4th order Suzuki-Trotter coefficients mapped to 12 steps
    # Standard 4th order: c1 = 1/(2 - 2^(1/3)), c0 = -c1 / (2^(1/3) - 1)
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c0 = -c1 / (2.0**(1.0/3.0) - 1.0)
    
    # We want to distribute these into 12 steps.
    # A symmetric 4th order scheme S4(t) = S2(c1 t) S2(c0 t) S2(c1 t)
    # S2(tau) = exp(-i tau B/2) exp(-i tau A) exp(-i tau B/2)
    # Sequence of A and B exponents in S4:
    # B(c1/2) A(c1) B(c1/2) B(c0/2) A(c0) B(c0/2) B(c1/2) A(c1) B(c1/2)
    # Merged B's:
    # B(c1/2) A(c1) B((c1+c0)/2) A(c0) B((c0+c1)/2) A(c1) B(c1/2)
    # This has 3 A's and 4 B's (effectively, if we count non-zero).
    # We have 12 A slots and 12 B slots.
    # We can use the optimizer to find the best distribution.
    # Let's start with a symmetric distribution of the 4th order weights.
    # We'll define 6 independent a's and 6 independent b's due to symmetry a_k = a_{13-k}, b_k = b_{13-k}.
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Initial guess based on equal split (Strang) is safe.
    # a_init = [1/12] * 12
    # b_init = [1/12] * 12
    # But 4th order is better.
    # Let's try to map the 4th order coefficients to the first 6 slots and mirror.
    # Total sum of a's must be 1.
    # Total sum of b's must be 1.
    
    # Let's use a simpler initial guess: Strang splitting (2nd order)
    a_init_half = [1.0/12.0] * 6
    b_init_half = [1.0/12.0] * 6
    
    # Parameters to optimize: a1..a5, b1..b5. a6 and b6 are determined by sum constraint.
    # a6 = 1 - sum(a1..a5)
    # b6 = 1 - sum(b1..b5)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = params
        a6 = 1.0 - (a1 + a2 + a3 + a4 + a5)
        b6 = 1.0 - (b1 + b2 + b3 + b4 + b5)
        
        a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        # Check for non-finite or out of bounds? 
        # The problem says coefficients in [-50, 50].
        # We should penalize if outside.
        if any(abs(x) > 50 for x in a + b):
            return 1e10
            
        return compute_error(a, b, A, B, exact_propagators)
        
    x0 = [1.0/12.0]*5 + [1.0/12.0]*5
    
    # Use Nelder-Mead for robustness, or L-BFGS-B if we add bounds.
    # Let's try L-BFGS-B with bounds [-50, 50] for each param.
    bounds = [(-50, 50)] * 10
    
    try:
        res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        # Fallback to Strang
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b
        
    if res.success:
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = res.x
        a6 = 1.0 - (a1 + a2 + a3 + a4 + a5)
        b6 = 1.0 - (b1 + b2 + b3 + b4 + b5)
        
        a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        # Final sanity check
        if abs(sum(a) - 1.0) > 1e-9 or abs(sum(b) - 1.0) > 1e-9:
            a = [1.0/12.0] * 12
            b = [1.0/12.0] * 12
        if any(abs(x) > 50 for x in a + b):
            a = [1.0/12.0] * 12
            b = [1.0/12.0] * 12
        else:
            return a, b
    else:
        # Fallback
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b

# EVOLVE-BLOCK-END
