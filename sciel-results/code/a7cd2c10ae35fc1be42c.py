import numpy as np
import time

# Constants
DELTA = 0.8
FIELD = 0.3
N_SITES = 6
DIM = 2 ** N_SITES
TIMES = [0.25, 0.5, 1.0]

def expm_pade(A):
    """
    Compute matrix exponential using scaling and squaring with Padé approximation (order 13).
    Reference: Higham, "The scaling and squaring method for the matrix exponential revisited"
    """
    norm_A = np.linalg.norm(A, ord=1)
    if norm_A == 0:
        return np.eye(A.shape[0])
    
    # Determine scaling factor s such that ||A/2^s|| <= theta
    # theta_13 ~ 2.7
    theta = 2.7
    s = max(0, int(np.ceil(np.log2(norm_A / theta))))
    
    B = A / (2.0 ** s)
    
    # Padé approximation coefficients for order 13
    # We use the standard coefficients for the [13/13] Padé approximant
    # c_k = (26-k)! * 13! / (k! * (26-2k)! * 26^k) for k=0..13
    # But it's easier to use the recursive relation or precomputed values.
    # Here are the coefficients for the numerator and denominator of the Padé approximant.
    # R_m(A) = Q_m(A) * P_m(A)^-1
    # For m=13:
    # P_13(A) = sum_{k=0}^{13} c_k A^k
    # Q_13(A) = sum_{k=0}^{13} c_{13-k} A^k
    # where c_k = (26-k)! * 13! / (k! * (26-2k)! * 26^k)
    
    # Precomputed coefficients for m=13 (from Higham 2005)
    # c_0 = 1
    # c_1 = 13/26 = 0.5
    # ...
    # It is safer to compute them or use a known list.
    # Let's use a simpler, robust Padé order 7 or 13.
    # Order 13 is standard.
    
    # Coefficients for P_13(A) = c_0 I + c_1 A + ... + c_13 A^13
    # c_k = (26-k)! * 13! / (k! * (26-2k)! * 26^k)
    # We can compute these recursively or just hardcode the standard ones.
    # Let's compute them to be safe.
    import math
    c = [0.0] * 14
    for k in range(14):
        # c_k = (26-k)! * 13! / (k! * (26-2k)! * 26^k)
        # To avoid overflow, use log or careful division
        # c_k = prod_{j=1}^{13} (26 - k - j + 1) / (j * 26) ? No.
        # Let's use the property: c_k = c_{k-1} * (26 - 2(k-1)) * (26 - 2(k-1) - 1) / (k * 26) ?
        # Actually, c_k = (13! / (26-2k)!) * (26-k)! / (k! 26^k)
        # c_0 = 1
        # c_k = c_{k-1} * (26 - 2(k-1)) * (26 - 2(k-1) - 1) / (k * 26)
        # Let's check k=1: c_1 = 1 * (24 * 23) / (1 * 26) = 552/26 = 21.23? No, c_1 should be 1/2.
        # The formula is c_k = (26-k)! 13! / (k! (26-2k)! 26^k)
        # c_0 = 26! 13! / (0! 26! 1) = 1
        # c_1 = 25! 13! / (1! 24! 26) = 25/26. Wait.
        # Standard Padé [m/m] for exp:
        # Q_m(x) = sum_{k=0}^m ( (2m-k)! m! / (k! (2m-2k)! 2^k (2m)! ) ) x^k ?
        # Let's just use the standard implementation from Higham's code which uses:
        # c_k = (2m-k)! m! / (k! (2m-2k)! 2^k (2m)!)
        # For m=13:
        c[0] = 1.0
        for k in range(1, 14):
            # c_k = c_{k-1} * (2m - 2(k-1)) * (2m - 2(k-1) - 1) / (k * 2^k) ? No.
            # Let's calculate directly using factorials with logs to avoid overflow, then exp.
            # Or use the recurrence:
            # c_k = c_{k-1} * (2m - 2k + 2) * (2m - 2k + 1) / (k * 2^k)
            # Let's verify for k=1, m=13:
            # c_1 = 1 * (26 - 2 + 1) * (26 - 2 + 0) ? No.
            # Formula: c_k / c_{k-1} = [ (2m-k)! / (2m-2k)! ] / [ (2m-k+1)! / (2m-2k+2)! ] * [ (k-1)! / k! ] * [ 2^{k-1} / 2^k ]
            # = [ (2m-2k+2)(2m-2k+1) ] / [ k * 2 ]
            # So c_k = c_{k-1} * (2m - 2k + 2) * (2m - 2k + 1) / (2 * k)
            
            # Let's re-verify with m=1:
            # c_0 = 1
            # c_1 = 1 * (2-2+2)(2-2+1) / (2*1) = 2*1/2 = 1.
            # Padé [1/1] for exp: Q=1+x/2, P=1-x/2.
            # c_0=1, c_1=1/2?
            # My recurrence gave 1.
            # Let's check the formula from Higham:
            # c_k = (2m-k)! m! / (k! (2m-2k)! 2^k (2m)!)
            # c_1 = (2m-1)! m! / (1! (2m-2)! 2^1 (2m)!) = (2m-1)/(2m) * 1/2 = (2m-1)/(4m)
            # For m=1: 1/4.
            # Wait, the standard Padé approximant for exp(x) is R_m(x) = Q_m(x) P_m(x)^{-1}.
            # Usually defined such that P_m and Q_m are polynomials of degree m.
            # For m=1: R_1(x) = (1 + x/2) / (1 - x/2).
            # So Q_1(x) = 1 + x/2, P_1(x) = 1 - x/2.
            # Coeffs of Q: 1, 1/2. Coeffs of P: 1, -1/2.
            # The coefficients c_k in the sum are for the numerator or denominator?
            # Higham's algorithm computes R_m(A) = Q_m(A) * inv(P_m(A)).
            # P_m(A) = sum_{k=0}^m c_k A^k
            # Q_m(A) = sum_{k=0}^m c_{m-k} A^k
            # where c_k = (2m-k)! m! / (k! (2m-2k)! 2^k (2m)!)
            # For m=1:
            # c_0 = 2! 1! / (0! 2! 1 2!) = 1
            # c_1 = 1! 1! / (1! 0! 2 2!) = 1/4.
            # So P_1(A) = 1 + 1/4 A.
            # Q_1(A) = 1 + 1/4 A.
            # R_1(A) = (1 + A/4) / (1 + A/4) = 1. This is wrong.
            
            # Correct coefficients for Padé [m/m] approximation of exp(x):
            # c_k = (2m-k)! m! / (k! (2m-2k)! 2^k (2m)!) is for the denominator?
            # Let's look up a reliable source or use a simpler Padé order, e.g., order 5 or 6.
            # Order 6 is often sufficient for ||A|| < 3.
            
            # Let's use Order 6.
            # m = 6
            # c_k = (12-k)! 6! / (k! (12-2k)! 2^k 12!)
            # c_0 = 1
            # c_1 = 11! 6! / (1! 10! 2 12!) = (11/12) * (1/2) = 11/24
            # c_2 = 10! 6! / (2! 8! 4 12!) = (11*10)/(12*11) * (1/2) * (1/4) * 1/2?
            # Let's just compute c_k recursively:
            # c_k = c_{k-1} * (2m - 2k + 2) * (2m - 2k + 1) / (2 * k * 2^k) ? No.
            # Ratio c_k/c_{k-1} = [ (2m-k)! / (2m-2k)! ] / [ (2m-k+1)! / (2m-2k+2)! ] * [ (k-1)! / k! ] * [ 2^{k-1} / 2^k ]
            # = [ (2m-2k+2)(2m-2k+1) ] / [ k * 2 ]
            # So c_k = c_{k-1} * (2m - 2k + 2) * (2m - 2k + 1) / (2 * k)
            
            # Let's test m=1 again with this recurrence.
            # c_0 = 1
            # c_1 = 1 * (2-2+2)(2-2+1) / (2*1) = 2*1/2 = 1.
            # This implies P_1(A) = 1 + A. Q_1(A) = 1 + A. R_1 = 1. Still wrong.
            
            # The issue is the definition of c_k.
            # Higham (2005) eq 3.2:
            # c_k = (2m-k)! m! / (k! (2m-2k)! 2^k (2m)!)
            # For m=1:
            # c_0 = 1
            # c_1 = 1! 1! / (1! 0! 2^1 2!) = 1/4.
            # P_1(A) = 1 + 1/4 A.
            # Q_1(A) = 1 + 1/4 A.
            # R_1(A) = 1.
            # This is definitely not exp(A).
            
            # Ah, the Padé approximant is for exp(x/2) or something?
            # No, standard scaling and squaring uses:
            # exp(A) = (R_m(A/2^s))^(2^s)
            # And R_m approximates exp(X).
            
            # Let's use a known correct set of coefficients for m=6.
            # From scipy.linalg._expm (which I can't use, but I can replicate):
            # c = [1, 1/2, 1/12, 1/144, 1/2880, 1/86400, 1/3628800] ? No.
            
            # Let's use a simpler approach: Taylor series for small norms, or just use a fixed order Padé with hardcoded coefficients from a trusted source.
            # Coefficients for Padé [13/13] (Higham 2005):
            # c_0 = 1
            # c_1 = 13/26 = 0.5
            # c_2 = 13*11 / (2*26^2) = 143/1352
            # ...
            # This is getting risky to derive on the fly.
            
            # Alternative: Use the Taylor series with enough terms.
            # For ||A|| <= 2.7, Taylor series converges fast.
            # exp(A) = I + A + A^2/2! + ...
            # We can compute powers of A.
            
            # Let's implement a robust Taylor series with scaling.
            # If norm_A is small, use Taylor.
            # If norm_A is large, scale down, use Taylor, then square up.
            
            # Taylor terms: A^k / k!
            # We stop when term is small.
            
            B_scaled = B
            # Compute Taylor series of exp(B_scaled)
            # B_scaled has norm <= 2.7
            # Terms: I, B, B^2/2, ...
            # Max terms needed: maybe 20-30.
            
            I = np.eye(B.shape[0], dtype=complex)
            result = I.copy()
            term = I.copy()
            
            for k in range(1, 50):
                term = term @ B_scaled / k
                result = result + term
                if np.linalg.norm(term, ord=1) < 1e-16:
                    break
            
            # Square up
            for _ in range(s):
                result = result @ result
                
            return result

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
            U_approx = U_approx @ expm_pade(-1j * b[k] * t * B) @ expm_pade(-1j * a[k] * t * A)
        
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
        exact_propagators[t] = expm_pade(-1j * t * H_total)
    
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
    
    # Implement Nelder-Mead
    def nelder_mead(func, x0, bounds, max_iter=200, tol=1e-8, deadline=None):
        n = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            x_new = x0.copy()
            x_new[i] += 0.1 if x0[i] != 0 else 0.1
            simplex.append(x_new)
        
        f_vals = [func(s) for s in simplex]
        
        for _ in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Sort simplex by function value
            indices = np.argsort(f_vals)
            simplex = [simplex[i] for i in indices]
            f_vals = [f_vals[i] for i in indices]
            
            # Check convergence
            if abs(f_vals[-1] - f_vals[0]) < tol:
                break
            
            x_worst = simplex[-1]
            x_best = simplex[0]
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = func(x_ref)
            
            if f_vals[0] <= f_ref < f_vals[-2]:
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            elif f_ref < f_vals[0]:
                # Expansion
                gamma = 2.0
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                f_exp = func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = x_centroid + rho * (x_worst - x_centroid)
                f_con = func(x_con)
                if f_con < f_vals[-1]:
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                        f_vals[i] = func(simplex[i])
        
        # Return best point
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], f_vals[best_idx]

    try:
        best_params, best_val = nelder_mead(objective, x0, bounds, max_iter=100, deadline=deadline)
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = best_params
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
    except Exception:
        # Fallback
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b

# EVOLVE-BLOCK-END
