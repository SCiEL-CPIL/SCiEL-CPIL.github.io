import numpy as np
import time
import sys
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def get_schoolbook():
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Attempt Rank 34
    R = 34
    if time.time() > deadline - 10:
        return get_schoolbook()

    # Pre-compute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    num_targets = len(target_indices)

    # Residual function for optimizer
    # x is a flat array of size R * (n*m + m*p + p*n)
    # Split into U, V, W
    def residual(x):
        U_flat = x[:R * n * m]
        V_flat = x[R * n * m : R * n * m + R * m * p]
        W_flat = x[R * n * m + R * m * p :]
        
        U = U_flat.reshape(R, n * m)
        V = V_flat.reshape(R, m * p)
        W = W_flat.reshape(R, p * n)
        
        res = np.zeros(num_targets)
        for idx, (a, b, c) in enumerate(target_indices):
            val = 0.0
            for r in range(R):
                val += U[r, a] * V[r, b] * W[r, c]
            res[idx] = val - 1.0
        return res

    # Initial Guess: Structured
    # We use a heuristic initialization based on the block structure 
    # to give the optimizer a head start, assuming the block decomposition is valid.
    x0 = np.zeros(R * (n * m + m * p + p * n))
    
    # We'll try to map the known valid terms from a "loose" block decomposition
    # into the first few slots, and leave the rest random/zero to allow optimization.
    
    # For now, let's just use a random initialization with small noise, 
    # or better, use the schoolbook terms as a base if we can map them?
    # Schoolbook has 36 terms. We need 34.
    # Let's try a random initialization scaled to 1.0
    np.random.seed(42)
    # Scale initialization to avoid huge gradients
    x0 = np.random.randn(R * (n * m + m * p + p * n)) * 0.5
    
    # Optimization
    try:
        # Use 'lm' (Levenberg-Marquardt) if number of variables >= number of residuals
        # Here: Vars = 34 * (9 + 12 + 12) = 34 * 33 = 1122
        # Residuals = 36
        # So we are underdetermined. LM is good for underdetermined? 
        # Actually, 'lm' requires M >= N (residuals >= vars) usually? 
        # No, lm works for M >= N. If M < N, it might fail or need 'trf'.
        # Let's use 'trf' (Trust Region Reflective) which handles bounds and general cases.
        
        # We want to find x such that residual(x) = 0.
        # least_squares minimizes sum(residuals^2).
        
        res = opt.least_squares(residual, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        
        if time.time() > deadline - 5:
            return get_schoolbook()

        if res.success or res.cost < 1e-10:
            # Extract optimal U, V, W
            U_opt = res.x[:R * n * m].reshape(R, n * m)
            V_opt = res.x[R * n * m : R * n * m + R * m * p].reshape(R, m * p)
            W_opt = res.x[R * n * m + R * m * p :].reshape(R, p * n)
            
            # Try to recover exact rationals (integers or half-integers)
            # Check if values are close to k/2
            def snap_to_half_int(val):
                nearest = np.round(val * 2) / 2
                if abs(val - nearest) < 1e-6:
                    return Fraction(int(nearest * 2), 2)
                return None

            U_rat = []
            V_rat = []
            W_rat = []
            success = True
            
            for r in range(R):
                u_row = []
                v_row = []
                w_row = []
                for val in U_opt[r]:
                    frac = snap_to_half_int(val)
                    if frac is None:
                        success = False
                        break
                    u_row.append(frac)
                if not success: break
                for val in V_opt[r]:
                    frac = snap_to_half_int(val)
                    if frac is None:
                        success = False
                        break
                    v_row.append(frac)
                if not success: break
                for val in W_opt[r]:
                    frac = snap_to_half_int(val)
                    if frac is None:
                        success = False
                        break
                    w_row.append(frac)
                if not success: break
                
                U_rat.append(u_row)
                V_rat.append(v_row)
                W_rat.append(w_row)
                
            if success:
                # Verify exact decomposition
                valid = True
                for idx, (a, b, c) in enumerate(target_indices):
                    val = Fraction(0)
                    for r in range(R):
                        val += U_rat[r][a] * V_rat[r][b] * W_rat[r][c]
                    if val != 1:
                        valid = False
                        break
                
                if valid:
                    # Convert to floats for output
                    U_out = [[float(v) for v in row] for row in U_rat]
                    V_out = [[float(v) for v in row] for row in V_rat]
                    W_out = [[float(v) for v in row] for row in W_rat]
                    return {"U": U_out, "V": V_out, "W": W_out}
            
        # If optimization failed to find exact solution, fallback
        return get_schoolbook()
        
    except Exception:
        return get_schoolbook()
