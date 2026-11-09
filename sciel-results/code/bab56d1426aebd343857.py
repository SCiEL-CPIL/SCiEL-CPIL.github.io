"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    import time
    from fractions import Fraction

    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8

    start_time = time.time()
    deadline = start_time + 140.0

    def verify_decomposition(U, V, W):
        """Verify that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all i,j,k."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    def to_int_or_half(matrix):
        """Convert Fraction matrix to int or float (half-integer) representation."""
        result = []
        for row in matrix:
            new_row = []
            for x in row:
                if x.denominator == 1:
                    new_row.append(int(x))
                elif x.denominator == 2:
                    new_row.append(float(x))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    def schoolbook():
        rank = 24
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return U, V, W

    # Construct rank-22 decomposition using Strassen blocks
    def strassen_2x2_2x2(j_offset, k_offset):
        """
        Return 7 (U, V, W) factor lists for 2x2*2x2 multiplication
        where A indices are i in {0,1}, j in {j_offset, j_offset+1}
        and B indices are j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        and C indices are i in {0,1}, k in {k_offset, k_offset+1}
        """
        factors = []
        # Each factor: (U_dict, V_dict, W_dict)
        # U_dict: {(i, j): coeff}, V_dict: {(j, k): coeff}, W_dict: {(i, k): coeff}
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # Contributes: C[0,0]+=1, C[0,1]+=1, C[1,0]+=-1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1, (1, j_offset+1): 1},
            'V': {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): 1, (0, k_offset+1): 1, (1, k_offset): -1, (1, k_offset+1): 1}
        })
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # Contributes: C[1,0]+=1, C[1,1]+=-1
        factors.append({
            'U': {(1, j_offset): 1, (1, j_offset+1): 1},
            'V': {(j_offset, k_offset): 1},
            'W': {(1, k_offset): 1, (1, k_offset+1): -1}
        })
        
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # Contributes: C[0,1]+=1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1},
            'V': {(j_offset, k_offset+1): 1, (j_offset+1, k_offset+1): -1},
            'W': {(0, k_offset+1): 1, (1, k_offset+1): 1}
        })
        
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # Contributes: C[0,0]+=1, C[1,0]+=1
        factors.append({
            'U': {(1, j_offset+1): 1},
            'V': {(j_offset+1, k_offset): 1, (j_offset, k_offset): -1},
            'W': {(0, k_offset): 1, (1, k_offset): 1}
        })
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # Contributes: C[0,0]+=-1, C[1,1]+=1
        factors.append({
            'U': {(0, j_offset): 1, (0, j_offset+1): 1},
            'V': {(j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): -1, (1, k_offset+1): 1}
        })
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # Contributes: C[1,1]+=1
        factors.append({
            'U': {(1, j_offset): 1, (0, j_offset): -1},
            'V': {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1},
            'W': {(1, k_offset+1): 1}
        })
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # Contributes: C[0,0]+=1, C[1,0]+=1
        factors.append({
            'U': {(0, j_offset+1): 1, (1, j_offset+1): -1},
            'V': {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1},
            'W': {(0, k_offset): 1, (1, k_offset): 1}
        })
        
        return factors
    
    # Build the full decomposition
    all_factors = []
    
    # Strassen for j in {0,1}, k in {0,1}
    all_factors.extend(strassen_2x2_2x2(0, 0))
    
    # Strassen for j in {0,1}, k in {2,3}
    all_factors.extend(strassen_2x2_2x2(0, 2))
    
    # Add 8 factors for j=2, one per (i,k)
    for i in range(2):
        for k in range(4):
            all_factors.append({
                'U': {(i, 2): 1},
                'V': {(2, k): 1},
                'W': {(i, k): 1}
            })
    
    # Convert factors to U, V, W matrices
    def factors_to_matrices(factors):
        rank = len(factors)
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for r, f in enumerate(factors):
            for (i, j), coeff in f['U'].items():
                U[i*m + j][r] += Fraction(coeff)
            for (j, k), coeff in f['V'].items():
                V[j*p + k][r] += Fraction(coeff)
            for (i, k), coeff in f['W'].items():
                W[k*n + i][r] += Fraction(coeff)
        return U, V, W
    
    U22, V22, W22 = factors_to_matrices(all_factors)
    
    if verify_decomposition(U22, V22, W22):
        # We have a valid rank-22 decomposition. Try to reduce to 21.
        
        # Strategy: Try to merge two columns by checking if one is a linear combination
        # that allows elimination. Use optimization to find if rank 21 is possible.
        
        # Convert to numpy for optimization
        U_np = np.array([[float(x) for x in row] for row in U22], dtype=float)
        V_np = np.array([[float(x) for x in row] for row in V22], dtype=float)
        W_np = np.array([[float(x) for x in row] for row in W22], dtype=float)
        
        # Try dropping each column and optimizing
        r_target = 21
        best_err = float('inf')
        best_params = None
        
        def bilinear_error(U, V, W):
            """Compute squared error of bilinear verification."""
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            r = U.shape[1]
            total_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        total_err += (actual - 1.0) ** 2
            return total_err
        
        from scipy.optimize import minimize
        
        for drop_col in range(22):
            if time.time() > deadline - 20:
                break
            
            keep = [c for c in range(22) if c != drop_col]
            U_init = U_np[:, keep]
            V_init = V_np[:, keep]
            W_init = W_np[:, keep]
            
            # Flatten for optimization
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:nm * r_target].reshape(nm, r_target)
                V = x[nm * r_target:nm * r_target + mp * r_target].reshape(mp, r_target)
                W = x[nm * r_target + mp * r_target:].reshape(pn, r_target)
                return bilinear_error(U, V, W)
            
            try:
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 100, 'gtol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                continue
        
        # Check if we found a valid rank-21 solution
        if best_params is not None and best_err < 1e-6:
            x = best_params
            U_opt = x[:nm * r_target].reshape(nm, r_target)
            V_opt = x[nm * r_target:nm * r_target + mp * r_target].reshape(mp, r_target)
            W_opt = x[nm * r_target + mp * r_target:].reshape(pn, r_target)
            
            # Round to nearest half-integer and verify
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Convert to Fraction for exact verification
            U_frac = [[Fraction(int(v*2))/2 for v in row] for row in U_rounded]
            V_frac = [[Fraction(int(v*2))/2 for v in row] for row in V_rounded]
            W_frac = [[Fraction(int(v*2))/2 for v in row] for row in W_rounded]
            
            if verify_decomposition(U_frac, V_frac, W_frac):
                return {
                    "U": to_int_or_half(U_frac),
                    "V": to_int_or_half(V_frac),
                    "W": to_int_or_half(W_frac)
                }
        
        # Return rank-22 as valid improvement over 24
        return {
            "U": to_int_or_half(U22),
            "V": to_int_or_half(V22),
            "W": to_int_or_half(W22)
        }
    
    # Fallback: schoolbook rank 24
    U, V, W = schoolbook()
    return {
        "U": to_int_or_half(U),
        "V": to_int_or_half(V),
        "W": to_int_or_half(W)
    }

# EVOLVE-BLOCK-END
