"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

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

    def build_rank22():
        """Build the rank-22 decomposition using Strassen blocks."""
        rank = 22
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        
        col = 0
        
        # Block 1: A_L (2x2) * B_T1 (2x2) -> C1 (2x2)
        # A_L: A[:, 0:2], indices 0,1,3,4
        # B_T1: B[0:2, 0:2], indices 0,1,4,5
        # C1: C[:, 0:2], W indices 0,2,1,3
        
        # M1: (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[5][col] = 1
        W[0][col] = 1; W[3][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[3][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[0][col] = 1
        V[1][col] = 1; V[5][col] = -1
        W[2][col] = 1; W[3][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[4][col] = 1
        V[4][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[5][col] = 1
        W[0][col] = -1; W[2][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[3][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[4][col] = 1; V[5][col] = 1
        W[0][col] = 1
        col += 1
        
        # Block 2: A_L (2x2) * B_T2 (2x2) -> C2 (2x2)
        # B_T2: B[0:2, 2:4], indices 2,3,6,7
        # C2: C[:, 2:4], W indices 4,6,5,7
        
        # M1: (A00+A11)(B02+B13)
        U[0][col] = 1; U[4][col] = 1
        V[2][col] = 1; V[7][col] = 1
        W[4][col] = 1; W[7][col] = 1
        col += 1
        
        # M2: (A10+A11)B02
        U[3][col] = 1; U[4][col] = 1
        V[2][col] = 1
        W[5][col] = 1; W[7][col] = -1
        col += 1
        
        # M3: A00(B03-B13)
        U[0][col] = 1
        V[3][col] = 1; V[7][col] = -1
        W[6][col] = 1; W[7][col] = 1
        col += 1
        
        # M4: A11(B12-B02)
        U[4][col] = 1
        V[6][col] = 1; V[2][col] = -1
        W[4][col] = 1; W[5][col] = 1
        col += 1
        
        # M5: (A00+A01)B13
        U[0][col] = 1; U[1][col] = 1
        V[7][col] = 1
        W[4][col] = -1; W[6][col] = 1
        col += 1
        
        # M6: (A10-A00)(B02+B03)
        U[3][col] = 1; U[0][col] = -1
        V[2][col] = 1; V[3][col] = 1
        W[7][col] = 1
        col += 1
        
        # M7: (A01-A11)(B12+B13)
        U[1][col] = 1; U[4][col] = -1
        V[6][col] = 1; V[7][col] = 1
        W[4][col] = 1
        col += 1
        
        # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4)
        # A_R: A[:, 2], indices 2,5
        # B_B: B[2, :], indices 8,9,10,11
        
        for i in range(2):
            for k in range(4):
                U[i*m+2][col] = 1
                V[2*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        
        return U, V, W, rank

    # Build rank-22 decomposition
    U22, V22, W22, rank22 = build_rank22()
    
    # Verify rank-22
    U22_frac = [[Fraction(int(v)) for v in row] for row in U22]
    V22_frac = [[Fraction(int(v)) for v in row] for row in V22]
    W22_frac = [[Fraction(int(v)) for v in row] for row in W22]
    
    if verify_decomposition(U22_frac, V22_frac, W22_frac):
        # Try to reduce to rank 21 using optimization
        U_np = np.array(U22, dtype=float)
        V_np = np.array(V22, dtype=float)
        W_np = np.array(W22, dtype=float)
        
        r_target = 21
        best_err = float('inf')
        best_params = None
        
        def bilinear_error(x):
            """Compute squared error of bilinear verification from flattened params."""
            U = x[:nm * r_target].reshape(nm, r_target)
            V = x[nm * r_target:nm * r_target + mp * r_target].reshape(mp, r_target)
            W = x[nm * r_target + mp * r_target:].reshape(pn, r_target)
            
            total_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.dot(np.multiply(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                        total_err += (actual - 1.0) ** 2
            return total_err
        
        from scipy.optimize import minimize
        
        # Try dropping each column
        for drop_col in range(22):
            if time.time() > deadline - 15:
                break
            
            keep = [c for c in range(22) if c != drop_col]
            U_init = U_np[:, keep]
            V_init = V_np[:, keep]
            W_init = W_np[:, keep]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Try BFGS
            try:
                res = minimize(bilinear_error, x0, method='BFGS',
                               options={'maxiter': 50, 'gtol': 1e-7})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            
            # Try L-BFGS-B with slightly perturbed initial values
            if time.time() > deadline - 10:
                break
            try:
                x0_perturbed = x0 + np.random.normal(0, 0.1, len(x0))
                res2 = minimize(bilinear_error, x0_perturbed, method='L-BFGS-B',
                                options={'maxiter': 50, 'gtol': 1e-7})
                if res2.fun < best_err:
                    best_err = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
        
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
            
            U_frac = [[Fraction(int(round(v*2)))/2 for v in row] for row in U_rounded]
            V_frac = [[Fraction(int(round(v*2)))/2 for v in row] for row in V_rounded]
            W_frac = [[Fraction(int(round(v*2)))/2 for v in row] for row in W_rounded]
            
            if verify_decomposition(U_frac, V_frac, W_frac):
                return {
                    "U": to_int_or_half(U_frac),
                    "V": to_int_or_half(V_frac),
                    "W": to_int_or_half(W_frac)
                }
        
        # Return rank-22 as valid result
        return {
            "U": to_int_or_half(U22_frac),
            "V": to_int_or_half(V22_frac),
            "W": to_int_or_half(W22_frac)
        }
    
    # Fallback: schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    U_frac = [[Fraction(int(v)) for v in row] for row in U]
    V_frac = [[Fraction(int(v)) for v in row] for row in V]
    W_frac = [[Fraction(int(v)) for v in row] for row in W]
    
    return {
        "U": to_int_or_half(U_frac),
        "V": to_int_or_half(V_frac),
        "W": to_int_or_half(W_frac)
    }

# EVOLVE-BLOCK-END
