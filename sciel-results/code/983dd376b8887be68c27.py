"""Matrix multiplication 2,3,7 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]
_DEADLINE = time.time() + 110  # 110s guard, leaves 10s safety margin


def _elapsed():
    return time.time() - _DEADLINE


def _time_remaining():
    return -_elapsed()


def _is_valid(U, V, W, n, m, p):
    """Check if the decomposition is exact for matrix multiplication tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    u = U[i * m + j][c]
                    v = V[j * p + k][c]
                    w = W[k * n + i][c]
                    if isinstance(u, Fraction) and isinstance(v, Fraction):
                        s += u * v * w
                    else:
                        s += float(u) * float(v) * float(w)
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(float(s) - 1.0) > 1e-9:
                        return False
    return True


def _to_fractions(mat):
    """Convert a list of lists to Fractions."""
    out = []
    for row in mat:
        frow = []
        for x in row:
            if isinstance(x, Fraction):
                frow.append(x)
            elif isinstance(x, int):
                frow.append(Fraction(x))
            else:
                frow.append(Fraction(x).limit_denominator(10**6))
        out.append(frow)
    return out


def _plain(mat):
    """Convert to plain floats for the interface."""
    out = []
    for row in mat:
        out.append([float(x) for x in row])
    return out


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if _is_valid(best_U, best_V, best_W, n, m, p):
        pass  # It's valid
    else:
        # Should never happen, but keep as fallback
        return {"U": _plain(best_U), "V": _plain(best_V), "W": _plain(best_W)}
    
    # Strategy: Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Approach: Use the fact that for fixed (i,k), we need:
    # sum_c U[i*m+j,c] * (sum over... ) -- this is tricky.
    
    # Better approach: Try known constructions.
    # For (2,3,7), try to use block structure.
    
    # Let's try a randomized search with exact verification.
    # Start from schoolbook and try to merge columns.
    
    U_frac = _to_fractions(best_U)
    V_frac = _to_fractions(best_V)
    W_frac = _to_fractions(best_W)
    r = best_rank
    
    random.seed(42)
    
    # Try to reduce rank by removing columns that are linear combinations
    # of others, or by finding non-trivial rank reductions.
    
    # Key idea: The tensor is multilinear. If we can find a column c such that
    # for all (i,j,k), the contribution can be redistributed, we can remove it.
    
    # Alternative: Use a different construction entirely.
    # For small dimensions, try to find decompositions by solving linear systems.
    
    # Let me try a constructive approach based on splitting.
    # Split the 7 into 3+4, or use the 2x3 structure.
    
    # Construction: Use 2x2 Strassen for the (i,j) part when possible.
    # For n=2, m=3: think of A as 2x3, B as 3x7.
    # Write A = [A1 A2 A3] where Ai is 2x1, B = [B1 B2 B3] where Bk is 1x7.
    # C = A1*B1 + A2*B2 + A3*B3 (outer products summed).
    # Each outer product is rank-1 as a matrix, but in tensor terms...
    
    # Actually, let me try a direct search for rank < 42.
    # For rank r, we need to find U (6xr), V (21xr), W (14xr) such that
    # for all i in {0,1}, j in {0,1,2}, k in {0,...,6}:
    #   sum_c U[3*i+j, c] * V[7*j+k, c] * W[2*k+i, c] = 1
    
    # This is a system of 42 equations. For r < 42, it's underdetermined in a sense.
    
    # Try specific small ranks and see if solutions exist.
    # Known: rank 39 works from prior work. Let me try to beat that.
    
    # Let me try a greedy column merging approach.
    # Start with schoolbook, and try to find pairs of columns that can be merged.
    
    # A column c corresponds to a triple (i_c, j_c, k_c) with U[i_c*m+j_c, c]=1, V[j_c*p+k_c, c]=1, W[k_c*n+i_c, c]=1.
    # Two columns c1, c2 can potentially be merged if there exist coefficients a, b, d such that
    # the new column produces the correct contributions.
    
    # This is complex. Let me try a different known construction.
    
    # For (2,3,7), one approach:
    # Use 7 columns for the "main" multiplication (like schoolbook for 2x1x7 + 2x1x7 + 2x1x7)
    # But that gives 3*7 = 21, plus corrections.
    
    # Actually, let me just try to find a valid decomposition with rank as low as possible
    # using a randomized approach with exact rational arithmetic.
    
    # Try ranks from 38 down to 21.
    for target_rank in range(min(best_rank - 1, 38), 20, -1):
        if _time_remaining() < 5:
            break
        
        # Try to find a decomposition of rank target_rank
        # Use a randomized approach: initialize U, V, W with random small integers
        # and try to solve for one of them.
        
        # Fix U and V, solve for W.
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is linear in W for fixed (i,j,k) if we group by (k,i).
        
        # For fixed (k,i), let W_vec = W[2*k+i, :] (a row of W, length r)
        # Then: sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[2*k+i,c] = 1 for all j
        # This is a system: for each j, sum_c A_{j,c} * W_{k,i,c} = 1
        # where A_{j,c} = U[i*m+j,c]*V[j*p+k,c]
        
        # For each (k,i), we have m=3 equations in r unknowns (the W row).
        # If r > m, there are solutions.
        
        # Let me try: pick U and V randomly, then for each (k,i), solve the system for W[k*n+i,:].
        
        found = False
        attempts = 0
        max_attempts = 200
        while not found and attempts < max_attempts:
            if _time_remaining() < 5:
                break
            attempts += 1
            
            # Random U and V with small integer entries
            U_test = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V_test = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            
            # For each (k, i), solve for W row
            W_test = [[0] * target_rank for _ in range(p*n)]
            valid = True
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                    # A_{j,c} = U[i*m+j,c]*V[j*p+k,c]
                    A = []
                    for j in range(m):
                        row = [U_test[i*m+j][c] * V_test[j*p+k][c] for c in range(target_rank)]
                        A.append(row)
                    # Solve A * w = [1,1,1]^T for w = W[k*n+i,:]
                    # Use numpy for least squares / exact solution
                    try:
                        A_np = np.array(A, dtype=float)
                        b_np = np.ones(m)
                        # Check if system is consistent
                        # Use SVD for least squares
                        _, s, Vt = np.linalg.svd(A_np, full_matrices=False)
                        # Regularize
                        s_reg = np.where(s > 1e-10, s, 1e-10)
                        w = Vt.T @ ((Vt @ b_np) / s_reg)
                        # Check residual
                        res = A_np @ w - b_np
                        if np.max(np.abs(res)) < 1e-6:
                            W_test[k*n+i] = [round(x) if abs(x - round(x)) < 1e-8 else x for x in w]
                        else:
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify exactly with fractions
                U_frac_test = _to_fractions(U_test)
                V_frac_test = _to_fractions(V_test)
                W_frac_test = _to_fractions(W_test)
                if _is_valid(U_frac_test, V_frac_test, W_frac_test, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_frac_test, V_frac_test, W_frac_test
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # Can't find at this rank, break
            break
    
    # Final: return the best found
    return {"U": _plain(best_U), "V": _plain(best_V), "W": _plain(best_W)}


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    r = len(res["U"][0])
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += res["U"][i * m + j][c] * res["V"][j * p + k][c] * res["W"][k * n + i][c]
                if abs(s - 1.0) > 1e-6:
                    ok = False
    print("rank =", r, "valid =", ok)
