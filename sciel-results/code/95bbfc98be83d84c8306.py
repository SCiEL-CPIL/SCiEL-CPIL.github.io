import numpy as np
from itertools import product
import time

def construct():
    t0 = time.time()
    deadline = t0 + 130
    
    n, m, p = 3, 4, 4
    
    # Build the multiplication tensor T[a,b,c] where c = p*n + row, and we want
    # sum_k U[a,k]*V[b,k]*W[c,k] == T[a,b,c]
    # T[a,b,c] = 1 if the entry (a,b) of A*B contributes to entry c of C
    # A is n x m, B is m x p, C is n x p
    # C[i,j] = sum_l A[i,l] * B[l,j]
    # a = i*m + l, b = l*p + j, c = j*n + i
    
    # Build sparse list of non-zero tensor entries
    tensor_entries = []  # list of (a, b, c)
    for i in range(n):
        for l in range(m):
            for j in range(p):
                a = i * m + l
                b = l * p + j
                c = j * n + i
                tensor_entries.append((a, b, c))
    
    # Verify schoolbook: rank = m = 4? No, schoolbook rank for matmul is m (sum over l)
    # Actually schoolbook rank = m for n x m times m x p? No!
    # C[i,j] = sum_{l=0}^{m-1} A[i,l] B[l,j]
    # So rank = m = 4? That can't be right for 3x4x4.
    # Wait - the tensor is the matmul tensor. Its rank is the matmul rank.
    # For (n,m,p) matmul, schoolbook rank = m (the shared dimension).
    # But that's only if we use the standard decomposition. The tensor rank might be lower.
    # Actually no - for (n,m,p), the "schoolbook" decomposition has rank m, but we need
    # to check: each term is A[i,l] * B[l,j] * delta_{c, j*n+i}
    # So for fixed l, we get terms for all (i,j). That's n*p terms per l.
    # U has entries at (i*m+l, l) = 1 for all i
    # V has entries at (l*p+j, l) = 1 for all j  
    # W has entries at (j*n+i, l) = 1 for all (i,j)
    # So rank = m = 4 schoolbook!
    
    # Let me verify: with rank 4, U[i*m+l, l] = 1, V[l*p+j, l] = 1, W[j*n+i, l] = 1
    # Then sum_l U[a,k]V[b,k]W[c,k] = sum_l [a=im+l][b=l'p+j][l=l'][c=j'n+i']
    # = sum_l delta(a, i*m+l) * delta(b, l*p+j) * delta(c, j*n+i)
    # For this to be 1, we need a = i*m+l, b = l*p+j, c = j*n+i for some consistent i,j,l
    # a = i*m + l, b = l*p + j, c = j*n + i
    # So the product is 1 iff there exist i,j,l such that a=im+l, b=lp+j, c=jn+i
    # This is exactly the tensor! So rank 4 works!
    
    # Wait, but the current solution has rank 44. Let me re-examine.
    # n=3, m=4, p=4. Schoolbook rank = m = 4.
    # Let me build it and verify.
    
    rank = m  # = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for l in range(m):
            for j in range(p):
                U[i * m + l, l] = 1
                V[l * p + j, l] = 1
                W[j * n + i, l] = 1
    
    # Verify: check that the decomposition is exact
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # For each (a, b, c), check sum_k U[a,k]*V[b,k]*W[c,k] == T[a,b,c]
        for (a, b, c) in tensor_entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-12:
                return False, (a, b, c, val)
        # Also check that entries NOT in tensor_entries are 0
        all_indices = list(product(range(n*m), range(m*p), range(p*n)))
        for (a, b, c) in all_indices:
            if (a, b, c) not in tensor_entries:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-12:
                    return False, (a, b, c, val)
        return True, None
    
    ok, err = verify(U, V, W)
    
    if ok:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If verification failed (shouldn't happen), fall back to a known working decomposition
    # Try the current rank-44 approach as fallback
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Build the local 3x2x2 factors (same as original)
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def place_block(col_off, a_row_s, a_col_s, b_row_s, b_col_s, c_row_s, c_col_s):
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                ga = (a_row_s + r) * m + (a_col_s + c)
                U[ga, col_off:col_off+11] += U_loc[li, :]
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gb = (b_row_s + r) * p + (b_col_s + c)
                V[gb, col_off:col_off+11] += V_loc[li, :]
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gc = (c_col_s + c) * n + (c_row_s + r)
                W[gc, col_off:col_off+11] += W_loc[li, :]
    
    place_block(0, 0, 0, 0, 0, 0, 0)
    place_block(11, 0, 2, 2, 0, 0, 0)
    place_block(22, 0, 0, 0, 2, 0, 2)
    place_block(33, 0, 2, 2, 2, 0, 2)
    
    ok2, err2 = verify(U, V, W)
    if ok2:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: return zeros (will score 0 but shouldn't happen)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
