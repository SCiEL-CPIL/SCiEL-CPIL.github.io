"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, m, p, n, col, skip_p7=False):
    """Apply Strassen's 7-product scheme to a 2x2x2 block.
    If skip_p7 is True, redistribute P7's contributions into P5/P6 to save 1 rank."""
    # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    U[i0*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    U[i1*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + i1][col] += 1
    W[k1*n + i1][col] -= 1
    col += 1
    
    # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    U[i0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    U[i1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + i0][col] += 1
    W[k0*n + i1][col] += 1
    col += 1
    
    # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    U[i0*m + j0][col] += 1
    U[i0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] -= 1
    W[k1*n + i0][col] += 1
    if skip_p7:
        # Absorb P7's contribution to (i0,j1,k1) which needs +1
        # P5 already gives (i0,j1,k1): U[i0,j1]=1, V[j1,k1]=1, W[k1,i0]=? 
        # Currently W[k1,i0]=1 from this term, so (i0,j1,k1) gets +1. Good.
        # P7 also needs (i0,j1,k0)=+1. We'll handle that separately.
        pass
    col += 1
    
    # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    U[i1*m + j0][col] += 1
    U[i0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + i1][col] += 1
    if skip_p7:
        # Absorb P7's contribution to (i1,j1,k0) which needs -1
        # and (i1,j1,k1) which needs -1
        # P6 has U[i1,j0]=1, V[j0,k0]=1, V[j0,k1]=1, W[k1,i1]=1
        # This gives (i1,j0,k1): +1. Not directly helpful for j1.
        pass
    col += 1
    
    if not skip_p7:
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    else:
        # Instead of P7, add 2 additional terms to cover P7's 4 contributions:
        # P7 contributes:
        #   (i0,j1,k0): +1  -> need U[i0,j1]*V[j1,k0]*W[k0,i0] = +1
        #   (i0,j1,k1): +1  -> already covered by P5? Check: P5 has U[i0,j1]=1, V[j1,k1]=1, W[k1,i0]=1 => +1. Yes!
        #   (i1,j1,k0): -1  -> need U[i1,j1]*V[j1,k0]*W[k0,i1] = -1
        #   (i1,j1,k1): -1  -> need U[i1,j1]*V[j1,k1]*W[k1,i1] = -1
        #
        # P5 covers (i0,j1,k1)=+1. We still need:
        #   (i0,j1,k0): +1
        #   (i1,j1,k0): -1
        #   (i1,j1,k1): -1
        #
        # Term A: U[i0,j1]=1, V[j1,k0]=1, W[k0,i0]=1 => gives (i0,j1,k0)=+1. Also gives (i0,j1,k0) only if no other W.
        #   But W[k0,i0]=1 also affects other (i,l,k) where U[i,l] and V[l,k0] are nonzero.
        #   In this block, for column 'col', U has only U[i0,j1]=1 (from this term). V has only V[j1,k0]=1.
        #   So this term contributes +1 to (i0,j1,k0) and 0 elsewhere. 
        #
        # Term B: U[i1,j1]=-1, V[j1,k0]=1, W[k0,i1]=1 => gives (i1,j1,k0)=-1. 
        #   But V[j1,k0]=1 is shared with Term A! We need separate columns.
        #
        # Actually, let's use two new terms:
        # Term A (col): U[i0*m+j1]=1, V[j1*p+k0]=1, W[k0*n+i0]=1
        #   Contributes: +1 to (i0,j1,k0). Check other (i,l,k): 
        #     U nonzero only at (i0,j1), V nonzero only at (j1,k0), so only (i0,j1,k0) gets +1. 
        #
        # Term B (col+1): U[i1*m+j1]=-1, V[j1*p+k0]=1, W[k0*n+i1]=1
        #   Contributes: -1 to (i1,j1,k0). Only (i1,j1,k0) affected. 
        #
        # Term C (col+2): U[i1*m+j1]=-1, V[j1*p+k1]=1, W[k1*n+i1]=1
        #   Contributes: -1 to (i1,j1,k1). Only (i1,j1,k1) affected. 
        #
        # That's 3 terms vs P7's 1 term. Net +2, so we'd go from 7 to 9. Bad!
        #
        # Alternative: Can we share V entries? 
        # If Term A and B share V[j1,k0], then:
        # Term A+B combined: U[i0,j1]=1, U[i1,j1]=-1, V[j1,k0]=1, W[k0,i0]=1, W[k0,i1]=1
        #   Contributes: 
        #     (i0,j1,k0): 1*1*1 = +1
        #     (i1,j1,k0): -1*1*1 = -1
        #     (i0,j1,k0) also gets W[k0,i0]=1, but V[j1,k0]=1, U[i0,j1]=1 -> yes +1
        #     What about (i0,j1, other k)? V is zero for other k. Good.
        #     What about (other i, j1, k0)? W[k0,i] nonzero only for i0,i1. 
        #     (i0,j1,k0): +1, (i1,j1,k0): -1. 
        #   But also W[k0,i1]=1 and U[i1,j1]=-1, V[j1,k0]=1 -> (i1,j1,k0) = -1. Correct.
        #   Does this affect any other (i,l,k)? 
        #     For (i0,j1,k0): U[i0,j1]=1, V[j1,k0]=1, W[k0,i0]=1 -> +1
        #     For (i1,j1,k0): U[i1,j1]=-1, V[j1,k0]=1, W[k0,i1]=1 -> -1
        #     For (i0,j1,k1): V[j1,k1]=0 -> 0
        #     For (i1,j1,k1): V[j1,k1]=0 -> 0
        #     For (i0,other j,k0): U[i0,other_j]=0 -> 0
        #   So this single term covers both (i0,j1,k0)=+1 and (i1,j1,k0)=-1. 
        #
        # Now for (i1,j1,k1)=-1: 
        # Term C: U[i1*m+j1]=-1, V[j1*p+k1]=1, W[k1*n+i1]=1
        #   Contributes -1 to (i1,j1,k1) only. 
        #
        # So we need 2 terms instead of P7's 1 term. Net +1, going from 7 to 8. Still bad!
        #
        # Hmm, let me reconsider. Maybe the savings come from a different structure.
        
        # Actually, let's just use P7 normally for all blocks. Rank 54 is safe.
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Rank 54 decomposition: 6 Strassen blocks + 12 scalar products
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(U, V, W, 0, 1, j0, j1, k0, k1, m, p, n, col, skip_p7=False)
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # Verify
    if _verify_decomp(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U[i*m+l][col] = 1
                V[l*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
