"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    where i,j,k in {0,1}
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Each column r corresponds to one Strassen term
    # U[i*m+j, r] = coefficient of A[i,j] in the left factor
    # V[j*p+k, r] = coefficient of B[j,k] in the right factor  
    # W[k*n+i, r] = coefficient in output C[i,k]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0*2+0]=1, U[1*2+1]=1
    # V: B00=1, B11=1 -> V[0*2+0]=1, V[1*2+1]=1
    # W: C00 += 1, C11 += 1 -> W[0*2+0]=1, W[1*2+1]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> U[1*2+0]=1, U[1*2+1]=1
    # V: B00=1 -> V[0*2+0]=1
    # W: C10 += 1, C11 -= 1 -> W[0*2+1]=1, W[1*2+1]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00(B10-B11)
    # U: A00=1 -> U[0*2+0]=1
    # V: B10=1, B11=-1 -> V[1*2+0]=1, V[1*2+1]=-1
    # W: C01 += 1, C11 += 1 -> W[0*2+1]=1, W[1*2+1]=1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B01-B10)
    # U: A11=1 -> U[1*2+1]=1
    # V: B01=1, B10=-1 -> V[0*2+1]=1, V[1*2+0]=-1
    # W: C00 += 1, C10 += 1 -> W[0*2+0]=1, W[0*2+1]=1
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # U: A00=1, A01=1 -> U[0*2+0]=1, U[0*2+1]=1
    # V: B11=1 -> V[1*2+1]=1
    # W: C00 -= 1, C01 += 1 -> W[0*2+0]=-1, W[0*2+1]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[1*2+0]=1, U[0*2+0]=-1
    # V: B00=1, B01=1 -> V[0*2+0]=1, V[0*2+1]=1
    # W: C11 += 1 -> W[1*2+1]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[0*2+1]=1, U[1*2+1]=-1
    # V: B10=1, B11=1 -> V[1*2+0]=1, V[1*2+1]=1
    # W: C00 += 1 -> W[0*2+0]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_4x4x4():
    """Build rank-49 decomposition for 4x4x4 using block Strassen.
    
    Split 4x4 into 2x2 blocks. For each of the 4 output 2x2 blocks,
    we need two 2x2 products. Use Strassen (rank 7) for each product.
    Total: 4 blocks * 2 products * 7 = 56... but we can share.
    
    Actually, the standard approach: 
    C00 = A00*B00 + A01*B10  (2 products, each rank 7 = 14)
    C01 = A00*B01 + A01*B11  (2 products, each rank 7 = 14)
    C10 = A10*B00 + A11*B10  (2 products, each rank 7 = 14)
    C11 = A10*B01 + A11*B11  (2 products, each rank 7 = 14)
    Total = 56
    
    But we can do better by sharing Strassen terms. The known rank-49
    comes from a more careful construction. Let me use the 56-rank
    version first, then try to optimize.
    """
    n, m, p = 4, 4, 4
    U, V, W = _strassen_2x2()
    
    # We'll build the 4x4x4 decomposition by placing 2x2x2 Strassen
    # decompositions at the appropriate block positions.
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C = [[C00, C01], [C10, C11]] (each 2x2)
    
    # For 4x4: indices i,j,k in 0..3
    # Block row: i//2, block col: j//2 for A
    # Block row: j//2, block col: k//2 for B
    # Block row: i//2, block col: k//2 for C
    
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # Each 2x2x2 Strassen term uses 7 rank-1 terms
    # We need 8 such terms (for the 4 C blocks, each needing 2 products)
    # Total rank = 8 * 7 = 56
    
    total_rank = 56
    U4 = np.zeros((16, total_rank))
    V4 = np.zeros((16, total_rank))
    W4 = np.zeros((16, total_rank))
    
    # For each output block (ci, ck) in {0,1}^2:
    # C[ci, ck] = A[ci, 0]*B[0, ck] + A[ci, 1]*B[1, ck]
    # For each of these two products, use Strassen
    
    col = 0
    for ci in range(2):  # output block row
        for ck in range(2):  # output block col
            for aj in range(2):  # which A block column / B block row
                # This is product: A[ci, aj] * B[aj, ck]
                # A[ci, aj] is at rows (ci*2, aj*2) to (ci*2+1, aj*2+1)
                # B[aj, ck] is at rows (aj*2, ck*2) to (aj*2+1, ck*2+1)
                # C[ci, ck] is at rows (ck*2, ci*2) to (ck*2+1, ci*2+1)
                # Note: W index is k*n+i, so for C[ci,ck], k=ck*2+dk, i=ci*2+di
                
                for r in range(7):
                    # U: A[ci, aj] block
                    for di in range(2):
                        for dj in range(2):
                            U4[(ci*2+di)*4 + (aj*2+dj), col] = U[di*2+dj, r]
                    
                    # V: B[aj, ck] block
                    for dj in range(2):
                        for dk in range(2):
                            V4[(aj*2+dj)*4 + (ck*2+dk), col] = V[dj*2+dk, r]
                    
                    # W: C[ci, ck] block
                    for dk in range(2):
                        for di in range(2):
                            W4[(ck*2+dk)*4 + (ci*2+di), col] = W[dk*2+di, r]
                    
                    col += 1
    
    return U4, V4, W4, total_rank

def _build_4x4x2_schoolbook():
    """Build rank-32 schoolbook decomposition for 4x4x2."""
    n, m, p = 4, 4, 2
    rank = n * m * p  # 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n=4, m=4, p=6):
    """Verify that the decomposition is exact."""
    # T[i,j,k] = 1 for all i,j,k (since every output depends on every input pair)
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    return max_err

def _try_optimize_rank(U_base, V_base, W_base, target_rank, n=4, m=4, p=6, time_limit=100.0):
    """Try to find a lower-rank decomposition by optimizing.
    
    Start with the base decomposition and try to reduce rank by
    combining terms or finding a lower-rank representation.
    """
    start_time = time.time()
    U = np.array(U_base, dtype=float)
    V = np.array(V_base, dtype=float)
    W = np.array(W_base, dtype=float)
    
    base_rank = U.shape[1]
    
    # Strategy: Try to find a rank-r decomposition by solving
    # a system of equations. For each candidate rank r, we need to find
    # U (16 x r), V (24 x r), W (24 x r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a non-convex problem. We'll use a greedy approach:
    # Start with the full decomposition and try to merge/remove terms.
    
    best_U, best_V, best_W = U, V, W
    best_rank = base_rank
    
    # Try various target ranks
    for target_r in range(min(base_rank - 1, 85), 40, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random perturbations of the base
        np.random.seed(target_r)
        
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Start from base and try to zero out some columns
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Try to reduce by combining similar terms
            # For now, just check if we can simply drop some terms
            # (this won't work for exact decomposition, but let's try)
            
            # Better approach: use the fact that if we have a valid
            # decomposition of rank R, we can try to find a valid
            # decomposition of rank R-1 by solving a linear system.
            
            # For a rank-r decomposition, we have r * (16 + 24 + 24) = 56r variables
            # and 4*4*6 = 96 constraints (one for each i,j,k)
            # Plus the bilinear structure makes it non-linear.
            
            # Let's try a different approach: use the base decomposition
            # and see if any column is redundant (can be expressed as
            # a combination of others that still satisfies the tensor equation).
            
            # Check if any single term can be removed
            for c in range(target_r, base_rank):
                # Try removing column c
                U_test = np.delete(U, c, axis=1)
                V_test = np.delete(V, c, axis=1)
                W_test = np.delete(W, c, axis=1)
                
                err = _verify_decomposition(U_test, V_test, W_test, n, m, p)
                if err < 1e-10:
                    U = U_test
                    V = V_test
                    W = W_test
                    best_rank = U.shape[1]
                    break
            else:
                continue
            
            # If we removed a term, continue trying to remove more
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 
    multiplying a 4x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    start_time = time.time()
    DEADLINE = 140.0
    
    # Build the 4x4x4 Strassen part (rank 56)
    U4, V4, W4, rank4 = _build_strassen_4x4x4()
    
    # Build the 4x4x2 schoolbook part (rank 32)
    U2, V2, W2, rank2 = _build_4x4x2_schoolbook()
    
    # Combine into full 4x4x6 decomposition
    total_rank = rank4 + rank2  # 56 + 32 = 88
    
    U_full = np.zeros((n*m, total_rank))  # 16 x 88
    V_full = np.zeros((m*p, total_rank))  # 24 x 88
    W_full = np.zeros((p*n, total_rank))  # 24 x 88
    
    # Place 4x4x4 part (k=0,1,2,3)
    # U4 has shape (16, 56) - A indices
    # V4 has shape (16, 56) - B indices for k=0..3
    # W4 has shape (16, 56) - C indices for k=0..3
    
    U_full[:, :rank4] = U4
    
    # For V: B has shape 4x6, so V index is j*p+k where j in 0..3, k in 0..5
    # For the 4x4x4 part, k is 0..3, so V4[j*4+k, c] -> V_full[j*6+k, c]
    for j in range(4):
        for k in range(4):
            V_full[j*6+k, :rank4] = V4[j*4+k, :rank4]
    
    # For W: C has shape 4x6, so W index is k*n+i where k in 0..5, i in 0..3
    # For the 4x4x4 part, k is 0..3, so W4[k*4+i, c] -> W_full[k*4+i, c]
    W_full[:rank4*0 + 16, :rank4] = W4  # k=0..3, i=0..3 -> indices 0..15
    
    # Place 4x4x2 part (k=4,5)
    U_full[:, rank4:] = U2
    
    # For V: B indices j*6+k where k=4,5
    # U2 has shape (16, 32), V2 has shape (8, 32) for 4x2 B
    # V2[j*2+k, c] where j in 0..3, k in 0..1 -> V_full[j*6+(4+k), c]
    for j in range(4):
        for k in range(2):
            V_full[j*6+(4+k), rank4:] = V2[j*2+k, rank4-rank4:]  # V2[j*2+k, :]
    
    # For W: C indices k*4+i where k=4,5
    # W2 has shape (8, 32) for 4x2 C: W2[k*4+i, c] where k in 0..1 (local), i in 0..3
    # -> W_full[(4+k)*4+i, c]
    for k in range(2):
        for i in range(4):
            W_full[(4+k)*4+i, rank4:] = W2[k*4+i, :]
    
    # Verify
    err = _verify_decomposition(U_full, V_full, W_full, n, m, p)
    
    if err > 1e-10:
        # Fallback to schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to optimize to lower rank
    if time.time() - start_time < DEADLINE - 20:
        U_opt, V_opt, W_opt, opt_rank = _try_optimize_rank(
            U_full, V_full, W_full, total_rank - 1, n, m, p,
            time_limit=DEADLINE - time.time() - 15
        )
        if opt_rank < total_rank:
            U_full, V_full, W_full = U_opt, V_opt, W_opt
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U_full]
    V_list = [[float(x) for x in row] for row in V_full]
    W_list = [[float(x) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
