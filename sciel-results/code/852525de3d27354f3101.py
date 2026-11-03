"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication, rank 7."""
    # U indexes A entries: (i,j) -> i*2+j, shape (4, 7)
    # V indexes B entries: (j,k) -> j*2+k, shape (4, 7)
    # W indexes C entries: (k,i) -> k*2+i, shape (4, 7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A entries: A00=0, A01=1, A10=2, A11=3
    # B entries: B00=0, B01=1, B10=2, B11=3
    # C entries: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] += 1; U[3,0] += 1
    V[0,0] += 1; V[3,0] += 1
    W[0,0] += 1; W[3,0] -= 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] += 1; U[3,1] += 1
    V[0,1] += 1
    W[2,1] += 1  # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0,2] += 1
    V[2,2] += 1; V[3,2] -= 1
    W[1,2] += 1; W[3,2] += 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B10)
    U[3,3] += 1
    V[1,3] += 1; V[2,3] -= 1
    W[0,3] -= 1; W[1,3] += 1  # C00 -= M4, C01 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] += 1; U[1,4] += 1
    V[3,4] += 1
    W[0,4] += 1; W[2,4] += 1  # C00 += M5, C10 += M5
    
    # M6 = (A10-A00)(B00+B10)
    U[2,5] += 1; U[0,5] -= 1
    V[0,5] += 1; V[2,5] += 1
    # M6 not used in C for 2x2? Wait, let me recheck.
    # Actually M6 is not directly used. Let me recheck Strassen.
    
    # Actually, standard Strassen uses M1-M7 as above, and M6 is:
    # M6 = (A10-A00)(B00+B10) -- but this isn't in the standard 7.
    # Let me use the standard formulation:
    
    # Reset and use correct Strassen
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A11(B01-B10)
    U[3,3] = 1
    V[1,3] = 1; V[2,3] = -1
    W[0,3] = -1; W[1,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[2,4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    # M6 contributes to C? In standard Strassen, M6 is not directly in C.
    # Wait, I think I'm confusing myself. Let me just verify by checking
    # if sum_r U[a,r]*V[b,r]*W[c,r] gives the right tensor.
    
    # M7 = (A01-A11)(B01+B11)
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = -1; W[3,6] = 1
    
    return U, V, W

def _verify_2x2(U, V, W):
    """Verify that the 2x2x2 decomposition is correct."""
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(7):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _greedy_eliminate(U, V, W, tol=1e-9):
    """Try to remove linearly dependent columns from U, V, W."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Try removing each column and checking if the decomposition still holds
    # This is expensive, so we do it greedily
    kept = list(range(rank))
    
    for col in range(rank):
        if col not in kept:
            continue
        # Try removing column 'col'
        new_kept = [k for k in kept if k != col]
        if not new_kept:
            continue
        U_new = U[:, new_kept]
        V_new = V[:, new_kept]
        W_new = W[:, new_kept]
        
        # Check if the tensor is still decomposed
        n, m, p = 4, 4, 6
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(len(new_kept)):
                        actual += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                    if abs(actual - expected) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            kept = new_kept
    
    if not kept:
        return U, V, W
    
    return U[:, kept], V[:, kept], W[:, kept]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 4, 4, 6
    
    # Strategy: Use Kronecker product of Strassen(2x2x2) and schoolbook(2x2x3)
    # 4x4x6 = (2x2 blocks) x (2x2 blocks) x (2x3 blocks)
    # A is 4x4 = 2x2 blocks of 2x2
    # B is 4x6 = 2x3 blocks of 2x2 (2 rows, 3 cols of 2x2 blocks)
    # C is 4x6 = 2x3 blocks of 2x2
    
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x3: schoolbook rank = 2*2*3 = 12
    # Kronecker product rank: 7 * 12 = 84
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Schoolbook for 2x2x3
    n2, m2, p2 = 2, 2, 3
    rank2 = n2 * m2 * p2  # 12
    U_sb = np.zeros((n2*m2, rank2))
    V_sb = np.zeros((m2*p2, rank2))
    W_sb = np.zeros((p2*n2, rank2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U_sb[i*m2+j, col] = 1
                V_sb[j*p2+k, col] = 1
                W_sb[k*n2+i, col] = 1
                col += 1
    
    # Kronecker product: rank = 7 * 12 = 84
    rank = 7 * 12
    U = np.zeros((4*4, rank))  # 16 x 84
    V = np.zeros((4*6, rank))  # 24 x 84
    W = np.zeros((6*4, rank))  # 24 x 84
    
    # The Kronecker structure:
    # A[i1*2+i2, j1*2+j2] where i1,j1 in {0,1} are block indices, i2,j2 in {0,1} are within-block
    # B[j1*2+j2, k1*3+k2] where k1 in {0,1,2} is block col, k2 in {0,1} is within-block
    # C[k1*3+k2, i1*2+i2] -- wait, the trace convention is W[k*n+i, r]
    
    # Actually, let me think about this more carefully.
    # The tensor T[a, b, c] where:
    # a = i*m + j = i*4 + j (A entry)
    # b = j*p + k = j*6 + k (B entry)  
    # c = k*n + i = k*4 + i (C entry, trace convention)
    
    # For the Kronecker product approach, I need to be careful about how the
    # block structure maps to the tensor indices.
    
    # Let me use a different approach: directly construct using the known
    # 4x4x4 decomposition (if available) and extend to 4x4x6.
    
    # Actually, let me just use schoolbook and then try greedy elimination,
    # but start with a smarter initial decomposition.
    
    # Alternative: Use the fact that 4x4x6 can be seen as two 4x4x3 problems
    # or use other factorizations.
    
    # Let me try: 4x4x6 = 4x4x(4+2). Use rank-49 for 4x4x4 (if we have one) 
    # and rank-24 for 4x4x2, total 73. But we need a good 4x4x4.
    
    # For 4x4x4, we can use Strassen recursively: 2x2x2 has rank 7, so 
    # 4x4x4 would have rank 7*7 = 49? No, that's not how it works.
    # Strassen for 2x2 has rank 7. For 4x4, we decompose into 2x2 blocks,
    # giving 8 block multiplications, each of rank 7, so 56. But there are
    # linear dependencies, so the actual rank is less.
    
    # Let me just go with schoolbook (96) and do aggressive greedy elimination.
    
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now try to find a lower-rank decomposition using the structure.
    # The key insight: many of these rank-1 terms are linearly dependent.
    
    # Use the Strassen structure for 2x2x2 subproblems.
    # Decompose 4x4x6 into 2x2 block structure:
    # A: 2x2 blocks of 2x2 -> indices (a1,a2, b1,b2) where a1,b1 in {0,1}
    # B: 2x3 blocks of 2x2 -> indices (b1,b2, c1,c2) where c1 in {0,1,2}
    # C: 2x3 blocks of 2x2 -> indices (c1,c2, a1,a2)
    
    # For each block combination (a1, b1, c1), we have a 2x2x2 multiplication.
    # There are 2*2*3 = 12 such blocks.
    # Each can use Strassen (rank 7) instead of schoolbook (rank 8).
    # Total: 12 * 7 = 84 instead of 12 * 8 = 96.
    
    # Let me construct this properly.
    U = np.zeros((16, 84))
    V = np.zeros((24, 84))
    W = np.zeros((24, 84))
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    col = 0
    for a1 in range(2):  # block row of A
        for b1 in range(2):  # block col of A = block row of B
            for c1 in range(3):  # block col of B = block row of C
                for r in range(7):
                    # Within-block indices: a2, b2, c2 in {0,1}
                    # A entry: (a1*2 + a2, b1*2 + b2) -> index (a1*2+a2)*4 + (b1*2+b2)
                    # B entry: (b1*2 + b2, c1*2 + c2) -> index (b1*2+b2)*6 + (c1*2+c2)
                    # C entry: (c1*2 + c2, a1*2 + a2) -> index (c1*2+c2)*4 + (a1*2+a2)
                    
                    for a2 in range(2):
                        for b2 in range(2):
                            for c2 in range(2):
                                # This isn't quite right. The Strassen decomposition
                                # gives us the coefficients for the 2x2x2 tensor.
                                # U_s[a2*2+b2, r] * V_s[b2*2+c2, r] * W_s[c2*2+a2, r]
                                # should sum to 1 for the 2x2x2 tensor.
                                
                                a_idx = (a1*2 + a2) * 4 + (b1*2 + b2)
                                b_idx = (b1*2 + b2) * 6 + (c1*2 + c2)
                                c_idx = (c1*2 + c2) * 4 + (a1*2 + a2)
                                
                                U[a_idx, col] += U_s[a2*2 + b2, r]
                                V[b_idx, col] += V_s[b2*2 + c2, r]
                                W[c_idx, col] += W_s[c2*2 + a2, r]
                    col += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(84):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try greedy elimination
    if time.time() < deadline - 20:
        U, V, W = _greedy_eliminate(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
