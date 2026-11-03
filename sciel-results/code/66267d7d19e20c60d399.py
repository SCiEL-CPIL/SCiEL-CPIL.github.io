"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 is (0,0), A22 is (1,1)
    # B11 is (0,0), B22 is (1,1)
    U[0, 0] = 1 # A11
    U[3, 0] = 1 # A22
    V[0, 0] = 1 # B11
    V[3, 0] = 1 # B22
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W indices: k*n + i. n=2.
    # C11 is i=0, k=0 -> idx 0
    # C12 is i=0, k=1 -> idx 2
    # C21 is i=1, k=0 -> idx 1
    # C22 is i=1, k=1 -> idx 3
    
    W[0, 0] = 1 # C11 += M1
    W[2, 0] = 0 # C12 += 0
    W[1, 0] = 0 # C21 += 0
    W[3, 0] = 1 # C22 += M1
    
    # M2 = (A21 + A22)B11
    # A21 is (1,0), A22 is (1,1)
    # B11 is (0,0)
    U[2, 1] = 1 # A21
    U[3, 1] = 1 # A22
    V[0, 1] = 1 # B11
    # C21 += M2
    # C22 -= M2
    W[1, 1] = 1 # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B12 - B22)
    # A11 is (0,0)
    # B12 is (0,1), B22 is (1,1)
    U[0, 2] = 1 # A11
    V[1, 2] = 1 # B12
    V[3, 2] = -1 # -B22
    # C12 += M3
    # C22 += M3
    W[2, 2] = 1 # C12
    W[3, 2] = 1 # C22
    
    # M4 = A22(B21 - B11)
    # A22 is (1,1)
    # B21 is (1,0), B11 is (0,0)
    U[3, 3] = 1 # A22
    V[2, 3] = 1 # B21
    V[0, 3] = -1 # -B11
    # C11 -= M4
    # C21 += M4
    W[0, 3] = -1 # C11
    W[1, 3] = 1 # C21
    
    # M5 = (A11 + A12)B22
    # A11 is (0,0), A12 is (0,1)
    # B22 is (1,1)
    U[0, 4] = 1 # A11
    U[1, 4] = 1 # A12
    V[3, 4] = 1 # B22
    # C11 += M5
    # C22 += 0? No, standard Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Wait, my M5 definition above was (A11+A12)B22.
    # Let's check standard Strassen formulas again.
    # M5 = (A11 + A12)B22
    # C11 = M1 - M4 + M5 - M7
    # C22 = M1 - M2 + M3 + M6
    # So M5 contributes to C11 only?
    # Let's verify.
    # A11 B11 + A12 B21 -> C11
    # A11 B12 + A12 B22 -> C12
    # A21 B11 + A22 B21 -> C21
    # A21 B12 + A22 B22 -> C22
    
    # M5 = (A11+A12)B22 = A11 B22 + A12 B22.
    # This term appears in C12? No.
    # In C11: A11 B11.
    # In C12: A11 B12 + A12 B22.
    # In C21: A21 B11 + A22 B21.
    # In C22: A21 B12 + A22 B22.
    
    # My M5 is A11 B22 + A12 B22.
    # A12 B22 is in C12.
    # A11 B22 is not in C11.
    
    # Let's use a known correct set of 7 products.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C11:
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # -M4 = -A22B21 + A22B11
    # +M5 = A11B22 + A12B22
    # -M7 = -A12B21 - A12B22 + A22B21 + A22B22
    # Sum:
    # A11B11: 1
    # A11B22: 1+1 = 2? No.
    # Let's re-evaluate M5 and M7 contributions.
    # This is getting error-prone. I will use a verified Strassen implementation.
    
    # Verified Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify C11 with these definitions.
    # M1: A11B11 + A11B22 + A22B11 + A22B22
    # -M4: -A22B21 + A22B11
    # +M5: A11B22 + A12B22
    # -M7: -A12B21 - A12B22 + A22B21 + A22B22
    #
    # A11B11: 1
    # A11B22: 1 + 1 = 2?
    # A22B11: 1 + 1 = 2?
    # A22B22: 1 + 1 = 2?
    # A12B22: 1 - 1 = 0
    # A22B21: -1 + 1 = 0
    # A12B21: -1
    #
    # This is wrong. C11 should be A11B11 + A12B21.
    
    # The standard Strassen formulas I recalled might be for a different indexing or I made an arithmetic error.
    # Let's use a simpler, verified set from a reliable source.
    
    # Alternative: Use the schoolbook for 2x2x2? Rank 4.
    # Then 4x4x4 rank would be 16.
    # 4x4x3 rank would be 12.
    # Total 28.
    # This is MUCH better than 93.
    # Why did I think Strassen was necessary?
    # Rank of 2x2x2 is 4 (schoolbook) or 7 (Strassen).
    # 4 < 7.
    # So Schoolbook is better for 2x2x2!
    
    # Let's re-evaluate.
    # Rank(M_2,2,2) = 4.
    # Rank(M_4,4,4) using kron of two schoolbook 2x2x2 is 4*4 = 16.
    # Rank(M_4,4,3) using kron of schoolbook 2x2x2 and 2x2x1?
    # M_4,4,2 = kron(M_2,2,2, M_2,2,1). Rank 4*4 = 16.
    # M_4,4,1 = kron(M_2,2,1, M_2,2,1). Rank 4*4 = 16.
    # Total M_4,4,3 rank = 16 + 16 = 32.
    # Total M_4,4,7 rank = 16 + 32 = 48.
    
    # This is 48 vs 112.
    # Is 48 minimal?
    # Lower bound for 4x4x4 is 16?
    # Actually, the rank of M_4,4,4 is known to be 16?
    # No, the rank of M_n,n,n is at least n^2.
    # For n=2, rank is 4.
    # For n=4, rank is at least 16.
    # The tensor product gives 16.
    # So 16 is likely optimal for 4x4x4.
    
    # So using Schoolbook for all base blocks is better.
    
    # Let's implement Schoolbook for 2x2x2 and 2x2x1.
    
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((4, 4), dtype=int)
    W = np.zeros((4, 4), dtype=int)
    
    # Schoolbook 2x2x2
    # M_ij = A_ik B_kj
    # U[i*2+j, c] = 1
    # V[j*2+k, c] = 1
    # W[k*2+i, c] = 1
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, c] = 1
                V[j*2+k, c] = 1
                W[k*2+i, c] = 1
                c += 1
                
    return U, V, W

def _schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    c = 0
    for i in range(2):
        for j in range(2):
            # k is always 0
            U[i*2+j, c] = 1
            V[j*1+0, c] = 1
            W[0*2+i, c] = 1
            c += 1
            
    return U, V, W

def _kron(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two decompositions.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            # Iterate over all indices to fill U, V, W
            # This is O(n^2 m^2 p^2) which is small for 4x4x4
            for i1 in range(n1):
                for j1 in range(m1):
                    for k1 in range(p1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    
                                    u1 = U1[i1 * m1 + j1, r1_idx]
                                    v1 = V1[j1 * p1 + k1, r1_idx]
                                    w1 = W1[k1 * n1 + i1, r1_idx]
                                    
                                    u2 = U2[i2 * m2 + j2, r2_idx]
                                    v2 = V2[j2 * p2 + k2, r2_idx]
                                    w2 = W2[k2 * n2 + i2, r2_idx]
                                    
                                    U[i * m + j, r_idx] += u1 * u2
                                    V[j * p + k, r_idx] += v1 * v2
                                    W[k * n + i, r_idx] += w1 * w2
                                    
    return U, V, W

def construct():
    # M_4,4,4
    U222, V222, W222 = _schoolbook_222()
    U444, V444, W444 = _kron(U222, V222, W222, U222, V222, W222, 2, 2, 2, 2, 2, 2)
    
    # M_4,4,2
    U221, V221, W221 = _schoolbook_221()
    U442, V442, W442 = _kron(U222, V222, W222, U221, V221, W221, 2, 2, 2, 2, 2, 1)
    
    # M_4,4,1
    U441, V441, W441 = _kron(U221, V221, W221, U221, V221, W221, 2, 2, 1, 2, 2, 1)
    
    # Combine M_4,4,2 and M_4,4,1 to get M_4,4,3
    U443 = np.hstack([U442, U441])
    V443 = np.hstack([V442, V441])
    W443 = np.hstack([W442, W441])
    
    # Map V443 to global indices
    # V442 is for p=2. Rows j*2+k. Map to j*7+k.
    # V441 is for p=1. Rows j*1+0. Map to j*7+2.
    V443_global = np.zeros((4 * 7, U443.shape[1]), dtype=int)
    
    # Process V442
    for j in range(4):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + k
            V443_global[global_idx, :V442.shape[1]] = V442[local_idx, :]
            
    # Process V441
    for j in range(4):
        local_idx = j
        global_idx = j * 7 + 2
        V443_global[global_idx, V442.shape[1]:] = V441[local_idx, :]
        
    # Map W443 to global indices
    # W442 is for p=2. Rows k*4+i. Map to (4+k)*4+i.
    # W441 is for p=1. Rows 0*4+i = i. Map to (4+0)*4+i = 16+i.
    W443_global = np.zeros((7 * 4, U443.shape[1]), dtype=int)
    
    # Process W442
    for k in range(2):
        for i in range(4):
            local_idx = k * 4 + i
            global_idx = (4 + k) * 4 + i
            W443_global[global_idx, :W442.shape[1]] = W442[local_idx, :]
            
    # Process W441
    for i in range(4):
        local_idx = i
        global_idx = 16 + i
        W443_global[global_idx, W442.shape[1]:] = W441[local_idx, :]
        
    # Final Assembly
    U_total = np.hstack([U444, U443])
    V_total = np.hstack([V444, V443_global])
    W_total = np.hstack([W444, W443_global])
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
# EVOLVE-BLOCK-END
