<<<<<<< SEARCH
def _strassen_4x4x4():
    """
    Return a rank-49 decomposition for 4x4x4 matrix multiplication.
    This uses the known Strassen-based decomposition for 4x4x4.
    """
    n = m = p = 4
    # Strassen's algorithm for 4x4x4 achieves rank 49
    # We'll use a known construction based on Strassen's 7-multiplication method
    # extended to 4x4 blocks
    
    # For 2x2x2, Strassen uses 7 multiplications
    # For 4x4x4, we can use a block decomposition
    
    # Actually, let me use a simpler approach: use the known fact that
    # R(4,4,4) <= 49 via specific constructions.
    
    # Let me construct it using the bilinear form approach.
    # For 4x4x4, one known rank-49 decomposition exists.
    
    # For now, let's use a block approach: split 4x4x4 into blocks
    # and use Strassen on 2x2x2 blocks.
    
    # 4x4x4 can be seen as 2x2 blocks of 2x2 matrices
    # Using Strassen on each pair of 2x2 blocks gives rank 7*4 = 28? No...
    
    # Let me just use schoolbook for now and note that rank 49 is achievable.
    # Actually, let me try to construct a rank-49 solution.
    
    # Known result: R(4,4,4) = 49 (proven by Strassen)
    # Construction: Use the 7-term Strassen formula for 2x2x2,
    # then extend to 4x4x4 using block structure.
    
    # For simplicity and correctness, let's use a verified construction.
    # I'll use the approach of decomposing 4x4x4 using 49 terms.
    
    # Actually, let me just return schoolbook (rank 64) for safety,
    # and focus on the overall 4x4x9 decomposition.
    
    # Wait - let me try to actually implement Strassen for 4x4x4.
    # The 4x4x4 tensor can be decomposed using the following approach:
    # Split into 2x2 blocks, use Strassen's 7 multiplications for each
    # combination of block positions.
    
    # For 4x4x4 with 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = A*B, so Cij = sum_k Aik * Bkj
    
    # Using Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we have 4x4 = 2x2 blocks of 2x2
    # The product involves all combinations of 2x2 block multiplications
    # Each 2x2x2 multiplication uses 7 terms, and there are 4 such
    # "positions" in the 4x4 product (since C has 4 block positions,
    # each requiring a sum of 2 block products)
    
    # Actually, C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a sum of two 2x2x2 multiplications.
    # Using Strassen for each 2x2x2: 7 terms each, 8 such multiplications total
    # = 56 terms. But we can share some computations.
    
    # Actually, the standard result is that R(4,4,4) = 49.
    # Let me look up the exact construction...
    
    # For now, let me use a different approach: I'll implement a search
    # for a low-rank decomposition using optimization.
    
    # Actually, let me just use schoolbook for the 4x4x4 part (rank 64)
    # and focus on getting the overall rank as low as possible.
    
    return _schoolbook(n, m, p)
=======
def _strassen_2x2x2():
    """
    Return a rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U entries: coefficients for A
    # V entries: coefficients for B
    # W entries: coefficients for C (sum of products)
    
    U = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 1, 0, 0, 0, -1, 1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 1, 0, -1]  # A11
    ]
    
    V = [
        [1, 1, 1, -1, 0, 1, 1], # B00
        [0, 0, 1, 0, 1, 1, 0],  # B01
        [1, 0, -1, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W: C = M1 - M4 + M5 ... etc?
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Wait, let's verify standard Strassen W coefficients.
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # The tensor decomposition is sum_r U[:,r] V[:,r] W[:,r]^T ?
    # No, the decomposition is T = sum_r u_r v_r w_r.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] ? No.
    # The multiplication tensor T_{i,j,k} = 1 if A_{i,k} B_{k,j} contributes to C_{i,j}.
    # Indices: i (row C), j (col C), k (sum index).
    # U corresponds to A (i, k). V corresponds to B (k, j). W corresponds to C (i, j).
    # So U shape (n*m, rank) -> (2*2, 7). Index i*m+k.
    # V shape (m*p, rank) -> (2*2, 7). Index k*p+j.
    # W shape (p*n, rank) -> (2*2, 7). Index j*n+i.
    
    # Let's map indices:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(0), 01(1), 10(2), 11(3)
    # C: 00(0), 01(1), 10(2), 11(3)
    
    # U rows:
    # r=0: M1. A00(1), A11(1). Others 0.
    # r=1: M2. A10(1), A11(1). Others 0.
    # r=2: M3. A00(1). Others 0.
    # r=3: M4. A11(1). Others 0.
    # r=4: M5. A00(1), A01(1). Others 0.
    # r=5: M6. A10(1), A00(-1). Others 0.
    # r=6: M7. A01(1), A11(-1). Others 0.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    # V rows:
    # r=0: M1. B00(1), B11(1).
    # r=1: M2. B00(1).
    # r=2: M3. B01(1), B11(-1).
    # r=3: M4. B10(1), B00(-1).
    # r=4: M5. B11(1).
    # r=5: M6. B00(1), B01(1).
    # r=6: M7. B10(1), B11(1).
    
    V = [
        [1, 1, -1, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W rows (C coefficients):
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    W = [
        [1, 0, 0, -1, 1, 0, 0],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Return a rank-56 decomposition for 4x4x4 matrix multiplication
    by applying Strassen 2x2x2 to each of the 8 block products.
    """
    n = m = p = 4
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    base = _strassen_2x2x2()
    
    # Block indices I, J, K in {0, 1}
    # Local indices i', j', k' in {0, 1}
    # Global i = 2*I + i', j = 2*J + j', k = 2*K + k'
    
    # The 8 block products are:
    # 1. A00 B00 -> C00
    # 2. A00 B01 -> C01
    # 3. A01 B10 -> C00
    # 4. A01 B11 -> C01
    # 5. A10 B00 -> C10
    # 6. A10 B01 -> C11
    # 7. A11 B10 -> C10
    # 8. A11 B11 -> C11
    
    # We iterate through these 8 configurations
    configs = [
        (0, 0, 0, 0), # I=0, K=0, J=0 -> C00
        (0, 0, 1, 0), # I=0, K=0, J=1 -> C01
        (0, 1, 0, 0), # I=0, K=1, J=0 -> C00
        (0, 1, 1, 0), # I=0, K=1, J=1 -> C01
        (1, 0, 0, 0), # I=1, K=0, J=0 -> C10
        (1, 0, 1, 0), # I=1, K=0, J=1 -> C11
        (1, 1, 0, 0), # I=1, K=1, J=0 -> C10
        (1, 1, 1, 0)  # I=1, K=1, J=1 -> C11
    ]
    
    # Note: The standard matrix multiplication C = A B has C[I, J] = sum_K A[I, K] B[K, J].
    # So the block product is A[I, K] * B[K, J].
    # My configs above list (I, K, J, dummy).
    # Let's re-verify the mapping.
    # A block index (I, K). B block index (K, J). C block index (I, J).
    
    # For each of the 8 (I, K, J) combinations, we add a Strassen 2x2x2 decomposition.
    # Offset in rank: idx * 7
    
    for idx, (I, K, J, _) in enumerate(configs):
        offset = idx * 7
        
        # A indices: i = 2*I + i', k = 2*K + k'
        # B indices: k = 2*K + k', j = 2*J + j'
        # C indices: i = 2*I + i', j = 2*J + j'
        
        for r in range(7):
            # U: A part
            for ip in range(2):
                for kp in range(2):
                    u_val = base['U'][ip * 2 + kp][r]
                    if u_val != 0:
                        i_global = 2 * I + ip
                        k_global = 2 * K + kp
                        U[i_global * 4 + k_global][offset + r] += u_val
                        
            # V: B part
            for kp in range(2):
                for jp in range(2):
                    v_val = base['V'][kp * 2 + jp][r]
                    if v_val != 0:
                        k_global = 2 * K + kp
                        j_global = 2 * J + jp
                        V[k_global * 4 + j_global][offset + r] += v_val
                        
            # W: C part
            for ip in range(2):
                for jp in range(2):
                    w_val = base['W'][jp * 2 + ip][r] # W index is j*n + i = j*2 + i
                    if w_val != 0:
                        i_global = 2 * I + ip
                        j_global = 2 * J + jp
                        W[j_global * 4 + i_global][offset + r] += w_val
                        
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Now update `_construct_4x4x9` to use the new `_strassen_4x4x4` and split 9 into 4+4+1 for better rank.

