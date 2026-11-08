"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 4 blocks of 2x2x2 Strassen decomposition
    # Total rank = 4 * 7 = 28
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each step r defines:
    # u: vector for i (size 2)
    # v: vector for j (size 2)
    # w: vector for k (size 2)
    # Such that sum(u_i * v_j * w_k) contributes to C_ik
    # We define the 7 multiplications P1..P7
    # And their contributions to C00, C01, C10, C11
    
    # P1 = A00 * (B00 - B10)
    # P2 = A10 * (B00 + B10)
    # P3 = (A00 + A01) * B11
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A10) * B01
    # P6 = (A01 - A00) * (B10 + B11)
    # P7 = (A10 - A11) * (B00 + B01)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to U, V, W vectors.
    # U corresponds to A, V to B, W to C.
    # Term r is u_r (i) * v_r (j) * w_r (k)
    
    # Step 1: P1
    # u = [1, 0] (A00)
    # v = [1, -1] (B00 - B10) -> j0=1, j1=-1
    # w = [1, 0] (contributes to C00) -> k0=1, k1=0
    # Wait, W is for C. The term is u_i v_j w_k.
    # So for P1, it contributes to C00.
    # So w = [1, 0] (k0=1, k1=0).
    
    strassen_steps = [
        # (u_i, u_i+1), (v_j, v_j+1), (w_k, w_k+1)
        # P1
        ([1, 0], [1, -1], [1, 0]),
        # P2
        ([0, 1], [1, 1], [0, 1]), # A10, B00+B10, contributes to C10 (k1? No, C10 is i=1, k=0)
        # Let's re-verify C indices.
        # C_ik. i is row of A, k is col of B.
        # C10 means i=1, k=0.
        # So w should be [0, 1] for k=1? No.
        # w_k is the coefficient for output k.
        # If it contributes to C10, then k=0. So w=[1, 0] for k=0, k=1?
        # w vector is length 2 (local k).
        # w[0] is coeff for local k=0. w[1] is coeff for local k=1.
        # So for C10 (i=1, k=0), w=[1, 0].
        
        # Let's list carefully.
        # P1: A00(B00-B10). u=[1,0], v=[1,-1]. Contributes to C00. w=[1,0].
        # P2: A10(B00+B10). u=[0,1], v=[1,1]. Contributes to C10. w=[1,0].
        # P3: (A00+A01)B11. u=[1,1], v=[0,1]. Contributes to C01 and C11.
        #    C01: i=0, k=1. w=[0,1].
        #    C11: i=1, k=1. w=[0,1].
        #    So w=[0,1].
        # P4: A11(B01-B00). u=[0,1], v=[-1,1]. Contributes to C00 and C11.
        #    C00: i=0, k=0. w=[1,0].
        #    C11: i=1, k=1. w=[0,1].
        #    This term contributes to TWO outputs?
        #    No, in the tensor decomposition, each rank-1 term is a single product.
        #    The sum of these products yields the matrix.
        #    So w is a vector. If w=[1,1], it adds to both C00 and C11?
        #    No, w is the vector for the third mode.
        #    $T_{ijk} = \sum u_i v_j w_k$.
        #    If w=[1,1], then $T_{ij0}$ gets $u_i v_j$ and $T_{ij1}$ gets $u_i v_j$.
        #    So yes, one rank-1 term can contribute to multiple k's.
        
        # Let's re-evaluate P4.
        # P4 = A11(B01-B00).
        # It contributes to C00?
        # C00 = P1 + P4 - P5 + P7.
        # So P4 contributes to C00 with coeff 1.
        # It contributes to C11?
        # C11 = P1 - P2 + P3 + P6.
        # Wait, P4 is NOT in C11 in standard Strassen?
        # Standard Strassen:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P6
        # C11 = P1 - P2 + P3 + P6
        # Let's check P4. P4 is in C00. Is it in C11? No.
        # My previous list said P4 contributes to C00 and C11. That was wrong.
        
        # Let's stick to the formulas:
        # P1: u=[1,0], v=[1,-1]. In C00 (+1), C11 (+1).
        #    Wait, C11 has P1. So P1 contributes to C00 and C11.
        #    w for P1: contributes to C00 (k=0) and C11 (k=1).
        #    So w=[1, 1].
        
        # P2: u=[0,1], v=[1,1]. In C10 (+1), C11 (-1).
        #    w=[1, -1].
        
        # P3: u=[1,1], v=[0,1]. In C01 (+1), C11 (+1).
        #    w=[1, 1].
        
        # P4: u=[0,1], v=[-1,1]. In C00 (+1).
        #    w=[1, 0].
        
        # P5: u=[1,1], v=[0,1]. In C00 (-1), C01 (+1).
        #    w=[-1, 1].
        
        # P6: u=[1,-1], v=[1,1]. In C10 (+1), C11 (+1).
        #    w=[1, 1].
        
        # P7: u=[0,1], v=[1,1]. In C00 (+1).
        #    w=[1, 0].
        
        # Let's verify C00 = P1 + P4 - P5 + P7
        # P1 (w0=1) + P4 (w0=1) - P5 (w0=-1 -> -(-1)=1? No, coeff is -1 in formula)
        # Formula: C00 = 1*P1 + 1*P4 - 1*P5 + 1*P7.
        # My w vectors are the coefficients in the sum.
        # So for P5, the coefficient in C00 is -1.
        # So w5[0] should be -1.
        # For P5 in C01, coeff is +1. w5[1] = 1.
        # So w5 = [-1, 1]. Correct.
        
        # Let's re-verify all w vectors based on formulas.
        # C00: P1(+1), P4(+1), P5(-1), P7(+1).
        # C01: P3(+1), P5(+1).
        # C10: P2(+1), P6(+1).
        # C11: P1(+1), P2(-1), P3(+1), P6(+1).
        
        # P1: C00(+1), C11(+1). w=[1, 1].
        # P2: C10(+1), C11(-1). w=[1, -1].
        # P3: C01(+1), C11(+1). w=[1, 1].
        # P4: C00(+1). w=[1, 0].
        # P5: C00(-1), C01(+1). w=[-1, 1].
        # P6: C10(+1), C11(+1). w=[1, 1].
        # P7: C00(+1). w=[1, 0].
    ]
    
    # Define the steps explicitly
    steps = [
        # (u, v, w)
        ([1, 0], [1, -1], [1, 1]),  # P1
        ([0, 1], [1, 1], [1, -1]),  # P2
        ([1, 1], [0, 1], [1, 1]),   # P3
        ([0, 1], [-1, 1], [1, 0]),  # P4
        ([1, 1], [0, 1], [-1, 1]),  # P5
        ([1, -1], [1, 1], [1, 1]),  # P6
        ([0, 1], [1, 1], [1, 0])    # P7
    ]
    
    # Blocks: (j_offset, k_offset)
    blocks = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for u, v, w in steps:
            # u is for i (0,1)
            # v is for local j (0,1) -> global j = j_off + local_j
            # w is for local k (0,1) -> global k = k_off + local_k
            
            for i in range(2):
                if u[i] != 0:
                    row_u = i * m + j_off + 0 # Wait, U index is i*m + j?
                    # U shape (n*m, rank). Index is i*m + j?
                    # No, U corresponds to A. A is n x m.
                    # Index in U is i*m + j?
                    # In the tensor T[i,j,k], i is A row, j is A col / B row.
                    # So U index is i*m + j.
                    # But here u[i] is the coefficient for A[i, local_j].
                    # So for a fixed step, u[i] multiplies A[i, local_j] for each local_j?
                    # No, v[local_j] is the coefficient for B[local_j, ...].
                    # The term is sum_i sum_j sum_k u_i v_j w_k A_{i,j} B_{j,k}.
                    # So for a fixed step (rank-1 term), we have:
                    # U[i*m + j] = u[i] * v[j] ?
                    # No.
                    # The decomposition is T = sum_r U_r * V_r * W_r.
                    # U_r is a vector of size n*m.
                    # V_r is a vector of size m*p.
                    # W_r is a vector of size p*n.
                    
                    # For this rank-1 term, the contribution to T[i,j,k] is U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]?
                    # No, the standard is T[i,j,k] = sum_r U_r[i] V_r[j] W_r[k]?
                    # If U_r is size n, V_r size m, W_r size p.
                    # But here U is size n*m.
                    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
                    # This implies the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
                    # And the decomposition is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
                    # Let's check the schoolbook example.
                    # U[i*m+j, col] = 1. V[j*p+k, col] = 1. W[k*n+i, col] = 1.
                    # Sum over col: U[i*m+j] V[j*p+k] W[k*n+i] = 1*1*1 = 1.
                    # This matches T[i,j,k]=1 for the schoolbook term (i,j,k).
                    # So yes, the indices are flattened.
                    
                    # So for our Strassen term:
                    # We need to fill U, V, W such that:
                    # U[i*m + j, col] * V[j*p + k, col] * W[k*n + i, col]
                    # sums to the Strassen contribution.
                    
                    # In Strassen, the term is u_i * v_j * w_k * A_{i, local_j} * B_{local_j, local_k}.
                    # Note: A_{i, local_j} corresponds to index i*m + (j_off + local_j).
                    # B_{local_j, local_k} corresponds to index (j_off + local_j)*p + (k_off + local_k).
                    # C_{i, local_k} corresponds to index (k_off + local_k)*n + i.
                    
                    # So for this column:
                    # For each i in 0..1:
                    #   For each local_j in 0..1:
                    #     j_global = j_off + local_j
                    #     U[i*m + j_global, col] = u[i] * v[local_j]
                    #   For each local_k in 0..1:
                    #     k_global = k_off + local_k
                    #     V[j_global*p + k_global, col] = v[local_j] * w[local_k] ?
                    #     Wait. V is indexed by j*p + k.
                    #     The term in the product is V[j*p+k].
                    #     In the sum, we sum over r.
                    #     T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                    #     We want this to equal sum_steps u_i v_j w_k (for the specific i,j,k).
                    #     So for a specific step, we can set:
                    #     U[i*m+j, col] = u[i]
                    #     V[j*p+k, col] = v[j_local] w[k_local] ?
                    #     No, V depends on j and k.
                    #     If we set U = u[i], V = v[j_local] * w[k_local], W = 1?
                    #     Then product is u[i] * v[j] w[k] * 1.
                    #     But W is indexed by k*n+i.
                    #     If W[k*n+i] = 1, then it works?
                    #     Let's check.
                    #     U[i*m+j] = u[i].
                    #     V[j*p+k] = v[j_local] * w[k_local].
                    #     W[k*n+i] = 1.
                    #     Product = u[i] v[j] w[k].
                    #     This matches!
                    
                    # So:
                    # U[i*m + j_global, col] = u[i]
                    # V[j_global*p + k_global, col] = v[j_local] * w[k_local]
                    # W[k_global*n + i, col] = 1
                    
                    # Wait, is this valid?
                    # V[j*p+k] must be consistent for all i?
                    # Yes, V is shared for all i.
                    # But in the product U[i*m+j] V[j*p+k] W[k*n+i],
                    # V[j*p+k] is the same for all i.
                    # In my assignment, V[j*p+k] = v[j_local] w[k_local].
                    # This does not depend on i. So it is consistent.
                    # U[i*m+j] = u[i]. Depends on i.
                    # W[k*n+i] = 1. Depends on i.
                    # So the product is u[i] * (v[j] w[k]) * 1.
                    # This is correct.
                    
                    pass # Placeholder for logic below
            
            # Fill U
            for i in range(2):
                if u[i] != 0:
                    for local_j in range(2):
                        j_global = j_off + local_j
                        U[i * m + j_global][col] = u[i]
                        
            # Fill V
            for local_j in range(2):
                if v[local_j] != 0:
                    j_global = j_off + local_j
                    for local_k in range(2):
                        if w[local_k] != 0:
                            k_global = k_off + local_k
                            V[j_global * p + k_global][col] = v[local_j] * w[local_k]
                            
            # Fill W
            for local_k in range(2):
                if w[local_k] != 0:
                    k_global = k_off + local_k
                    for i in range(2):
                        W[k_global * n + i][col] = 1
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
