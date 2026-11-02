import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Build decomposition with rank 50
    # 3x5x6 = (3x2x6) + (3x3x6)
    # 3x2x6 = 2*(3x2x3), each 3x2x3 = (2x2x3)+(1x2x3), 2x2x3=(2x2x2)+(2x2x1), 1x2x3 rank 2
    # 3x3x6 = 3*(3x3x2), each 3x3x2 = 2x2x2 + 1x2x2 + 2x1x2 + 1x1x2
    
    # Collect all (u_indices, v_indices, w_indices, u_vals, v_vals, w_vals) tuples
    terms = []
    
    def add_term(i_idx, j_idx, k_idx, u_val, v_val, w_val):
        """Add a rank-1 term. i_idx is index in [0,n), j_idx in [0,m), k_idx in [0,p)."""
        terms.append((i_idx, j_idx, k_idx, u_val, v_val, w_val))
    
    def add_strassen(i0, j0, k0):
        """Add Strassen 2x2x2 decomposition at offset (i0, j0, k0)."""
        # u: (a11,a12,a21,a22) -> indices (i0,j0),(i0,j0+1),(i0+1,j0),(i0+1,j0+1)
        # v: (b11,b12,b21,b22) -> indices (j0,k0),(j0,k0+1),(j0+1,k0),(j0+1,k0+1)
        # w: (C11,C12,C21,C22) -> indices (k0,i0),(k0,i0+1),(k0+1,i0),(k0+1,i0+1)
        # But w is indexed as W[k*n+i], so (k,i) -> k*n+i
        
        strassen_u = [
            (1, 0, 0, 1),  # M1
            (0, 0, 1, 1),  # M2
            (1, 0, 0, 0),  # M3
            (0, 0, 0, 1),  # M4
            (1, 1, 0, 0),  # M5
            (0, 0, 1, -1), # M6
            (0, 1, 0, -1), # M7
        ]
        strassen_v = [
            (1, 1, 0, 0),  # M1
            (1, 0, 0, 0),  # M2
            (0, 1, 0, -1), # M3
            (0, 0, 1, -1), # M4
            (0, 0, 0, 1),  # M5
            (1, 1, 0, 0),  # M6
            (0, 0, 1, 1),  # M7
        ]
        strassen_w = [
            (1, 0, 0, 1),  # M1
            (0, 0, 1, -1), # M2
            (0, 1, 0, 1),  # M3
            (1, 0, 1, 0),  # M4
            (-1, 1, 0, 0),# M5
            (0, 0, 0, 1),  # M6
            (1, 0, 0, 0),  # M7
        ]
        
        for c in range(7):
            for a in range(4):
                if strassen_u[c][a] != 0:
                    # Map a to (di, dj)
                    if a == 0: di, dj = 0, 0
                    elif a == 1: di, dj = 0, 1
                    elif a == 2: di, dj = 1, 0
                    else: di, dj = 1, 1
                    i_idx = i0 + di
                    j_idx = j0 + dj
                    # Need to also specify v and w, but they're coupled
                    # This approach won't work directly since it's not a single (i,j,k)
                    pass
        
        # Better approach: for each of the 7 Strassen terms, it's a sum over 4 (i,j) pairs
        # Each Strassen term c contributes:
        # sum_a strassen_u[c][a] * A[i0+di_a, j0+dj_a] * sum_b strassen_v[c][b] * B[j0+dj_b, k0+dk_b] * sum_d strassen_w[c][d] * C[k0+dk_d, i0+di_d]
        # This is NOT a rank-1 term in the (i,j,k) tensor! It's a product of sums.
        # 
        # Actually, the Strassen decomposition IS a rank-7 decomposition of the 2x2x2 tensor.
        # Each of the 7 terms is a rank-1 tensor in the 2x2x2 space.
        # So for term c: u_c ⊗ v_c ⊗ w_c where u_c is a vector in R^4, v_c in R^4, w_c in R^4.
        # This means for each c, we have a single rank-1 term with:
        # U entries: for each (i,j) in the 2x2 block, U[i,j,c] = strassen_u[c][a] where a corresponds to (i,j)
        # V entries: for each (j,k) in the 2x2 block, V[j,k,c] = strassen_v[c][b]
        # W entries: for each (k,i) in the 2x2 block, W[k,i,c] = strassen_w[c][d]
        
        for c in range(7):
            for a in range(4):
                if strassen_u[c][a] != 0:
                    di, dj = (a//2, a%2)  # a=0:(0,0), a=1:(0,1), a=2:(1,0), a=3:(1,1)
                    # Wait, let me use: a=0:(0,0), a=1:(0,1), a=2:(1,0), a=3:(1,1)
                    # So di = a//2, dj = a%2? No: a=0: di=0,dj=0; a=1: di=0,dj=1; a=2: di=1,dj=0; a=3: di=1,dj=1
                    # di = a//2, dj = a%2 gives: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1). Yes!
                    i_idx = i0 + di
                    j_idx = j0 + dj
                    add_term(i_idx, j_idx, None, strassen_u[c][a], None, None)
            
            # This is getting messy. Let me use a different approach.
            # I'll collect (U_entries, V_entries, W_entries) for each rank-1 term.
            pass
    
    # Let me restart with a cleaner approach.
    # Each rank-1 term is defined by:
    # - A set of (i, j) pairs with values for U
    # - A set of (j, k) pairs with values for V  
    # - A set of (k, i) pairs with values for W
    # Such that sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = delta(i,i')delta(j,j')delta(k,k')
    
    # Actually, the standard form is: for each c, U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}
    # And the decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # So for each c, I specify:
    # U_c: vector of length n*m, mostly zeros
    # V_c: vector of length m*p, mostly zeros
    # W_c: vector of length p*n, mostly zeros
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    c = 0  # current column index
    
    def add_strassen(i0, j0, k0):
        nonlocal c
        strassen_u = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1],
        ]
        strassen_v = [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        strassen_w = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        
        for cc in range(7):
            for a in range(4):
                if strassen_u[cc][a] != 0:
                    di, dj = a//2, a%2
                    U[(i0+di)*m + (j0+dj)][c] = strassen_u[cc][a]
            for b in range(4):
                if strassen_v[cc][b] != 0:
                    dj, dk = b//2, b%2
                    V[(j0+dj)*p + (k0+dk)][c] = strassen_v[cc][b]
            for d in range(4):
                if strassen_w[cc][d] != 0:
                    dk, di = d//2, d%2
                    W[(k0+dk)*n + (i0+di)][c] = strassen_w[cc][d]
            c += 1
    
    def add_1x2x2(i0, j0, k0):
        """1x2x2 at offset: i fixed, j in {j0,j0+1}, k in {k0,k0+1}"""
        nonlocal c
        # u = (1), v = (1,1), w = (1,1)
        U[i0*m + j0][c] = 1
        U[i0*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1
        c += 1
    
    def add_2x1x2(i0, j0, k0):
        """2x1x2 at offset: i in {i0,i0+1}, j fixed, k in {k0,k0+1}"""
        nonlocal c
        # u = (1,1), v = (1), w = (1,1)
        U[i0*m + j0][c] = 1
        U[(i0+1)*m + j0][c] = 1
        V[j0*p + k0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + (i0+1)][c] = 1
        c += 1
    
    def add_1x1x2(i0, j0, k0):
        """1x1x2 at offset"""
        nonlocal c
        U[i0*m + j0][c] = 1
        V[j0*p + k0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1
        c += 1
    
    def add_2x2x1(i0, j0, k0):
        """2x2x1 at offset: i in {i0,i0+1}, j in {j0,j0+1}, k fixed"""
        nonlocal c
        # u = (1,1,1,1), v = (1,1,1,1), w = (1)
        # Actually 2x2x1: C[i,k] = sum_j A[i,j]*B[j,k]. For fixed k, this is 2 outputs, each sum of 2 products.
        # Rank 1? No, rank 4 for 2x2x1 schoolbook. But we said rank 1 earlier...
        # Wait, 2x2x1: T = E_0⊗E_0⊗E_0 + E_0⊗E_1⊗E_0 + E_1⊗E_0⊗E_0 + E_1⊗E_1⊗E_0
        # = (E_0⊗E_0 + E_0⊗E_1 + E_1⊗E_0 + E_1⊗E_1) ⊗ E_0
        # = (E_0+E_1)⊗(E_0+E_1)⊗E_0. Rank 1!
        # u = (1,1,1,1) for (a11,a12,a21,a22), v = (1,1,1,1) for (b11,b12,b21,b22), w = (1) for C
        # But w is for the 2x2 output block, so w = (1,1,1,1) for (C11,C12,C21,C22)? No, k is fixed.
        # For 2x2x1, the output is 2x1, so w has 2 entries.
        # T = sum_{i,j} E_i⊗E_j⊗E_0 (for the single k)
        # = (sum_i E_i) ⊗ (sum_j E_j) ⊗ E_0
        # u = (1,1) for i in {i0,i0+1}, v = (1,1) for j in {j0,j0+1}, w = (1) for k=k0
        U[i0*m + j0][c] = 1
        U[i0*m + (j0+1)][c] = 1
        U[(i0+1)*m + j0][c] = 1
        U[(i0+1)*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        V[(j0+1)*p + k0][c] = 1
        W[k0*n + i0][c] = 1
        W[k0*n + (i0+1)][c] = 1
        c += 1
    
    def add_1x2x3(i0, j0, k0):
        """1x2x3 at offset: i fixed, j in {j0,j0+1}, k in {k0,k0+1,k0+2}"""
        nonlocal c
        # 1x2x3: C[0,k] = A[0,j0]*B[j0,k] + A[0,j0+1]*B[j0+1,k] for k=k0,k0+1,k0+2
        # This is rank 2 (one for each j)
        # Term 1: j=j0, u=(1), v=(1 for j0, 0 for j0+1), w=(1,1,1) for k
        U[i0*m + j0][c] = 1
        V[j0*p + k0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        V[j0*p + (k0+2)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1
        W[(k0+2)*n + i0][c] = 1
        c += 1
        # Term 2: j=j0+1
        U[i0*m + (j0+1)][c] = 1
        V[(j0+1)*p + k0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        V[(j0+1)*p + (k0+2)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1
        W[(k0+2)*n + i0][c] = 1
        c += 1
    
    # Now build the full decomposition
    # 3x5x6 = (3x2x6) + (3x3x6)
    # Part 1: 3x2x6, i in [0,3), j in [0,2), k in [0,6)
    #   = 2*(3x2x3), k in [0,3) and [3,6)
    #   Each 3x2x3 = (2x2x3) + (1x2x3)
    #     2x2x3 = (2x2x2) + (2x2x1)
    #     1x2x3: rank 2
    
    # 3x2x3 for k in [0,3):
    #   2x2x2 at i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0)  # rank 7
    #   2x2x1 at i=0,1; j=0,1; k=2
    add_2x2x1(0, 0, 2)  # rank 1
    #   1x2x3 at i=2; j=0,1; k=0,1,2
    add_1x2x3(2, 0, 0)  # rank 2
    # Subtotal: 10, c=10
    
    # 3x2x3 for k in [3,6):
    add_strassen(0, 0, 3)  # rank 7
    add_2x2x1(0, 0, 5)  # rank 1
    add_1x2x3(2, 0, 3)  # rank 2
    # Subtotal: 10, c=20
    
    # Part 2: 3x3x6, i in [0,3), j in [2,5), k in [0,6)
    #   = 3*(3x3x2), k in [0,2), [2,4), [4,6)
    #   Each 3x3x2 = 2x2x2 + 1x2x2 + 2x1x2 + 1x1x2
    #     2x2x2 at i=0,1; j=2,3; k range
    #     1x2x2 at i=2; j=2,3; k range
    #     2x1x2 at i=0,1; j=4; k range
    #     1x1x2 at i=2; j=4; k range
    
    # 3x3x2 for k in [0,2):
    add_strassen(0, 2, 0)  # rank 7
    add_1x2x2(2, 2, 0)  # rank 1
    add_2x1x2(0, 4, 0)  # rank 1
    add_1x1x2(2, 4, 0)  # rank 1
    # Subtotal: 10, c=30
    
    # 3x3x2 for k in [2,4):
    add_strassen(0, 2, 2)  # rank 7
    add_1x2x2(2, 2, 2)  # rank 1
    add_2x1x2(0, 4, 2)  # rank 1
    add_1x1x2(2, 4, 2)  # rank 1
    # Subtotal: 10, c=40
    
    # 3x3x2 for k in [4,6):
    add_strassen(0, 2, 4)  # rank 7
    add_1x2x2(2, 2, 4)  # rank 1
    add_2x1x2(0, 4, 4)  # rank 1
    add_1x1x2(2, 4, 4)  # rank 1
    # Subtotal: 10, c=50
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
