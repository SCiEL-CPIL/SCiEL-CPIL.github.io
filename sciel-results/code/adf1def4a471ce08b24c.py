import time

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to set U, V, W for a term
    # u_idx: index in U (i*m+j)
    # v_idx: index in V (j*p+k)
    # w_idx: index in W (k*n+i)
    # coeff: value (1, -1, etc.)
    
    def add_term(u_idx, coeff_u, v_idx, coeff_v, w_idx, coeff_w):
        nonlocal col
        U[u_idx][col] = coeff_u
        V[v_idx][col] = coeff_v
        W[w_idx][col] = coeff_w
        col += 1

    # Strassen blocks for b in 0..5
    # A indices:
    # A00: 0, A01: 1, A10: 2, A11: 3
    # B indices for block b:
    # B0, 2b: 0*13 + 2b = 2b
    # B0, 2b+1: 0*13 + 2b + 1 = 2b + 1
    # B1, 2b: 1*13 + 2b = 13 + 2b
    # B1, 2b+1: 1*13 + 2b + 1 = 13 + 2b + 1
    # W indices for block b:
    # C0, 2b: (2b)*2 + 0 = 4b
    # C0, 2b+1: (2b+1)*2 + 0 = 4b + 2
    # C1, 2b: (2b)*2 + 1 = 4b + 1
    # C1, 2b+1: (2b+1)*2 + 1 = 4b + 3
    
    for b in range(6):
        # M1: (A00+A11)(B0,2b + B1,2b+1)
        # U: A00(1), A11(1)
        # V: B0,2b(1), B1,2b+1(1)
        # W: C0,2b(1), C1,2b+1(1)
        # Note: A term is sum of two U entries. So we need to represent the linear form.
        # The decomposition is sum of rank 1 tensors.
        # A rank 1 tensor is u_i v_j w_k.
        # If the term is (A00+A11) * (B0,2b+B1,2b+1), this is NOT a rank 1 tensor.
        # It is a product of sums.
        # Wait, Strassen's algorithm works by precomputing linear combinations.
        # The "multiplication" step is scalar multiplication of these combinations.
        # So, in the tensor decomposition context:
        # Term r corresponds to $M_r = L_{A,r} \cdot L_{B,r}$.
        # The contribution to $C$ is $M_r \cdot W_{r}$.
        # So the tensor is $\sum_r (L_{A,r} \otimes L_{B,r} \otimes W_r)$.
        # Each $L_{A,r}$ is a vector in $A$ space. Each $L_{B,r}$ is a vector in $B$ space.
        # So $U$ column $r$ contains the coefficients of $L_{A,r}$.
        # $V$ column $r$ contains the coefficients of $L_{B,r}$.
        # $W$ column $r$ contains the coefficients of $W_r$ (which adds to specific $C$ entries).
        
        # So for M1:
        # U col: [1, 0, 0, 1] (A00 + A11)
        # V col: [1, 0, ..., 1, ...] at indices 2b and 13+2b+1
        # W col: [1, 0, 0, 1, ...] at indices 4b and 4b+3
        
        # Let's implement a helper to set a column
        def set_column(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] = val
            col += 1

        # M1
        u1 = {0: 1, 3: 1}
        v1 = {2*b: 1, 13 + 2*b + 1: 1}
        w1 = {4*b: 1, 4*b + 3: 1}
        set_column(u1, v1, w1)
        
        # M2: (A10+A11) B0,2b
        u2 = {2: 1, 3: 1}
        v2 = {2*b: 1}
        w2 = {4*b + 1: 1, 4*b + 3: -1} # C1,2b += M2, C1,2b+1 -= M2
        set_column(u2, v2, w2)
        
        # M3: A00 (B0,2b+1 - B1,2b+1)
        u3 = {0: 1}
        v3 = {2*b + 1: 1, 13 + 2*b + 1: -1}
        w3 = {4*b + 2: 1, 4*b + 3: 1} # C0,2b+1 += M3, C1,2b+1 += M3
        set_column(u3, v3, w3)
        
        # M4: A11 (B1,2b - B0,2b)
        u4 = {3: 1}
        v4 = {13 + 2*b: 1, 2*b: -1}
        w4 = {4*b: 1, 4*b + 1: 1} # C0,2b += M4, C1,2b += M4
        set_column(u4, v4, w4)
        
        # M5: (A00+A01) B1,2b+1
        u5 = {0: 1, 1: 1}
        v5 = {13 + 2*b + 1: 1}
        w5 = {4*b: -1, 4*b + 2: 1} # C0,2b -= M5, C0,2b+1 += M5
        set_column(u5, v5, w5)
        
        # M6: (A10-A00) (B0,2b + B0,2b+1)
        u6 = {2: 1, 0: -1}
        v6 = {2*b: 1, 2*b + 1: 1}
        w6 = {4*b + 3: 1} # C1,2b+1 += M6
        set_column(u6, v6, w6)
        
        # M7: (A01-A11) (B1,2b + B1,2b+1)
        u7 = {1: 1, 3: -1}
        v7 = {13 + 2*b: 1, 13 + 2*b + 1: 1}
        w7 = {4*b: 1} # C0,2b += M7
        set_column(u7, v7, w7)

    # Schoolbook for last block (column 12)
    # B indices: 12, 13+12=25
    # W indices: 12*2+0=24, 12*2+1=25
    
    # T1: A00 B0,12 -> C0,12
    u_t1 = {0: 1}
    v_t1 = {12: 1}
    w_t1 = {24: 1}
    set_column(u_t1, v_t1, w_t1)
    
    # T2: A01 B1,12 -> C0,12
    u_t2 = {1: 1}
    v_t2 = {25: 1}
    w_t2 = {24: 1}
    set_column(u_t2, v_t2, w_t2)
    
    # T3: A10 B0,12 -> C1,12
    u_t3 = {2: 1}
    v_t3 = {12: 1}
    w_t3 = {25: 1}
    set_column(u_t3, v_t3, w_t3)
    
    # T4: A11 B1,12 -> C1,12
    u_t4 = {3: 1}
    v_t4 = {25: 1}
    w_t4 = {25: 1}
    set_column(u_t4, v_t4, w_t4)

    return {"U": U, "V": V, "W": W}
