"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Indexing:
    # U[i*m + j] for i in [0,n), j in [0,m)
    # V[j*p + k] for j in [0,m), k in [0,p)
    # W[k*n + i] for k in [0,p), i in [0,n)
    
    # We'll build the decomposition in blocks:
    # Block 1: First 2 columns of B (k=0,1) -> use Strassen for 2x2 part + schoolbook for 2x1
    # Block 2: Remaining 8 columns (k=2..9) -> schoolbook
    
    # Let's use a list-based approach to accumulate terms
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are dicts or lists of nonzero entries."""
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # --- Block 1: A (2x3) * B1 (3x2) ---
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A * B1 (2x2)
    # 
    # Partition: A1 = [[a00, a01], [a10, a11]] (2x2), A2 = [[a02], [a12]] (2x1)
    #             B11 = [[b00, b01], [b10, b11]] (2x2), B12 = [[b20, b21]] (1x2)
    # C1 = A1*B11 + A2*B12
    #
    # Strassen for A1*B11 (7 multiplications):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6
    
    # U indices: U[0]=a00, U[1]=a01, U[2]=a02, U[3]=a10, U[4]=a11, U[5]=a12
    # V indices for B1: V[0]=b00, V[1]=b01, V[10]=b10, V[11]=b11, V[20]=b20, V[21]=b21
    # W indices for C1: W[0]=C00, W[1]=C01, W[10]=C10, W[11]=C11
    
    # M1 = (a00+a11)*(b00+b11)
    add_term({0:1, 4:1}, {0:1, 11:1}, {0:1, 11:1})
    
    # M2 = (a10+a11)*b00
    add_term({3:1, 4:1}, {0:1}, {10:1, 11:-1})
    
    # M3 = a00*(b01-b11)
    add_term({0:1}, {1:1, 11:-1}, {1:1})
    
    # M4 = a11*(b10-b00)
    add_term({4:1}, {10:1, 0:-1}, {10:1})
    
    # M5 = (a00+a01)*b11
    add_term({0:1, 1:1}, {11:1}, {0:-1, 1:1})
    
    # M6 = (a10-a00)*(b00+b01)
    add_term({3:1, 0:-1}, {0:1, 1:1}, {11:1})
    
    # M7 = (a01-a11)*(b10+b11)
    add_term({1:1, 4:-1}, {10:1, 11:1}, {0:1})
    
    # Schoolbook for A2*B12 (4 terms):
    # C00 += a02*b20
    add_term({2:1}, {20:1}, {0:1})
    # C01 += a02*b21
    add_term({2:1}, {21:1}, {1:1})
    # C10 += a12*b20
    add_term({5:1}, {20:1}, {10:1})
    # C11 += a12*b21
    add_term({5:1}, {21:1}, {11:1})
    
    # --- Block 2: A (2x3) * B2 (3x8) where B2 has columns k=2..9 ---
    # Schoolbook: rank = 2*3*8 = 48
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                add_term({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
    
    # Now convert lists of dicts to full matrices
    rank = len(U_list)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col in range(rank):
        for idx, val in U_list[col].items():
            U[idx][col] = val
        for idx, val in V_list[col].items():
            V[idx][col] = val
        for idx, val in W_list[col].items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
