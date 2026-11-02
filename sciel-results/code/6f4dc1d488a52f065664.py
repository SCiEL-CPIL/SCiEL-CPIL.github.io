import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 1 (for j=2) + 3*7 (Strassen for 2x2x2) + 4 (Trivial for 2x2x1) = 1 + 21 + 4 = 26
    rank = 26
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: list of (i, j, val)
        # v_vals: list of (j, k, val)
        # w_vals: list of (k, i, val)
        nonlocal r
        for i, j, val in u_vals:
            U[i * m + j, r] += val
        for j, k, val in v_vals:
            V[j * p + k, r] += val
        for k, i, val in w_vals:
            W[k * n + i, r] += val
        r += 1

    # 1. Part for j = 2 (0-based index for the 3rd column)
    # A[i, 2] * B[2, k]
    # U[i, 2] = 1
    # V[2, k] = 1
    # W[k, i] = 1
    u_list = [(i, 2, 1.0) for i in range(n)]
    v_list = [(2, k, 1.0) for k in range(p)]
    w_list = [(k, i, 1.0) for k in range(p) for i in range(n)]
    add_term(u_list, v_list, w_list)
    
    # 2. Parts for j in {0, 1}
    # Split k into 0,1; 2,3; 4,5; 6
    # Strassen for 2x2x2
    def add_strassen(k_start):
        nonlocal r
        k1 = k_start
        k2 = k_start + 1
        
        # M1 = (A11+A22)(B11+B22)
        # A indices: (0,0), (1,1)
        # B indices: (0,k1), (1,k2) -- Wait, B is 2x2 in this block.
        # B rows are j=0,1. Cols are k1,k2.
        # So B[0,k1], B[0,k2], B[1,k1], B[1,k2].
        
        # M1
        u = [(0,0,1.0), (1,1,1.0)]
        v = [(0,k1,1.0), (0,k2,1.0), (1,k1,1.0), (1,k2,1.0)]
        w = [(k, i, 1.0) for k in [k1,k2] for i in range(n)] # W depends on output k,i
        # Wait, W is for output C[i,k].
        # In Strassen, the output is a 2x2 matrix.
        # We need to map the Strassen outputs to the specific k1, k2.
        
        # Let's re-evaluate how to add Strassen terms.
        # The Strassen algorithm computes 4 values: C11, C12, C21, C22.
        # Here C11 corresponds to C[0, k1], C12 to C[0, k2], etc.
        # Each Mr is a product of linear forms in A and B.
        # The contribution to C[i,k] is sum of Mr * coeff.
        
        # So for each Mr, we add ONE term to the decomposition?
        # No, Mr is a scalar product of linear forms.
        # The term in the tensor decomposition for Mr is:
        # U_r[i,j] = coeff of A[i,j] in L_A(Mr)
        # V_r[j,k] = coeff of B[j,k] in L_B(Mr)
        # W_r[k,i] = coeff of Mr in C[i,k]
        
        # So we add 7 terms, one for each Mr.
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[0,k2]+B[1,k1]+B[1,k2])
        # L_A: A[0,0]=1, A[1,1]=1
        # L_B: B[0,k1]=1, B[0,k2]=1, B[1,k1]=1, B[1,k2]=1
        # W: C[0,k1]+=1, C[0,k2]+=1, C[1,k1]+=1, C[1,k2]+=1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k1]
        # L_A: A[1,0]=1, A[1,1]=1
        # L_B: B[0,k1]=1
        # W: C[1,k1]+=1, C[1,k2]+=1
        
        # M3 = A[0,0] * (B[0,k2]-B[1,k2])
        # L_A: A[0,0]=1
        # L_B: B[0,k2]=1, B[1,k2]=-1
        # W: C[0,k2]+=1, C[1,k2]+=1
        
        # M4 = A[1,1] * (B[1,k1]-B[0,k1])
        # L_A: A[1,1]=1
        # L_B: B[1,k1]=1, B[0,k1]=-1
        # W: C[0,k1]+=1, C[1,k1]+=1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k2]
        # L_A: A[0,0]=1, A[0,1]=1
        # L_B: B[1,k2]=1
        # W: C[0,k2]+=-1, C[1,k2]+=1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
        # L_A: A[1,0]=1, A[0,0]=-1
        # L_B: B[0,k1]=1, B[0,k2]=1
        # W: C[1,k2]+=1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
        # L_A: A[0,1]=1, A[1,1]=-1
        # L_B: B[1,k1]=1, B[1,k2]=1
        # W: C[0,k1]+=1, C[1,k1]+=-1

        # Let's double check the W coefficients from the formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # W for M1: C11+=1, C22+=1
        # W for M2: C21+=1, C22+=-1
        # W for M3: C12+=1, C22+=1
        # W for M4: C11+=1, C21+=1
        # W for M5: C11+=-1, C12+=1
        # W for M6: C22+=1
        # W for M7: C11+=1, C21+=-1
        
        # Note: C11 is C[0, k1], C12 is C[0, k2], C21 is C[1, k1], C22 is C[1, k2].
        
        add_term(
            [(0,0,1.0), (1,1,1.0)],
            [(0,k1,1.0), (0,k2,1.0), (1,k1,1.0), (1,k2,1.0)],
            [(k1,0,1.0), (k2,1,1.0)]
        )
        
        add_term(
            [(1,0,1.0), (1,1,1.0)],
            [(0,k1,1.0)],
            [(k1,1,1.0), (k2,1,-1.0)]
        )
        
        add_term(
            [(0,0,1.0)],
            [(0,k2,1.0), (1,k2,-1.0)],
            [(k2,0,1.0), (k2,1,1.0)]
        )
        
        add_term(
            [(1,1,1.0)],
            [(1,k1,1.0), (0,k1,-1.0)],
            [(k1,0,1.0), (k1,1,1.0)]
        )
        
        add_term(
            [(0,0,1.0), (0,1,1.0)],
            [(1,k2,1.0)],
            [(k1,0,-1.0), (k2,0,1.0)]
        )
        
        add_term(
            [(1,0,1.0), (0,0,-1.0)],
            [(0,k1,1.0), (0,k2,1.0)],
            [(k2,1,1.0)]
        )
        
        add_term(
            [(0,1,1.0), (1,1,-1.0)],
            [(1,k1,1.0), (1,k2,1.0)],
            [(k1,0,1.0), (k1,1,-1.0)]
        )

    add_strassen(0)
    add_strassen(2)
    add_strassen(4)
    
    # Trivial for k=6
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    add_term(
        [(0,0,1.0)],
        [(0,6,1.0)],
        [(6,0,1.0)]
    )
    
    add_term(
        [(0,1,1.0)],
        [(1,6,1.0)],
        [(6,0,1.0)]
    )
    
    add_term(
        [(1,0,1.0)],
        [(0,6,1.0)],
        [(6,1,1.0)]
    )
    
    add_term(
        [(1,1,1.0)],
        [(1,6,1.0)],
        [(6,1,1.0)]
    )

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
