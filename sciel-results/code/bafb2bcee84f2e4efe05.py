import numpy as np

def get_decomposition(n, m, p):
    # Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # Using integers
    
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1]])
        V = np.array([[1]])
        W = np.array([[1]])
        return U, V, W
        
    # Base case: Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        # U rows: i*2+j. V rows: j*2+k. W rows: k*2+i.
        # M1: (a+d)(e+h)
        # a: U[0,0]=1, V[0,0]=1 (j=0,k=0? No, V is j*p+k. e is j=0,k=0. h is j=1,k=1.)
        # Let's trace carefully.
        # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d.
        # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h.
        # C indices: (0,0), (0,1), (1,0), (1,1).
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (a+d)(e+h)
        # U: a(0,0), d(1,1). V: e(0,0), h(1,1). W: C11(0,0), C22(1,1)?
        # C11 = M1 + M4 - M5 + M7. C22 = M1 - M2 + M3 + M6.
        # So W for M1 contributes to C11 and C22.
        # W row index k*n+i.
        # C11: k=0, i=0 -> idx 0.
        # C22: k=1, i=1 -> idx 3.
        # U: a(idx 0), d(idx 3).
        # V: e(idx 0), h(idx 3).
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        # U: c(idx 2), d(idx 3). V: e(idx 0).
        # C21 = M2 + M4. C21: k=1, i=0 -> idx 2.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = a(f-h)
        # U: a(idx 0). V: f(idx 1), h(idx 3).
        # C12 = M3 + M5. C12: k=1, i=0? No, C12 is row 1 col 2?
        # C is 2x2. C[0,1] is k=1, i=0?
        # W index k*n+i.
        # C[0,1]: i=0, k=1? No. C[i,k].
        # Task says W[k*n+i, r]. So W index corresponds to (k,i).
        # C[0,1] is i=0, k=1. Index 1*2+0 = 2.
        # C[1,0] is i=1, k=0. Index 0*2+1 = 1.
        # C[1,1] is i=1, k=1. Index 1*2+1 = 3.
        # C[0,0] is i=0, k=0. Index 0*2+0 = 0.
        
        # Re-check W indices.
        # M1 contributes to C[0,0] (idx 0) and C[1,1] (idx 3). Correct.
        # M2 contributes to C[1,0] (idx 1).
        # Correction: C21 in Strassen notation is C[1,0].
        # So W[1, 1] = 1.
        
        # Let's restart the Strassen mapping with clear indices.
        # C[0,0] (idx 0)
        # C[0,1] (idx 2)
        # C[1,0] (idx 1)
        # C[1,1] (idx 3)
        
        # M1 = (a+d)(e+h) -> C[0,0] +, C[1,1] +
        # M2 = (c+d)e -> C[1,0] +
        # M3 = a(f-h) -> C[0,1] +
        # M4 = d(g-e) -> C[0,0] +, C[1,0] +
        # M5 = (a+b)h -> C[0,0] -, C[0,1] +
        # M6 = (c-a)(e+f) -> C[1,1] +
        # M7 = (b-d)(g+h) -> C[0,0] +, C[1,1] -
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    # Recursive case
    # Try splitting m
    # We want to find a split that minimizes rank
    # Heuristic: split largest dimension
    
    # Try splitting m into m1 + m2
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook
    U_s = np.zeros((n * m, n * m * p), dtype=int)
    V_s = np.zeros((m * p, n * m * p), dtype=int)
    W_s = np.zeros((p * n, n * m * p), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = n * m * p

    # Try splits of m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = get_decomposition(n, m1, p)
        U2, V2, W2 = get_decomposition(n, m2, p)
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            # Remap U1
            U1_new = np.zeros((n * m, rank1), dtype=int)
            for i in range(n):
                for j in range(m1):
                    U1_new[i * m + j, :] = U1[i * m1 + j, :]
            
            # Remap U2
            U2_new = np.zeros((n * m, rank2), dtype=int)
            for i in range(n):
                for j in range(m2):
                    U2_new[i * m + (j + m1), :] = U2[i * m2 + j, :]
            
            # Remap V1
            V1_new = V1 # j in 0..m1-1 matches global j
            
            # Remap V2
            V2_new = np.zeros((m * p, rank2), dtype=int)
            for j in range(m2):
                for k in range(p):
                    V2_new[(j + m1) * p + k, :] = V2[j * p + k, :]
            
            # W is same
            W_new = np.hstack([W1, W2])
            
            U_new = np.hstack([U1_new, U2_new])
            V_new = np.hstack([V1_new, V2_new])
            
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank

    return best_U, best_V, best_W

def construct():
    n, m, p = 2, 4, 7
    U, V, W = get_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
