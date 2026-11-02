"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices: u = i*2+j, v = j*2+k, w = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # t=0 (M1): (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0,0]=1, U[3,3]=1
    U[0, 0] = 1; U[3, 0] = 1
    # V: B00=1, B11=1 -> V[0,0]=1, V[3,3]=1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00+=1, C11+=1 -> W[0,0]=1, W[3,3]=1 (w=k*2+i)
    # C00: k=0,i=0 -> w=0. C11: k=1,i=1 -> w=3.
    W[0, 0] = 1; W[3, 0] = 1
    
    # t=1 (M2): (A10+A11)B00
    # U: A10=1, A11=1 -> U[2,0]=1, U[3,0]=1 (i=1,j=0 -> 2; i=1,j=1 -> 3)
    U[2, 1] = 1; U[3, 1] = 1
    # V: B00=1 -> V[0,0]=1
    V[0, 1] = 1
    # W: C10+=1, C11-=1
    # C10: k=0,i=1 -> w=2. C11: k=1,i=1 -> w=3.
    W[2, 1] = 1; W[3, 1] = -1
    
    # t=2 (M3): A00(B01-B11)
    # U: A00=1 -> U[0,0]=1
    U[0, 2] = 1
    # V: B01=1, B11=-1 -> V[1,0]=1, V[3,0]=-1 (j=0,k=1 -> 1; j=1,k=1 -> 3)
    V[1, 2] = 1; V[3, 2] = -1
    # W: C01+=1, C11+=1
    # C01: k=1,i=0 -> w=2. C11: k=1,i=1 -> w=3.
    W[2, 2] = 1; W[3, 2] = 1
    
    # t=3 (M4): A11(B10-B00)
    # U: A11=1 -> U[3,0]=1
    U[3, 3] = 1
    # V: B10=1, B00=-1 -> V[2,0]=1, V[0,0]=-1 (j=1,k=0 -> 2; j=0,k=0 -> 0)
    V[2, 3] = 1; V[0, 3] = -1
    # W: C00+=1, C10+=1
    # C00: k=0,i=0 -> w=0. C10: k=0,i=1 -> w=2.
    W[0, 3] = 1; W[2, 3] = 1
    
    # t=4 (M5): (A00+A01)B11
    # U: A00=1, A01=1 -> U[0,0]=1, U[1,0]=1
    U[0, 4] = 1; U[1, 4] = 1
    # V: B11=1 -> V[3,0]=1
    V[3, 4] = 1
    # W: C00-=1, C01+=1
    # C00: k=0,i=0 -> w=0. C01: k=1,i=0 -> w=2.
    W[0, 4] = -1; W[2, 4] = 1
    
    # t=5 (M6): (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[2,0]=1, U[0,0]=-1
    U[2, 5] = 1; U[0, 5] = -1
    # V: B00=1, B01=1 -> V[0,0]=1, V[1,0]=1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C11+=1
    # C11: k=1,i=1 -> w=3.
    W[3, 5] = 1
    
    # t=6 (M7): (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[1,0]=1, U[3,0]=-1
    U[1, 6] = 1; U[3, 6] = -1
    # V: B10=1, B11=1 -> V[2,0]=1, V[3,0]=1
    V[2, 6] = 1; V[3, 6] = 1
    # W: C00+=1
    # C00: k=0,i=0 -> w=0.
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4x4_factors():
    """
    Construct factors for 4x4x4 multiplication using Strassen on 2x2 blocks.
    Rank 49.
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Outer Strassen terms t_o in 0..6
    # Inner Strassen terms t_i in 0..6
    for t_o in range(7):
        for t_i in range(7):
            r = t_o * 7 + t_i
            
            # Outer coefficients
            u_o = U2[:, t_o] # shape (4,)
            v_o = V2[:, t_o] # shape (4,)
            w_o = W2[:, t_o] # shape (4,)
            
            # Inner coefficients
            u_i = U2[:, t_i] # shape (4,)
            v_i = V2[:, t_i] # shape (4,)
            w_i = W2[:, t_i] # shape (4,)
            
            # Expand to 4x4
            # i = 2*I + a, j = 2*J + b
            # k = 2*K + c
            
            for I in range(2):
                for J in range(2):
                    for a in range(2):
                        for b in range(2):
                            idx_u = I * 4 + J * 2 + a # This is wrong. i*m+j. m=4.
                            # i = 2*I + a, j = 2*J + b
                            # idx_u = i*4 + j = (2*I+a)*4 + (2*J+b)
                            idx_u = (2*I + a) * 4 + (2*J + b)
                            coeff = u_o[I * 2 + J] * u_i[a * 2 + b]
                            U[idx_u, r] = coeff
                            
                            idx_v = (2*J + b) * 4 + (2*K + c) # Wait, K,c not defined here
                            # We need to fill V and W similarly.
                            # Let's do it in separate loops for clarity or nested properly.
                            
            # Let's re-do the filling logic more cleanly
            for I in range(2):
                for J in range(2):
                    u_o_val = u_o[I * 2 + J]
                    if u_o_val == 0: continue
                    
                    for a in range(2):
                        for b in range(2):
                            u_i_val = u_i[a * 2 + b]
                            if u_i_val == 0: continue
                            
                            i = 2 * I + a
                            j = 2 * J + b
                            idx_u = i * 4 + j
                            U[idx_u, r] = u_o_val * u_i_val
                            
                            for K in range(2):
                                v_o_val = v_o[J * 2 + K]
                                if v_o_val == 0: continue
                                
                                for c in range(2):
                                    v_i_val = v_i[b * 2 + c]
                                    if v_i_val == 0: continue
                                    
                                    k = 2 * K + c
                                    idx_v = j * 4 + k
                                    V[idx_v, r] = v_o_val * v_i_val
                                    
                                    w_o_val = w_o[K * 2 + I]
                                    if w_o_val == 0: continue
                                    
                                    w_i_val = w_i[c * 2 + a]
                                    if w_i_val == 0: continue
                                    
                                    idx_w = k * 4 + i
                                    W[idx_w, r] = w_o_val * w_i_val

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook rank 128
    best = _schoolbook(n, m, p)
    best_rank = 128
    
    # Try Strassen 4x4x4x4 twice (for 4x8)
    try:
        U4, V4, W4 = _strassen_4x4_4x4_factors()
        
        # We need to combine two 4x4x4 decompositions into one 4x4x8.
        # The second 4x4x4 handles k=4..7.
        # We can just stack them.
        # U is (16, 49). We need (16, 98).
        # V is (16, 49) for first part (k=0..3). For second part (k=4..7), V indices shift.
        # V index is j*p + k. p=8.
        # For first part, k' = k. V[j*8 + k, r].
        # For second part, k' = k + 4. V[j*8 + (k+4), r].
        
        U_full = np.zeros((16, 98))
        V_full = np.zeros((32, 98)) # m*p = 4*8 = 32
        W_full = np.zeros((32, 98)) # p*n = 8*4 = 32
        
        # First half (k=0..3)
        U_full[:, 0:49] = U4
        # V4 has indices j*4+k. We need j*8+k.
        for j in range(4):
            for k in range(4):
                V_full[j*8 + k, 0:49] = V4[j*4 + k, :]
        # W4 has indices k*4+i. We need k*8+i.
        for k in range(4):
            for i in range(4):
                W_full[k*8 + i, 0:49] = W4[k*4 + i, :]
        
        # Second half (k=4..7)
        U_full[:, 49:98] = U4
        # V4 indices j*4+k'. We need j*8 + (k'+4).
        for j in range(4):
            for k in range(4):
                V_full[j*8 + (k + 4), 49:98] = V4[j*4 + k, :]
        # W4 indices k'*4+i. We need (k'+4)*8 + i.
        for k in range(4):
            for i in range(4):
                W_full[(k + 4)*8 + i, 49:98] = W4[k*4 + i, :]
        
        # Verify
        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
    except Exception as e:
        pass
    
    return best
# EVOLVE-BLOCK-END
