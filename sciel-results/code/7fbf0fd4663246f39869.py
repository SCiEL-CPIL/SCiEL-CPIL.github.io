import numpy as np
import time
from scipy import optimize

DIMENSIONS = [2, 2, 11]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U: 0=a00, 1=a01, 2=a10, 3=a11
    V: 0=b00, 1=b01, 2=b10, 3=b11
    W: 0=c00, 1=c01, 2=c10, 3=c11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Returns U, V, W for rank-4 schoolbook decomposition of 2x2x1.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # a00*b00 -> c00
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1 # a01*b10 -> c01? No, c01 = a00*b01 + a01*b11
    # Let's map carefully.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    # r0: A00 B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r1: A01 B10 -> C01? No, C01 is index 1 in W (if W is 2x2 flattened 0,1,2,3? No, W is 2x1 flattened 0,1)
    # W shape (2, 4) for p=1? No, p=1, n=2. W shape (1*2, 4) = (2, 4).
    # W[0, :] corresponds to C[0, 0]. W[1, :] corresponds to C[1, 0].
    # Wait, W[k*n+i]. For p=1, k=0. W[i, r].
    # So W[0, r] is C[0,0], W[1, r] is C[1,0].
    
    # r0: A00 B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r1: A01 B10
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # r2: A10 B00
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # r3: A11 B10
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _build_rank10_2x2x3():
    """
    Attempts to build a rank-10 decomposition for 2x2x3.
    U: (4, 10), V: (6, 10), W: (6, 10)
    V indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
    W indices: 0=c00, 1=c01, 2=c02, 3=c10, 4=c11, 5=c12
    """
    # We try to solve for the entries.
    # We initialize with a rank-11 structure (Strassen for first 2 cols + 4 schoolbook for 3rd col)
    # and try to reduce it to rank 10 by merging/optimizing.
    
    U, V, W = np.zeros((4, 10)), np.zeros((6, 10)), np.zeros((6, 10))
    
    # We use a known rank-10 decomposition if we can find one.
    # Otherwise, we use a numerical solver.
    
    # Target Tensor T
    T = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 2 + i
                T[a, b, c] = 1.0

    # Variables: U (40), V (60), W (60). Total 160.
    # We fix gauge: Set U[0, 0] = 1, U[1, 1] = 1, etc?
    # Better: Use least squares on the flattened tensor.
    
    def tensor_from_factors(U_f, V_f, W_f):
        rank = U_f.shape[1]
        T_flat = np.zeros(4 * 6 * 6)
        for r in range(rank):
            u = U_f[:, r]
            v = V_f[:, r]
            w = W_f[:, r]
            T_3d = np.einsum('a,b,c->abc', u, v, w)
            T_flat += T_3d.flatten()
        return T_flat

    T_target_flat = T.flatten()
    
    # Initial guess: Rank 11 (7 Strassen + 4 Schoolbook)
    # But we need rank 10. We start with random rank 10.
    np.random.seed(42)
    U_init = np.random.randn(4, 10) * 0.5
    V_init = np.random.randn(6, 10) * 0.5
    W_init = np.random.randn(6, 10) * 0.5
    
    def loss(x):
        U_f = x[0:40].reshape(4, 10)
        V_f = x[40:100].reshape(6, 10)
        W_f = x[100:160].reshape(6, 10)
        T_calc = tensor_from_factors(U_f, V_f, W_f)
        return np.sum((T_calc - T_target_flat)**2)
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use scipy.optimize.minimize
    # This might be slow, so we limit it.
    # If it fails, we return None.
    
    try:
        res = optimize.minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < 1e-4:
            U_f = res.x[0:40].reshape(4, 10)
            V_f = res.x[40:100].reshape(6, 10)
            W_f = res.x[100:160].reshape(6, 10)
            
            # Round to nearest 0.5
            U_int = np.round(U_f * 2) / 2
            V_int = np.round(V_f * 2) / 2
            W_int = np.round(W_f * 2) / 2
            
            # Verify exactness
            T_calc = tensor_from_factors(U_int, V_int, W_int)
            if np.allclose(T_calc, T_target_flat, atol=1e-9):
                return U_int, V_int, W_int
    except Exception:
        pass
    
    return None

def _build_rank39_decomposition():
    """Build the rank-39 decomposition: 5 Strassen blocks + 1 schoolbook block."""
    n, m, p = 2, 2, 11
    total_rank = 5 * 7 + 1 * 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        U_s, V_s, W_s = _build_strassen_2x2x2()
        
        # Map V_s (4x7) to V (22x7)
        # V_s[0] -> v_e, V_s[1] -> v_f, V_s[2] -> v_g, V_s[3] -> v_h
        V_block = np.zeros((22, 7))
        V_block[v_e, :] = V_s[0, :]
        V_block[v_f, :] = V_s[1, :]
        V_block[v_g, :] = V_s[2, :]
        V_block[v_h, :] = V_s[3, :]
        
        # Map W_s (4x7) to W (22x7)
        # W_s[0] -> w_c00, W_s[1] -> w_c01, W_s[2] -> w_c10, W_s[3] -> w_c11
        W_block = np.zeros((22, 7))
        W_block[w_c00, :] = W_s[0, :]
        W_block[w_c01, :] = W_s[1, :]
        W_block[w_c10, :] = W_s[2, :]
        W_block[w_c11, :] = W_s[3, :]
        
        U[:, col:col+7] = U_s
        V[:, col:col+7] = V_block
        W[:, col:col+7] = W_block
        col += 7
    
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    V_block = np.zeros((22, 4))
    V_block[v_e2, :] = V_sb[0, :]
    V_block[v_g2, :] = V_sb[1, :]
    
    W_block = np.zeros((22, 4))
    W_block[w_c02, :] = W_sb[0, :]
    W_block[w_c12, :] = W_sb[1, :]
    
    U[:, col:col+4] = U_sb
    V[:, col:col+4] = V_block
    W[:, col:col+4] = W_block
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_dec[i, k] = val
        
        if not np.allclose(C, C_dec, atol=tol):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to build rank-38 decomposition
    # 4 Strassen blocks (2x2x2) + 1 Rank-10 block (2x2x3)
    U_38 = None
    V_38 = None
    W_38 = None
    
    if time.time() < deadline - 60:
        U_s, V_s, W_s = _build_strassen_2x2x2()
        U_10, V_10, W_10 = _build_rank10_2x2x3()
        
        if U_10 is not None:
            total_rank = 4 * 7 + 10
            U_38 = np.zeros((4, total_rank))
            V_38 = np.zeros((22, total_rank))
            W_38 = np.zeros((22, total_rank))
            
            col = 0
            for block in range(4):
                k0 = block * 2
                k1 = block * 2 + 1
                
                v_e = k0
                v_f = k1
                v_g = 11 + k0
                v_h = 11 + k1
                
                w_c00 = 2 * k0
                w_c10 = 2 * k0 + 1
                w_c01 = 2 * k1
                w_c11 = 2 * k1 + 1
                
                V_block = np.zeros((22, 7))
                V_block[v_e, :] = V_s[0, :]
                V_block[v_f, :] = V_s[1, :]
                V_block[v_g, :] = V_s[2, :]
                V_block[v_h, :] = V_s[3, :]
                
                W_block = np.zeros((22, 7))
                W_block[w_c00, :] = W_s[0, :]
                W_block[w_c01, :] = W_s[1, :]
                W_block[w_c10, :] = W_s[2, :]
                W_block[w_c11, :] = W_s[3, :]
                
                U_38[:, col:col+7] = U_s
                V_38[:, col:col+7] = V_block
                W_38[:, col:col+7] = W_block
                col += 7
            
            # Last block: 2x2x3 (k=8,9,10)
            k0 = 8
            k1 = 9
            k2 = 10
            
            v_e = k0
            v_f = k1
            v_g = k2
            v_h = 11 + k0
            v_i = 11 + k1
            v_j = 11 + k2
            
            w_c00 = 2 * k0
            w_c01 = 2 * k1
            w_c02 = 2 * k2
            w_c10 = 2 * k0 + 1
            w_c11 = 2 * k1 + 1
            w_c12 = 2 * k2 + 1
            
            V_block = np.zeros((22, 10))
            V_block[v_e, :] = V_10[0, :]
            V_block[v_f, :] = V_10[1, :]
            V_block[v_g, :] = V_10[2, :]
            V_block[v_h, :] = V_10[3, :]
            V_block[v_i, :] = V_10[4, :]
            V_block[v_j, :] = V_10[5, :]
            
            W_block = np.zeros((22, 10))
            W_block[w_c00, :] = W_10[0, :]
            W_block[w_c01, :] = W_10[1, :]
            W_block[w_c02, :] = W_10[2, :]
            W_block[w_c10, :] = W_10[3, :]
            W_block[w_c11, :] = W_10[4, :]
            W_block[w_c12, :] = W_10[5, :]
            
            U_38[:, col:col+10] = U_10
            V_38[:, col:col+10] = V_block
            W_38[:, col:col+10] = W_block
            
            if _verify(U_38, V_38, W_38, n, m, p):
                return {"U": U_38.tolist(), "V": V_38.tolist(), "W": W_38.tolist()}
    
    # Fallback to rank-39
    U, V, W = _build_rank39_decomposition()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: Schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, column] = 1
                V_fb[j*p+k, column] = 1
                W_fb[k*n+i, column] = 1
                column += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
