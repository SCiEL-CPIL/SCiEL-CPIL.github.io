"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; else 0."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W form a valid decomposition of the multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_decomp, T_true, atol=tol)

def _laderman_2x3x3():
    """Return the rank-7 Laderman decomposition for 2x3 * 3x3 -> 2x3.
    Returns (U, V, W) with shapes (6,7), (9,7), (6,7).
    A = [[a,b,c],[d,e,f]] indexed as a=0,b=1,c=2,d=3,e=4,f=5
    B = [[g,h,i],[j,k,l],[m,n,o]] indexed as g=0,h=1,i=2,j=3,k=4,l=5,m=6,n=7,o=8
    C = [[p,q,r],[s,t,u]] indexed as p=0,q=1,r=2,s=3,t=4,u=5
    """
    # Laderman's 7 multiplications for 2x3 * 3x3:
    # m1 = (a+d)(g+j)
    # m2 = (b+e)(h+k)  
    # m3 = (c+f)(i+l)
    # m4 = (a+b+e+f)(g+h+j+k)
    # m5 = (a+d-f)(g+j-l)
    # m6 = (b+e-c)(h+k-n)
    # m7 = (a+b+d+e)(g+h+j+k)  -- need to verify
    
    # Let me use a verified rank-7 scheme. The standard Laderman for 2x3x3:
    # Multiplications:
    # m1 = (a+d)*(g+j)
    # m2 = (b+e)*(h+k)
    # m3 = (c+f)*(i+l)
    # m4 = (a+e-c-f)*(g+j-h-k)
    # m5 = (a+b+d+e)*(g+h+j+k)
    # m6 = (a+b+e+f-c)*(g+h+j+k-i)
    # m7 = (a+d+f)*(g+j+l)
    
    # Actually, let me derive it properly. We need:
    # p = ag + bj + cm
    # q = ah + bk + cn
    # r = ai + bl + co
    # s = dg + ej + fm
    # t = dh + ek + fn
    # u = di + el + fo
    
    # A known rank-7 decomposition (from Laderman 1976):
    # Let me use the explicit coefficients.
    # U matrix (6x7): rows for a,b,c,d,e,f
    # V matrix (9x7): rows for g,h,i,j,k,l,m,n,o
    # W matrix (6x7): rows for p,q,r,s,t,u
    
    # From the literature, one valid rank-7 scheme:
    # m1 = (a+d)(g+j)
    # m2 = (b+e)(h+k)
    # m3 = (c+f)(i+l)
    # m4 = (a+e-c-f)(g+j-h-k)
    # m5 = (a+b+d+e)(g+h+j+k)
    # m6 = (a+b+e+f-c)(g+h+j+k-i)  -- this doesn't look right
    # m7 = (a+d+f)(g+j+l)
    
    # Let me just verify numerically. I'll construct candidate schemes and check.
    # For now, use a scheme that I can verify:
    
    # Scheme:
    # m1 = (a+d)(g+j)
    # m2 = (b+e)(h+k)
    # m3 = (c+f)(i+l)
    # m4 = (a+e-c-f)(g+j-h-k)
    # m5 = (a+b+d+e)(g+h+j+k)
    # m6 = (b+e+c+f)(h+k+i+l)  -- symmetric
    # m7 = (a+d+f)(g+j+l)
    
    # Let me build the coefficient arrays and verify.
    
    # U coefficients for each multiplication (7 terms), 6 rows (a,b,c,d,e,f):
    # m1: a+d -> [1,0,0,1,0,0]
    # m2: b+e -> [0,1,0,0,1,0]
    # m3: c+f -> [0,0,1,0,0,1]
    # m4: a+e-c-f -> [1,0,-1,0,1,-1]
    # m5: a+b+d+e -> [1,1,0,1,1,0]
    # m6: b+e+c+f -> [0,1,1,0,1,1]
    # m7: a+d+f -> [1,0,0,1,0,1]
    
    U_lad = np.array([
        [1, 0, 0, 1, 1, 0, 1],   # a
        [0, 1, 0, 0, 1, 1, 0],   # b
        [0, 0, 1, -1, 0, 1, 0],  # c
        [1, 0, 0, 0, 1, 0, 1],   # d
        [0, 1, 0, 1, 1, 1, 0],   # e
        [0, 0, 1, -1, 0, 1, 1],  # f
    ], dtype=float)
    
    # V coefficients for each multiplication (7 terms), 9 rows (g,h,i,j,k,l,m,n,o):
    # m1: g+j -> [1,0,0,1,0,0,0,0,0]
    # m2: h+k -> [0,1,0,0,1,0,0,0,0]
    # m3: i+l -> [0,0,1,0,0,1,0,0,0]
    # m4: g+j-h-k -> [1,-1,0,1,-1,0,0,0,0]
    # m5: g+h+j+k -> [1,1,0,1,1,0,0,0,0]
    # m6: h+k+i+l -> [0,1,1,0,1,1,0,0,0]
    # m7: g+j+l -> [1,0,0,1,0,1,0,0,0]
    
    V_lad = np.array([
        [1, 0, 0, 1, 1, 0, 1],   # g
        [0, 1, 0, -1, 1, 1, 0],  # h
        [0, 0, 1, 0, 0, 1, 0],   # i
        [1, 0, 0, 1, 1, 0, 1],   # j
        [0, 1, 0, -1, 1, 1, 0],  # k
        [0, 0, 1, 0, 0, 1, 1],   # l
        [0, 0, 0, 0, 0, 0, 0],   # m
        [0, 0, 0, 0, 0, 0, 0],   # n
        [0, 0, 0, 0, 0, 0, 0],   # o
    ], dtype=float)
    
    # W coefficients for each multiplication (7 terms), 6 rows (p,q,r,s,t,u):
    # p = ag + bj + cm
    # q = ah + bk + cn
    # r = ai + bl + co
    # s = dg + ej + fm
    # t = dh + ek + fn
    # u = di + el + fo
    
    # p: from m1(ag), m1(dg->no), m2(bj), m3(cm)...
    # m1 contributes: a*g -> p, d*g -> s
    # m2 contributes: b*h -> q, e*h -> t
    # m3 contributes: c*i -> r, f*i -> u
    # m4 contributes: a*g -> p, e*g -> s, -c*h -> -q, f*h -> -t, -a*h -> -q, e*h -> t, c*j -> r, f*j -> u, -c*j -> -r, f*j -> u
    # This is getting complex. Let me just compute W numerically.
    
    # For each term r, the contribution to C[i,k] is U[i,r]*V[j,r]*W[k,r] summed over j.
    # Actually, for the full tensor: sum_r U[i,j_A, r] * V[j_B, k_B, r] * W[k_C, i_C, r]
    # where i_A is row of A (0 or 1), j_A is col of A (0,1,2)
    # j_B is row of B (0,1,2), k_B is col of B (0,1,2)
    # k_C is col of C (0,1,2), i_C is row of C (0,1)
    
    # T[i_A, j_A; j_B, k_B; k_C, i_C] = 1 if i_A=i_C, j_A=j_B, k_B=k_C
    
    # So for each r: sum over (i_A,j_A,j_B,k_B,k_C,i_C) of U[i_A*3+j_A, r] * V[j_B*3+k_B, r] * W[k_C*3+i_C, r]
    # should equal delta_{i_A,i_C} * delta_{j_A,j_B} * delta_{k_B,k_C}
    
    # Let me just solve for W given U and V.
    # For each (i_C, k_C) output position, we need:
    # sum_r U[i_C*3+j_A, r] * V[j_A*3+k_C, r] * W[k_C*3+i_C, r] = 1 for all j_A
    # Wait, that's not quite right. Let me be more careful.
    
    # The condition is: for all (i_A, j_A, j_B, k_B, k_C, i_C):
    # sum_r U[i_A*3+j_A, r] * V[j_B*3+k_B, r] * W[k_C*3+i_C, r] = delta_{i_A,i_C} delta_{j_A,j_B} delta_{k_B,k_C}
    
    # This means: for fixed (i_C, k_C), the vector W[k_C*3+i_C, :] must satisfy:
    # For all j_A: sum_r U[i_C*3+j_A, r] * V[j_A*3+k_C, r] * W[k_C*3+i_C, r] = 1
    # And for i_A != i_C or k_B != k_C: the sum is 0.
    
    # Actually, the condition separates. For the output C[i_C, k_C] = sum_{j} A[i_C, j] * B[j, k_C]:
    # We need: sum_r U[i_C*3+j, r] * V[j*3+k_C, r] * W[k_C*3+i_C, r] = 1 for all j
    # And for any other combination, it should be 0.
    
    # Let me define: for each r, let u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
    # The r-th term contributes to the tensor as u_r ⊗ v_r ⊗ w_r
    # T_decomp[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    # where a = i_A*3+j_A, b = j_B*3+k_B, c = k_C*3+i_C
    
    # We need T_decomp[a,b,c] = 1 if i_A=i_C, j_A=j_B, k_B=k_C, else 0.
    
    # Let me just verify my U and V by checking if there exists a W.
    # For each output position c = k_C*3+i_C, I need:
    # sum_r U[a, r] * V[b, r] * W[c, r] = T_true[a, b, c]
    
    # This is a linear system in W. For each c, W[c, :] is a vector of length 7.
    # The equation is: for all (a,b): sum_r W[c, r] * (U[a,r] * V[b,r]) = T_true[a,b,c]
    
    # Let M_c be a matrix where M_c[(a,b), r] = U[a,r] * V[b,r]
    # Then M_c @ W[c,:] = T_true[:, :, c].ravel()
    
    # Let me construct this and solve.
    
    n, m, p = 2, 3, 3
    T_true = _build_tensor(n, m, p)
    
    # U_lad shape: (6, 7), V_lad shape: (9, 7)
    # a ranges over 0..5 (i_A*3+j_A), b ranges over 0..8 (j_B*3+k_B), c ranges over 0..5 (k_C*3+i_C)
    
    W_lad = np.zeros((6, 7), dtype=float)
    
    for c in range(6):
        # Build M_c: shape (54, 7) where 54 = 6*9
        M_c = np.zeros((54, 7), dtype=float)
        rhs = np.zeros(54, dtype=float)
        idx = 0
        for a in range(6):
            for b in range(9):
                for r in range(7):
                    M_c[idx, r] = U_lad[a, r] * V_lad[b, r]
                rhs[idx] = T_true[a, b, c]
                idx += 1
        
        # Solve M_c @ W_c = rhs (overdetermined, 54 equations, 7 unknowns)
        # Use least squares
        W_c, residuals, rank_c, sv = np.linalg.lstsq(M_c, rhs, rcond=None)
        W_lad[c, :] = W_c
    
    # Verify
    T_decomp = np.zeros((6, 9, 6), dtype=float)
    for r in range(7):
        T_decomp += np.einsum('i,j,k->ijk', U_lad[:,r], V_lad[:,r], W_lad[:,r])
    
    if not np.allclose(T_decomp, T_true, atol=1e-9):
        # My U,V scheme doesn't work. Let me try a different approach.
        # Use a known valid Laderman scheme.
        # 
        # Actually, let me just use a verified scheme from the literature.
        # The rank of 2x3x3 is indeed 7. Here's a verified decomposition:
        # 
        # I'll use a different construction. Let me try:
        # m1 = a*g
        # m2 = b*h  
        # m3 = c*i
        # m4 = d*j
        # m5 = e*k
        # m6 = f*l
        # m7 = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o) * something... 
        # 
        # This is getting too complex. Let me just use a numerical rank-7 finder.
        pass
    
    return U_lad, V_lad, W_lad

def _find_laderman_2x3x3():
    """Find a rank-7 decomposition for 2x3 * 3x3 -> 2x3 using numerical search."""
    n, m, p = 2, 3, 3
    T_true = _build_tensor(n, m, p)
    rank = 7
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    import random
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        U = rng.randn(6, rank) * 0.5
        V = rng.randn(9, rank) * 0.5
        W = rng.randn(6, rank) * 0.5
        
        for step in range(500):
            F = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = F - T_true
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-10:
                break
            
            dU = 2.0 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2.0 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2.0 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Adaptive learning rate
            lr = 0.01 / (1 + step * 0.01)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-9:
            break
    
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def _construct_rank21():
    """Construct rank-21 decomposition for 2x3 * 3x9 using Laderman on three 3x3 blocks."""
    n, m, p = 2, 3, 9
    
    # First, find the Laderman decomposition for 2x3 * 3x3
    lad_result = _find_laderman_2x3x3()
    
    if lad_result is None:
        # Fallback: try the analytic scheme
        U_lad, V_lad, W_lad = _laderman_2x3x3()
        # Verify
        T_check = np.zeros((6, 9, 6), dtype=float)
        for r in range(7):
            T_check += np.einsum('i,j,k->ijk', U_lad[:,r], V_lad[:,r], W_lad[:,r])
        T_true_check = _build_tensor(2, 3, 3)
        if not np.allclose(T_check, T_true_check, atol=1e-9):
            return None
    else:
        U_lad, V_lad, W_lad = lad_result
    
    # Now construct the full 2x3 * 3x9 decomposition.
    # B is 3x9, split into three 3x3 blocks: B[:,0:3], B[:,3:6], B[:,6:9]
    # C is 2x9, split into three 2x3 blocks: C[:,0:3], C[:,3:6], C[:,6:9]
    # C[:,k*3:(k+1)*3] = A * B[:,k*3:(k+1)*3]
    
    # U shape: (6, 21), V shape: (27, 21), W shape: (18, 21)
    U_full = np.zeros((6, 21), dtype=float)
    V_full = np.zeros((27, 21), dtype=float)
    W_full = np.zeros((18, 21), dtype=float)
    
    for block in range(3):
        # U: same for all blocks (A is the same)
        U_full[:, block*7:(block+1)*7] = U_lad
        
        # V: B[:, block*3:(block+1)*3] corresponds to rows block*3 to block*3+8 of V_full
        # V_lad has shape (9, 7) where row j*3+k corresponds to B[j,k]
        # In the full V, row j*3 + block*3 + k corresponds to B[j, block*3+k]
        for j in range(3):
            for k in range(3):
                v_idx = j * 3 + block * 3 + k  # row in V_full (0..26)
                v_lad_idx = j * 3 + k  # row in V_lad (0..8)
                V_full[v_idx, block*7:(block+1)*7] = V_lad[v_lad_idx, :]
        
        # W: C[:, block*3:(block+1)*3] corresponds to rows block*6 to block*6+5 of W_full
        # W_lad has shape (6, 7) where row k*3+i corresponds to C[i,k] (i=row, k=col)
        # Wait, the output index is k*n+i where n=2, so index = k*2+i
        # Hmm, let me re-check. The output C is 2x9, so p*n = 9*2 = 18 entries.
        # Index = k*n + i where k is column (0..8), i is row (0,1)
        # For block b, columns are 3*b, 3*b+1, 3*b+2
        # So indices are (3*b+k)*2 + i for k=0,1,2 and i=0,1
        # That's 6*b, 6*b+1, 6*b+2, 6*b+3, 6*b+4, 6*b+5
        
        # W_lad row index: for 2x3 output, index = k*2+i where k=col(0,1,2), i=row(0,1)
        # So W_lad[k*2+i, r] corresponds to C[i, k]
        
        for k in range(3):
            for i in range(2):
                w_full_idx = (3 * block + k) * 2 + i  # row in W_full (0..17)
                w_lad_idx = k * 2 + i  # row in W_lad (0..5)
                W_full[w_full_idx, block*7:(block+1)*7] = W_lad[w_lad_idx, :]
    
    return U_full, V_full, W_full

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W form a valid decomposition of the multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_decomp, T_true, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to construct rank-21 decomposition
    result = _construct_rank21()
    
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
