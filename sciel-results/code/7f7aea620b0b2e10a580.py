import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor T[(i,j),(j',k),(k',i')] = delta_{j,j'} delta_{k,k'}."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    """Verify that U, V, W exactly decompose T_true."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W):
    """Snap values to the nearest half-integer."""
    U_r = np.round(np.array(U) * 2) / 2
    V_r = np.round(np.array(V) * 2) / 2
    W_r = np.round(np.array(W) * 2) / 2
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10, deadline=None):
    """Perform gradient descent to minimize ||T_recon - T_true||."""
    n, m, p = T_true.shape
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Gradients
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        # Update
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _strassen_2x2x2_decomposition():
    """
    Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    
    The 7 multiplications are:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    Output:
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    """
    # Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> i*2+k
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11
    
    # m2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1              # b00
    W[2, 1] = 1; W[3, 1] = -1 # c10, c11
    
    # m3 = a00(b01-b11)
    U[0, 2] = 1              # a00
    V[1, 2] = 1; V[3, 2] = -1 # b01, b11
    W[1, 2] = 1; W[3, 2] = 1 # c01, c11
    
    # m4 = a11(b10-b00)
    U[3, 3] = 1              # a11
    V[2, 3] = 1; V[0, 3] = -1 # b10, b00
    W[0, 3] = 1; W[2, 3] = 1 # c00, c10
    
    # m5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1              # b11
    W[0, 4] = -1; W[1, 4] = 1 # c00, c01
    
    # m6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1 # a10, a00
    V[0, 5] = 1; V[1, 5] = 1  # b00, b01
    W[3, 5] = 1              # c11
    
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1 # a01, a11
    V[2, 6] = 1; V[3, 6] = 1  # b10, b11
    W[0, 6] = 1              # c00
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_based_decomposition(n, m, p):
    """
    Try to use Strassen decomposition on 2x2 sub-blocks.
    For 3x4x8: split j into (j1,j2) where j1 in {0,1}, j2 in {0,1}
    Split k into (k1,k2) where k1 in {0,1}, k2 in {0..3}
    
    This gives 4 sub-problems of size 3x2x4, but we can further decompose.
    Actually, let's think differently:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Group j by j%2 and k by k%4:
    For each j1 in {0,1}, k1 in {0,1}, k2 in {0,1,2,3}:
    C[i, 4*k1+k2] includes terms where j = 2*j1+j2
    
    This is complex. Let me try a simpler structured approach.
    
    Alternative: Use the fact that 3x4x8 can be viewed as:
    - 3 independent 1x4x8 problems (but they share B)
    - Or group as (3x2x2) ⊗ something
    
    Let me try: partition the 4 inner dims into two groups of 2.
    A = [A1 | A2] where A1, A2 are 3x2
    B = [B1; B2] where B1, B2 are 2x8
    
    C = A1*B1 + A2*B2
    
    Each 3x2x8 has rank 48 schoolbook, total 96.
    
    But can we do better? 3x2x8: for each (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is just 2 terms per (i,k), so rank = 2*3*8 = 48 for schoolbook.
    
    To beat 48 for 3x2x8, we'd need a non-schoolbook decomposition.
    The minimum rank for 3x2x8 is not known to be less than 48.
    
    For 3x4x8, the best known is 96 (schoolbook). Let me try to find
    something better via gradient descent with good initialization.
    """
    return None  # Fallback to other methods

def construct():
    global START_TIME
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    deadline = START_TIME + TIME_LIMIT - 5
    
    # Build the true tensor
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 96
    
    # Try Strassen-based structured decomposition
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # For 3x4x8, we can try to exploit this on sub-blocks
    
    # Let's try: decompose as 3x2x4 + 3x2x4 where we use Strassen on 2x2 parts
    # Actually, let me try a different approach:
    # 3x4x8 = (1+1+1)x4x8 = three 1x4x8 problems
    # 1x4x8: C[k] = sum_j A[j]*B[j,k], rank = 4*8 = 32 schoolbook
    # Total: 96. Same as before.
    
    # Let me try gradient descent with various ranks
    # First, try rank 88 (Strassen-inspired improvement)
    
    ranks_to_try = [88, 80, 72, 64]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(restart * 1000 + target_rank)
            
            # Initialize with small random values
            U = rng.randn(nm, target_rank) * 0.5
            V = rng.randn(mp, target_rank) * 0.5
            W = rng.randn(pn, target_rank) * 0.5
            
            # Multi-phase gradient descent
            eta = 0.002
            max_iters = 2000
            
            for phase in range(5):
                if time.time() > deadline - 5:
                    break
                
                U, V, W, err = _gradient_descent(
                    U, V, W, T_true, eta, max_iters,
                    tol=1e-9, deadline=deadline - 5
                )
                
                if err < 1e-8:
                    # Check if we can snap to half-integers
                    snapped = _snap_to_half_inters(U, V, W)
                    if _verify(snapped[0], snapped[1], snapped[2], T_true, atol=1e-6):
                        best_result = {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
                        best_rank = target_rank
                        return best_result
                    else:
                        # Use the un-snapped version
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        return best_result
                
                # Reduce learning rate and iterations
                eta *= 0.4
                max_iters = max(200, max_iters // 3)
    
    # If we didn't find a better solution, return schoolbook
    return best_result
