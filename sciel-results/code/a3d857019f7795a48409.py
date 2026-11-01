import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j*p+k, W row: k*n+i
    
    # Build the multiplication tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    # 1. Try to find rank 12 solution
    # Use a strategy of random sparse initialization + local polish
    solution_12 = _search_rank_sparse(n, m, p, T_flat, 12, deadline)
    if solution_12 is not None:
        U, V, W = solution_12
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # 2. Try to find rank 13 solution (current method, but optimized)
    # Get rank 14 solution for warm start
    U14, V14, W14 = _build_rank14_solution()
    
    # Try rank 13 with warm start
    solution_13 = _search_rank_13(n, m, p, T_flat, 0, deadline, 
                                  np.array(U14)[:, :13], 
                                  np.array(V14)[:, :13], 
                                  np.array(W14)[:, :13])
    if solution_13 is not None:
        U, V, W = solution_13
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # 3. Fall back to rank 14
    if _verify_solution(U14, V14, W14, n, m, p):
        return {"U": U14, "V": V14, "W": W14}
        
    # 4. Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def _search_rank_sparse(n, m, p, T_flat, rank, deadline):
    """
    Search for a solution of given rank using random sparse initialization 
    and local polishing.
    """
    np.random.seed(42)
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with sparse random values from {-1, -0.5, 0, 0.5, 1}
        # We want to encourage sparsity and simple values
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
        
        # Compute initial error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx.flatten() - T_flat)
        
        if error < 1e-3:
            # If initial error is small, polish
            U, V, W = _polish_solution(U, V, W, n, m, p, T_flat)
            if _verify_solution(U, V, W, n, m, p):
                return U, V, W
        
        # Also try a dense random init with small magnitude, then polish
        # This might find solutions that are not sparse
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx.flatten() - T_flat)
        
        if error < 1.0:
            U, V, W = _polish_solution(U, V, W, n, m, p, T_flat)
            if _verify_solution(U, V, W, n, m, p):
                return U, V, W
                
    return None

def _polish_solution(U, V, W, n, m, p, T_flat):
    """
    Polish a candidate solution using least squares.
    """
    rank = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return T_approx.flatten() - T_flat
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=1000)
        
        if result.cost < 1e-8:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Snap to nearest 0.5
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Verify snapped solution
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            else:
                # Return the polished solution even if not snapped perfectly, 
                # but verify again just in case
                if _verify_solution(U_opt, V_opt, W_opt, n, m, p):
                    return U_opt, V_opt, W_opt
                    
    except Exception:
        pass
        
    return U, V, W

def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]
    
    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]
    
    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]
    
    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]
    
    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]
    
    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True
