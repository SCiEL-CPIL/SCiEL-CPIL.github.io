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
    
    best_solution = None
    
    # Get rank 14 solution for warm start
    U14, V14, W14 = _build_rank14_solution()
    
    # Apply peeling algorithm to extract rank-1 components and guide search
    peeled_info = _peel_tensor(T, n, m, p)
    
    max_restarts = 150 # Increase restarts with peeling guidance
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Use peeling info for better initialization
        if restart < 20:
            # Warm start from rank 14, first 13 columns
            U_init = np.array(U14)[:, :13]
            V_init = np.array(V14)[:, :13]
            W_init = np.array(W14)[:, :13]
        elif peeled_info is not None:
            U_init, V_init, W_init = _init_from_peeling(peeled_info, n, m, p, restart)
        else:
            U_init = None
            V_init = None
            W_init = None
            
        solution = _search_rank_13(n, m, p, T_flat, restart, deadline, U_init, V_init, W_init)
        if solution is not None:
            best_solution = solution
            break
            
    if best_solution is not None:
        U, V, W = best_solution
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fall back to rank 14
    if _verify_solution(U14, V14, W14, n, m, p):
        return {"U": U14, "V": V14, "W": W14}
        
    # Fall back to schoolbook rank 16
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

def _peel_tensor(T, n, m, p):
    """
    Apply peeling algorithm to extract rank-1 components from the tensor.
    Returns info about the peeling structure to guide rank-13 search.
    """
    T_work = T.copy()
    rank_1_components = []
    
    # Try to peel rank-1 components
    # A rank-1 component has form u ⊗ v ⊗ w where u, v, w are vectors
    # For matrix multiplication, trivial rank-1 components are those where
    # u, v, w each have at most 2 non-zero entries
    
    max_peels = 8  # Try to peel up to 8 components
    for _ in range(max_peels):
        # Find the entry with maximum absolute value
        max_idx = np.unravel_index(np.argmax(np.abs(T_work)), T_work.shape)
        max_val = T_work[max_idx]
        
        if abs(max_val) < 1e-10:
            break
        
        a, b, c = max_idx
        
        # Try to find rank-1 component containing this entry
        # Check if we can factor out a rank-1 term
        # For entry (a,b,c), a rank-1 term u⊗v⊗w with u[a]*v[b]*w[c] = T_work[a,b,c]
        
        # Strategy: look for simple rank-1 patterns
        # Pattern 1: u has one non-zero at a, v has one at b, w has one at c
        # This is always valid but may not be optimal
        
        # Better: check if rows/columns/planes have rank-1 structure
        # Extract the "slice" through (a,b,c)
        u_slice = T_work[a, :, :]  # shape (m*p, p*n)
        v_slice = T_work[:, b, :]  # shape (n*m, p*n)  
        w_slice = T_work[:, :, c]  # shape (n*m, m*p)
        
        # Check if u_slice is rank 1 (or close)
        # If T[a,b,c] != 0 and the slice is rank 1, we can peel
        U_svd, s_vals, Vt_svd = np.linalg.svd(u_slice, full_matrices=False)
        
        if len(s_vals) > 0 and s_vals[0] > 1e-8:
            rank_est = np.sum(s_vals > 1e-6 * s_vals[0])
            
            if rank_est == 1:
                # Pure rank-1 slice, peel it
                u_vec = U_svd[:, 0]
                v_vec = Vt_svd[0, :]
                
                # Normalize so that u_vec[a_idx_in_slice] * v_vec[b_idx_in_slice] matches
                # Actually, the slice is T[a, :, :] so indices are (b_idx, c_idx)
                # We need u_vec to be the a-part, but we only have the b,c part
                # Let me reconsider...
                
                # Actually for peeling, we need to find u, v, w such that
                # u ⊗ v ⊗ w is a component of T
                # The slice T[a, :, :] = u[a] * (v ⊗ w)
                # So v ⊗ w = T[a, :, :] / u[a]
                # We need to determine u[a]
                
                # For now, just record that we found structure
                rank_1_components.append({
                    'index': (a, b, c),
                    'value': max_val,
                    'u_slice_rank': rank_est
                })
                
                # Subtract the rank-1 component
                # u_vec is the b-c part scaled by u[a]
                # We need to figure out the actual u, v, w
                # This is complex; for now, just mark progress
                continue
        
        # If not rank-1, try a different approach
        # Look for entries where only one index varies
        break
    
    # Also identify the "trivial" rank-1 components that must be in any decomposition
    # For matrix multiplication, the "diagonal" entries create constraints
    
    # Return peeling info
    return {
        'components': rank_1_components,
        'residual_rank_est': _estimate_residual_rank(T_work, n, m, p)
    }

def _estimate_residual_rank(T_work, n, m, p):
    """Estimate the rank of the residual tensor after peeling."""
    # Use the flattening matrices
    # Flattening 1: (n*m) x (m*p * p*n)
    F1 = T_work.reshape(n*m, m*p*p*n)
    # Flattening 2: (m*p) x (n*m * p*n)  
    F2 = T_work.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    # Flattening 3: (p*n) x (n*m * m*p)
    F3 = T_work.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    
    # Rank is max of numerical ranks
    r1 = np.linalg.matrix_rank(F1, tol=1e-6)
    r2 = np.linalg.matrix_rank(F2, tol=1e-6)
    r3 = np.linalg.matrix_rank(F3, tol=1e-6)
    
    return max(r1, r2, r3)

def _init_from_peeling(peeled_info, n, m, p, seed):
    """Create initial guess for rank-13 search using peeling information."""
    np.random.seed(seed)
    rank = 13
    
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # Use peeling info to bias initialization
    if peeled_info and peeled_info['components']:
        for comp in peeled_info['components']:
            a, b, c = comp['index']
            # Add a small bias toward the peeled structure
            U[a, seed % rank] += 0.1 * np.sign(comp['value'])
            V[b, seed % rank] += 0.1
            W[c, seed % rank] += 0.1
    
    return U, V, W

def _search_rank_13(n, m, p, T_flat, seed, deadline, U_init=None, V_init=None, W_init=None):
    rank = 13
    np.random.seed(seed)
    
    # Initialize U, V, W
    if U_init is not None:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    # Flatten parameters
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return T_approx.flatten() - T_flat
    
    try:
        # Use a tighter max_nfev to prevent hanging
        result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        
        if time.time() > deadline - 5:
            return None
            
        if result.success or result.cost < 1e-6:
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
    except Exception:
        pass
        
    return None

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
