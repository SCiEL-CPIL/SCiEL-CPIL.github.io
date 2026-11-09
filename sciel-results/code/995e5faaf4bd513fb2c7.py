"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _construct_tensor(n, m, p):
    """Construct the matrix multiplication tensor T[i*m+k, k*p+j, j*n+i]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly represents the tensor."""
    T = _construct_tensor(n, m, p)
    rank = len(U[0])
    # Reconstruct tensor
    T_recon = np.zeros((n * m, m * p, p * n))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    return np.allclose(T_recon, T, atol=tol)


def _round_to_half_integer(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2


def _greedy_rank1_extraction(T_remaining, n, m, p, max_iter=100, tol=1e-8):
    """Extract a rank-1 component from T_remaining using power iteration."""
    # Initialize
    a = np.random.randn(n * m)
    b = np.random.randn(m * p)
    c = np.random.randn(p * n)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)
    
    for _ in range(max_iter):
        # Update a
        T_bc = np.tensordot(T_remaining, np.outer(b, c), axes=([1, 2], [0, 1]))
        a_new = T_bc / (np.linalg.norm(T_bc) + 1e-15)
        # Update b
        T_ac = np.tensordot(T_remaining, np.outer(c, a), axes=([0, 2], [0, 1]))
        b_new = T_ac / (np.linalg.norm(T_ac) + 1e-15)
        # Update c
        T_ab = np.tensordot(T_remaining, np.outer(a, b), axes=([0, 1], [0, 1]))
        c_new = T_ab / (np.linalg.norm(T_ab) + 1e-15)
        
        if (np.linalg.norm(a_new - a) < tol and 
            np.linalg.norm(b_new - b) < tol and 
            np.linalg.norm(c_new - c) < tol):
            a, b, c = a_new, b_new, c_new
            break
        a, b, c = a_new, b_new, c_new
    
    # Compute the scale factor
    T_bc = np.tensordot(T_remaining, np.outer(b, c), axes=([1, 2], [0, 1]))
    scale = np.dot(a, T_bc)
    
    return a * scale, b, c


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    T = _construct_tensor(n, m, p)
    T_remaining = T.copy()
    
    U_list = []
    V_list = []
    W_list = []
    
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    np.random.seed(42)
    
    while time.time() < deadline:
        # Extract rank-1 component
        a, b, c = _greedy_rank1_extraction(T_remaining, n, m, p, max_iter=50)
        
        if np.linalg.norm(a) < 1e-10:
            break
        
        # Round to half-integers
        a_rounded = _round_to_half_integer(a)
        b_rounded = _round_to_half_integer(b)
        c_rounded = _round_to_half_integer(c)
        
        # Check if rounding preserves the decomposition quality
        # We need: T_remaining ≈ a_rounded ⊗ b_rounded ⊗ c_rounded + (new T_remaining)
        # But we also need the total to equal T exactly
        
        # Instead, let's check if the rounded component is "valid" by checking
        # if subtracting it from T_remaining doesn't introduce large errors
        component = np.einsum('a,b,c->abc', a_rounded, b_rounded, c_rounded)
        T_new = T_remaining - component
        
        # Check if T_new can be completed with half-integer entries
        # For now, just track the best decomposition
        U_list.append(a_rounded.tolist())
        V_list.append(b_rounded.tolist())
        W_list.append(c_rounded.tolist())
        
        T_remaining = T_new
        
        # Check if we've converged (T_remaining is small)
        if np.linalg.norm(T_remaining) < 1e-6:
            # Verify the full decomposition
            U_arr = np.array(U_list).T
            V_arr = np.array(V_list).T
            W_arr = np.array(W_list).T
            T_recon = np.zeros((n * m, m * p, p * n))
            for r in range(len(U_list)):
                T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
            if np.allclose(T_recon, T, atol=1e-6):
                if len(U_list) < best_rank:
                    best_rank = len(U_list)
                    best_result = {
                        "U": U_arr.T.tolist(),
                        "V": V_arr.T.tolist(),
                        "W": W_arr.T.tolist()
                    }
            break
        
        # If T_remaining has large entries, the rounding may have caused issues
        if np.max(np.abs(T_remaining)) > 1.0:
            # Rounding caused a problem, try without rounding for this step
            U_list.pop()
            V_list.pop()
            W_list.pop()
            T_remaining = T_remaining + component  # Restore
            
            # Use unrounded component
            U_list.append(a.tolist())
            V_list.append(b.tolist())
            W_list.append(c.tolist())
            T_remaining = T - sum(np.einsum('a,b,c->abc', U_list[i], V_list[i], W_list[i]) for i in range(len(U_list)))
            # This is getting complicated, let's just break and use what we have
            break
    
    # Verify the best result
    if best_rank < n * m * p:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        T_recon = np.zeros((n * m, m * p, p * n))
        for r in range(best_rank):
            T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        if not np.allclose(T_recon, T, atol=1e-6):
            # Invalid, fall back to schoolbook
            return {"U": U, "V": V, "W": W}
    
    return best_result
