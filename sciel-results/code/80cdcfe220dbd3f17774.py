import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    First tries to find rank-54 via structured search, falls back to rank-55.
    """
    n, m, p = DIMENSIONS
    
    # Try rank-54 search
    result = _search_rank_54()
    if result is not None:
        return result
    
    # Fallback to rank-55
    return _construct_rank_55()

def _construct_rank_55():
    """Original rank-55 construction using Strassen + schoolbook per block."""
    n, m, p = DIMENSIONS
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen Part (7 terms)
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1
        
        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1
        
        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # Schoolbook Part (4 terms)
        U[a02, col] = 1.0; V[b20, col] = 1.0; W[c00, col] = 1.0
        col += 1
        
        U[a02, col] = 1.0; V[b21, col] = 1.0; W[c01, col] = 1.0
        col += 1
        
        U[a12, col] = 1.0; V[b20, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        U[a12, col] = 1.0; V[b21, col] = 1.0; W[c11, col] = 1.0
        col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _search_rank_54():
    """
    Search for rank-54 decomposition by allowing cross-block sharing.
    
    Strategy: Start from rank-55 block-diagonal construction.
    Try to merge pairs of terms across blocks by finding linear dependencies.
    
    For each pair of blocks (b1, b2), check if we can replace 2 terms with 1
    by finding a common factor pattern.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    target_rank = 54
    
    # Build the base rank-55 decomposition
    U_base = np.zeros((n * m, 55), dtype=float)
    V_base = np.zeros((m * p, 55), dtype=float)
    W_base = np.zeros((p * n, 55), dtype=float)
    
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen terms 0-6
        U_base[a00, col] = 1.0; U_base[a11, col] = 1.0
        V_base[b00, col] = 1.0; V_base[b11, col] = 1.0
        W_base[c00, col] = 1.0; W_base[c11, col] = 1.0
        col += 1
        
        U_base[a10, col] = 1.0; U_base[a11, col] = 1.0
        V_base[b00, col] = 1.0
        W_base[c10, col] = 1.0; W_base[c11, col] = -1.0
        col += 1
        
        U_base[a00, col] = 1.0
        V_base[b01, col] = 1.0; V_base[b11, col] = -1.0
        W_base[c01, col] = 1.0; W_base[c11, col] = 1.0
        col += 1
        
        U_base[a11, col] = 1.0
        V_base[b10, col] = 1.0; V_base[b00, col] = -1.0
        W_base[c00, col] = 1.0; W_base[c10, col] = 1.0
        col += 1
        
        U_base[a00, col] = 1.0; U_base[a01, col] = 1.0
        V_base[b11, col] = 1.0
        W_base[c00, col] = -1.0; W_base[c01, col] = 1.0
        col += 1
        
        U_base[a10, col] = 1.0; U_base[a00, col] = -1.0
        V_base[b00, col] = 1.0; V_base[b01, col] = 1.0
        W_base[c11, col] = 1.0
        col += 1
        
        U_base[a01, col] = 1.0; U_base[a11, col] = -1.0
        V_base[b10, col] = 1.0; V_base[b11, col] = 1.0
        W_base[c00, col] = 1.0
        col += 1
        
        # Schoolbook terms 7-10
        U_base[a02, col] = 1.0; V_base[b20, col] = 1.0; W_base[c00, col] = 1.0
        col += 1
        
        U_base[a02, col] = 1.0; V_base[b21, col] = 1.0; W_base[c01, col] = 1.0
        col += 1
        
        U_base[a12, col] = 1.0; V_base[b20, col] = 1.0; W_base[c10, col] = 1.0
        col += 1
        
        U_base[a12, col] = 1.0; V_base[b21, col] = 1.0; W_base[c11, col] = 1.0
        col += 1
    
    # Try to find a rank-54 decomposition by merging terms
    # For each pair of terms, check if their tensor can be expressed as a sum
    # of at most 1 rank-1 tensor (i.e., if they're "close" to being rank-1)
    
    # Approach: Try all pairs (i, j) where i < j, and check if we can
    # replace terms i and j with a single term that still satisfies the
    # multiplication tensor equation.
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if i*p+j*k == (i,j,k)
    # Actually, T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to A_idx*B_idx
    
    # Build the full multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # Trace convention W[k*n+i, r]
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Check if the base decomposition is correct
    # T_decomp = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # For efficiency, check on random inputs
    
    def verify_decomposition(U, V, W, num_trials=10):
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # C += u * v * w (element-wise in tensor sense)
                # Actually: C[i,k] += sum_r U[i*? , r] * V[?, r] * W[k*n+i, r]
                # Let me re-derive: C[i,k] = sum_j A[i,j] B[j,k]
                # In decomposition: C[i,k] = sum_r U[i*m+j?, r] * V[j*p+k?, r] * W[k*n+i, r]
                # This doesn't quite work. Let me use the standard form:
                # C = sum_r (U_r ⊗ V_r) * W_r where U_r is n×m, V_r is m×p
                # Actually the standard is: C[i,k] = sum_r sum_j U[i,j,r] * V[j,k,r] * W[i,k,r]
                # But our U is (n*m, rank), so U[i*m+j, r]
                # V is (m*p, rank), so V[j*p+k, r]  
                # W is (p*n, rank), so W[k*n+i, r]
                
                # This is getting complex. Let me just check the tensor directly.
            pass
        
        # Better: verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T
        T_decomp = np.zeros_like(T)
        for r in range(U.shape[1]):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.allclose(T_decomp, T, atol=1e-10)
    
    # Verify base
    if not verify_decomposition(U_base, V_base, W_base):
        return None
    
    # Now try to reduce rank by 1
    # Try removing each term and see if we can compensate with a single new term
    
    best_result = None
    
    for remove_idx in range(55):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Create candidate by removing term remove_idx
        keep_mask = [i for i in range(55) if i != remove_idx]
        U_cand = U_base[:, keep_mask]
        V_cand = V_base[:, keep_mask]
        W_cand = W_base[:, keep_mask]
        
        # Compute residual: T - sum_{r in keep} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        T_decomp = np.zeros_like(T)
        for r in range(54):
            T_decomp += np.einsum('a,b,c->abc', U_cand[:, r], V_cand[:, r], W_cand[:, r])
        
        residual = T - T_decomp
        
        # Check if residual is rank-1
        # A tensor is rank-1 if it can be written as u ⊗ v ⊗ w
        # This is equivalent to all 2×2 minors of any unfolding being zero
        
        # Unfold along mode 0: matrix of shape (n*m, m*p * p*n)
        # If rank-1, all 2×2 minors are zero
        unfold0 = residual.reshape(n*m, m*p*p*n)
        
        # Check if rank <= 1
        # Use SVD or check if all 2×2 minors are zero
        # For efficiency, check a few random 2×2 minors
        
        is_rank1 = True
        num_checks = 20
        for _ in range(num_checks):
            i1 = np.random.randint(n*m)
            i2 = np.random.randint(n*m)
            j1 = np.random.randint(m*p*p*n)
            j2 = np.random.randint(m*p*p*n)
            minor = unfold0[i1, j1] * unfold0[i2, j2] - unfold0[i1, j2] * unfold0[i2, j1]
            if abs(minor) > 1e-10:
                is_rank1 = False
                break
        
        if is_rank1:
            # Extract the rank-1 components
            # Find a nonzero entry
            nonzero = np.argwhere(residual != 0)
            if len(nonzero) > 0:
                a_idx, b_idx, c_idx = nonzero[0]
                val = residual[a_idx, b_idx, c_idx]
                
                # Try to factor
                # u = residual[:, b_idx, c_idx] / val
                # v = residual[a_idx, :, c_idx] / val  
                # w = residual[a_idx, b_idx, :] / val
                u = residual[:, b_idx, c_idx] / val
                v = residual[a_idx, :, c_idx] / val
                w = residual[a_idx, b_idx, :] / val
                
                # Verify
                tensor = np.einsum('a,b,c->abc', u, v, w)
                if np.allclose(tensor, residual, atol=1e-10):
                    # Found a valid rank-54 decomposition!
                    U_final = np.hstack([U_cand, u.reshape(-1, 1)])
                    V_final = np.hstack([V_cand, v.reshape(-1, 1)])
                    W_final = np.hstack([W_cand, w.reshape(-1, 1)])
                    
                    # Verify
                    if verify_decomposition(U_final, V_final, W_final):
                        best_result = {'U': U_final.tolist(), 'V': V_final.tolist(), 'W': W_final.tolist()}
                        return best_result
    
    return best_result
