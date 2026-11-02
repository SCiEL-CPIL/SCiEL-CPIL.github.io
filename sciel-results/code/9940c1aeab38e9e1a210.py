"""Rewritten program for 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _verify(U, V, W, T_flat, pn):
    """
    Verify if U, V, W (integers/half-integers) exactly decompose the target tensor.
    T_flat is the target tensor reshaped to (pn, nm*mp).
    """
    # Scale to integers to avoid floating point issues
    U2 = np.array(U, dtype=np.float64) * 2
    V2 = np.array(V, dtype=np.float64) * 2
    W2 = np.array(W, dtype=np.float64) * 2
    
    # Check if they are integers (i.e., original values were half-integers)
    if np.max(np.abs(U2 - np.round(U2))) > 1e-9: return False
    if np.max(np.abs(V2 - np.round(V2))) > 1e-9: return False
    if np.max(np.abs(W2 - np.round(W2))) > 1e-9: return False
    
    U2 = np.round(U2).astype(np.int64)
    V2 = np.round(V2).astype(np.int64)
    W2 = np.round(W2).astype(np.int64)
    
    # Compute T_computed = sum_r U[:,r] V[:,r] W[:,r]^T
    # T_computed[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # We need to check if T_computed equals 8 * T_target
    # T_target is stored in T_flat as (pn, nm*mp) which is (k, i*m+j)
    
    # Efficient check using einsum
    # T_comp[i, j, k]
    T_comp = np.einsum('ir,jr,kr->ijk', U2, V2, W2, optimize=True)
    
    # T_comp is (nm, mp, pn)
    # T_flat is (pn, nm*mp) -> transpose to (nm, mp, pn) for comparison?
    # T_flat[k, i*mp+j] = T[i, j, k]
    # So T_target[i, j, k] = T_flat[k, i*mp+j]
    
    T_target = T_flat.T.reshape(T_comp.shape)
    
    # The decomposition is U V W^T = T.
    # Our U, V, W are scaled by 2.
    # So U2 V2 W2^T should be 8 * T.
    if np.max(np.abs(T_comp - 8 * T_target)) > 1e-6:
        return False
    
    return True

def _try_rank(target_rank, n, m, p, T_flat, deadline, num_restarts=20):
    """Try to find a valid decomposition of given rank."""
    if time.time() > deadline - 10:
        return None
        
    nm, mp, pn = n*m, m*p, p*n
    T_target = T_flat.T.reshape(nm, mp, pn)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            return None
            
        # Random initialization
        U = np.random.randint(-1, 2, size=(nm, target_rank))
        V = np.random.randint(-1, 2, size=(mp, target_rank))
        
        for als_iter in range(10):
            if time.time() > deadline - 5:
                return None
                
            # Solve for W
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed U, V:
            # T_flat[k, idx] = sum_r W[k,r] * (U[i,r] V[j,r])
            # Let B[r, idx] = U[i,r] V[j,r]
            # T_flat = W @ B.T
            
            B = np.zeros((target_rank, nm * mp))
            for r in range(target_rank):
                B[r] = np.outer(U[:, r], V[:, r]).flatten()
            
            try:
                W_sol, residuals, rank_sv, sv = np.linalg.lstsq(B.T, T_flat.T, rcond=1e-12)
            except:
                continue
                
            W_sol = W_sol.T # (pn, target_rank)
            
            # Quantize W to half-integers
            W_q = np.round(W_sol * 2) / 2.0
            
            # Verify
            if _verify(U, V, W_q, T_flat, pn):
                return U.tolist(), V.tolist(), W_q.tolist()
                
            # Update U
            # T[i,j,k] = sum_r U[i,r] (V[j,r] W[k,r])
            # For fixed V, W:
            # T_flat[k, idx] = sum_r U[i,r] * C[r, idx] where C[r, idx] = V[j,r] W[k,r]
            # This is a bit tricky to map to standard lstsq directly on T_flat.
            # T[i, j, k] = U[i, r] * (V[j, r] W[k, r])
            # Let's reshape T to (nm, mp*pn)
            T_reshaped = T_target.reshape(nm, mp * pn)
            
            C = np.zeros((target_rank, mp * pn))
            for r in range(target_rank):
                C[r] = np.outer(V[:, r], W_q[:, r]).flatten()
                
            try:
                U_sol, residuals, rank_sv, sv = np.linalg.lstsq(C.T, T_reshaped.T, rcond=1e-12)
            except:
                continue
            U_sol = U_sol.T
            U_q = np.round(U_sol * 2) / 2.0
            
            # Update V
            T_reshaped_v = T_target.reshape(mp, nm * pn)
            D = np.zeros((target_rank, nm * pn))
            for r in range(target_rank):
                D[r] = np.outer(U_q[:, r], W_q[:, r]).flatten()
                
            try:
                V_sol, residuals, rank_sv, sv = np.linalg.lstsq(D.T, T_reshaped_v.T, rcond=1e-12)
            except:
                continue
            V_sol = V_sol.T
            V_q = np.round(V_sol * 2) / 2.0
            
            # Assign back
            U = U_q
            V = V_q
            W_q = W_q # W was updated in the first step, but we used the quantized W for U/V update?
            # Actually, we should use the quantized values for the next iteration's design matrix.
            
            # Verify with quantized U, V, W
            if _verify(U, V, W_q, T_flat, pn):
                return U.tolist(), V.tolist(), W_q.tolist()

    return None

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U_sb[a*m + b, col] = 1
                V_sb[b*p + c, col] = 1
                W_sb[c*n + a, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 100 down to 49
    # We can likely find rank 100 very fast.
    # Rank 50 might take longer.
    for target_rank in range(100, 48, -1):
        if time.time() > deadline - 20:
            break
            
        # Increase restarts for lower ranks?
        restarts = 10 if target_rank > 80 else 50
        
        result = _try_rank(target_rank, n, m, p, T_flat, deadline, num_restarts=restarts)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            # If we find 49, we can stop
            if target_rank == 49:
                break
            
    return best_result

# EVOLVE-BLOCK-END
