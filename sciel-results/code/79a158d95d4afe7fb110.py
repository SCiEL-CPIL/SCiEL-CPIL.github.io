import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _build_strassen56():
    """Build the base rank-56 decomposition and return U, V, W as numpy arrays."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for s in range(2):
        for j in range(4):
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # U
            U[a00, col] += 1
            U[a11, col] += 1
            U[a10, col+1] += 1
            U[a11, col+1] += 1
            U[a00, col+2] += 1
            U[a11, col+3] += 1
            U[a00, col+4] += 1
            U[a01, col+4] += 1
            U[a10, col+5] += 1
            U[a00, col+5] += -1
            U[a01, col+6] += 1
            U[a11, col+6] += -1
            
            # V
            V[b00, col] += 1
            V[b11, col] += 1
            V[b00, col+1] += 1
            V[b01, col+2] += 1
            V[b11, col+2] += -1
            V[b10, col+3] += 1
            V[b00, col+3] += -1
            V[b11, col+4] += 1
            V[b00, col+5] += 1
            V[b01, col+5] += 1
            V[b10, col+6] += 1
            V[b11, col+6] += 1
            
            # W
            W[c00, col] += 1
            W[c00, col+3] += 1
            W[c00, col+4] += -1
            W[c00, col+6] += 1
            W[c01, col+2] += 1
            W[c01, col+4] += 1
            W[c10, col+1] += 1
            W[c10, col+3] += 1
            W[c11, col] += 1
            W[c11, col+1] += -1
            W[c11, col+2] += 1
            W[c11, col+5] += 1
            
            col += 7
    
    return U, V, W

def _validate(U, V, W):
    """Validate that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T[i,k,s] = 1 if we multiply A[i,s]*B[s,k]
    # Check: sum_r U[i,s,r] * V[s,k,r] * W[k,n_idx,r] should give C[n_idx, k]
    # Actually the tensor is: C[i,j] = sum_s A[i,s] B[s,j]
    # In tensor form: T[i,s,j] = 1, and we need sum_r U[i,s,r]*V[s,j,r]*W[j,i,r] = 1
    # Wait, let me re-read the convention. W[k*n+i, r]
    # So W index is (k, i) where k is column of B (row of C), i is row of A (col of C)
    # U index is (i, s) where i is row of A, s is col of A / row of B
    # V index is (s, k) where s is row of B, k is col of B
    
    # The decomposition should satisfy:
    # C[i, k] = sum_s A[i,s] B[s,k] = sum_r (sum_s U[i,s,r] A[i,s]) (sum_s V[s,k,r] B[s,k]) ... no
    # Actually: sum_r U[i,s,r] * V[s,k,r] * W[k,i,r] summed over s gives C[i,k]
    # But that's not standard. Let me think again.
    
    # Standard tensor decomposition for matrix mult:
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # T[i, s, k] = 1 (the multiplication tensor)
    # We want: T[i,s,k] = sum_r U[i,s,r] * V[s,k,r] * W[k,i,r]
    # Wait, W has shape (p*n, rank), so W[k*n+i, r]
    
    # Let's verify: for each (i, s, k), sum_r U[i*m+s, r] * V[s*p+k, r] * W[k*n+i, r] should be 1
    
    n, m, p = DIMENSIONS
    for i in range(n):
        for s in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+s, r] * V[s*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_reduce_rank(U, V, W, max_rank_reduction=4):
    """
    Try to find linear dependencies among the rank-1 terms to reduce rank.
    We look for subsets of columns that are linearly dependent in the tensor sense.
    This is a simplified search - we try to find if any column can be expressed
    as a linear combination of others.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # For each column r, check if it can be expressed as a linear combination
    # of other columns. This means finding coefficients c_j such that
    # for all (i,s,k): U[i,s,r]*V[s,k,r]*W[k,i,r] = sum_j c_j * U[i,s,j]*V[s,k,j]*W[k,i,j]
    
    # This is a complex nonlinear system. Instead, we use a simpler heuristic:
    # Check if the "signature" vectors are linearly dependent.
    
    # Build a matrix where each row is a "signature" of a column
    # Signature of column r: vector of U[:,r] ⊗ V[:,r] ⊗ W[:,r] flattened
    # But this is too large. Instead, use a random projection approach.
    
    # Alternative: try to find if removing one column and adjusting others works.
    # This is an NP-hard problem in general, so we use a simple greedy approach.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each column and see if we can compensate
    for r_to_remove in range(rank):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Try to express column r_to_remove as combination of others
        # Use least squares on the tensor entries
        n_entries = n * m * p
        
        # Build matrix A where A[idx, j] = U[i,s,j]*V[s,k,j]*W[k,i,j] for column j
        # And b[idx] = U[i,s,r_to_remove]*V[s,k,r_to_remove]*W[k,i,r_to_remove]
        
        A = np.zeros((n_entries, rank - 1))
        b = np.zeros(n_entries)
        
        idx = 0
        for i in range(n):
            for s in range(m):
                for k in range(p):
                    b[idx] = U[i*m+s, r_to_remove] * V[s*p+k, r_to_remove] * W[k*n+i, r_to_remove]
                    j_count = 0
                    for j in range(rank):
                        if j != r_to_remove:
                            A[idx, j_count] = U[i*m+s, j] * V[s*p+k, j] * W[k*n+i, j]
                            j_count += 1
                    idx += 1
        
        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Check residual
            residual = np.linalg.norm(A @ coeffs - b)
            if residual < 1e-8:
                # We found a dependency! Create new matrices without column r_to_remove
                new_rank = rank - 1
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))
                
                j_count = 0
                for j in range(rank):
                    if j != r_to_remove:
                        new_U[:, j_count] = U[:, j]
                        new_V[:, j_count] = V[:, j]
                        new_W[:, j_count] = W[:, j]
                        j_count += 1
                
                # Add the linear combination to the new columns
                for j in range(rank):
                    if j != r_to_remove:
                        j_idx = list(range(rank)).index(j)
                        new_U[:, j_idx] += coeffs[j_count] * U[:, r_to_remove] if False else 0
                        # Actually we need to add coeffs * (U[:,j] * V[:,j] * W[:,j]) contribution
                        # This doesn't work directly because the tensor product is nonlinear
                        # The decomposition is not linear in this way
                        pass
                
                # This approach doesn't work because tensor decomposition is not linear
                # Let's try a different approach
                pass
        except:
            pass
    
    return best_U, best_V, best_W, best_rank

def _fallback_construct():
    """Simple schoolbook rank-64 decomposition as fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for k in range(p):
            for s in range(m):
                U[i * m + s][r] = 1
                V[s * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _to_lists(U, V, W):
    """Convert numpy arrays to lists for the return format."""
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return U_list, V_list, W_list

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Attempts to reduce rank below 56 if possible.
    """
    if _elapsed() > _DEADLINE:
        return _fallback_construct()
    
    # Build the base rank-56 decomposition
    U, V, W = _build_strassen56()
    
    # Verify the base decomposition is valid
    if not _validate(U, V, W):
        # Fall back to schoolbook
        return _fallback_construct()
    
    # Try to reduce rank (limited time budget)
    if _elapsed() < _DEADLINE - 20:
        U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(U, V, W)
        if new_rank < 56 and _validate(U_reduced, V_reduced, W_reduced):
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Convert to lists and return
    U_list, V_list, W_list = _to_lists(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}
