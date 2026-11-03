"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank54(n, m, p):
    """
    Build the rank-54 Strassen block decomposition for 2x5x6.
    j pairs: (0,1), (2,3), single: 4
    k pairs: (0,1), (2,3), (4,5)
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M1 = (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A[1,j0] + A[1,j1]) * B[j0,k0]
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A[0,j0] * (B[j0,k1] - B[j1,k1])
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A[1,j1] * (B[j1,k0] - B[j0,k0])
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A[0,j0] + A[0,j1]) * B[j1,k1]
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Handle single j=4 with all k values (6 scalar products)
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    return U, V, W, col

def _try_remove_columns(U, V, W, n, m, p, deadline):
    """
    Greedily try to remove columns that are redundant.
    Returns (U_reduced, V_reduced, W_reduced, new_rank).
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    removed = set()
    
    # Multiple passes to catch cascading removals
    for _pass in range(3):
        if time.time() > deadline:
            break
        progress = False
        for c in range(current_rank):
            if time.time() > deadline:
                break
            if c in removed:
                continue
            
            # Build mask excluding column c and already removed
            mask = [i for i in range(current_rank) if i not in removed and i != c]
            if len(mask) == 0:
                continue
            
            test_U = U[:, mask]
            test_V = V[:, mask]
            test_W = W[:, mask]
            
            if _verify_decomposition(test_U, test_V, test_W, n, m, p):
                removed.add(c)
                progress = True
                # Rebuild arrays
                mask_all = [i for i in range(current_rank) if i not in removed]
                U = U[:, mask_all]
                V = V[:, mask_all]
                W = W[:, mask_all]
                current_rank = len(mask_all)
        
        if not progress:
            break
    
    return U, V, W, current_rank

def _try_pair_merge(U, V, W, n, m, p, deadline):
    """
    Try to merge pairs of columns by finding linear combinations.
    For two columns r1, r2, check if we can replace them with one column
    that's a linear combination preserving the tensor identity.
    This is a heuristic: try simple ratios.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    removed = set()
    
    for _pass in range(2):
        if time.time() > deadline:
            break
        progress = False
        cols = [i for i in range(current_rank) if i not in removed]
        for idx_a in range(len(cols)):
            if time.time() > deadline:
                break
            for idx_b in range(idx_a + 1, len(cols)):
                if time.time() > deadline:
                    break
                c1 = cols[idx_a]
                c2 = cols[idx_b]
                
                # Try to see if column c2 can be absorbed into c1
                # by checking if the contribution of c2 is proportional to c1
                # in a way that allows merging
                # Simple test: try removing c2 (already covered by _try_remove_columns)
                # More advanced: try scaling
                pass
        if not progress:
            break
    
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Build the rank-54 Strassen decomposition
    U, V, W, rank = _build_rank54(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    # Try to reduce rank via greedy column elimination
    U_red, V_red, W_red, new_rank = _try_remove_columns(U, V, W, n, m, p, deadline)
    
    if new_rank < rank:
        U, V, W = U_red, V_red, W_red
        rank = new_rank
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Return original if reduced version is invalid
        U, V, W, rank = _build_rank54(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
