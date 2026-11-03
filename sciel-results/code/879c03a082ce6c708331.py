"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def _build_strassen_54(n, m, p):
    """Build the rank-54 Strassen-based decomposition for 2x5*5x6."""
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: direct multiplication for 2x1 * 1x6 = 2x6
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    return U, V, W, rank

def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition correctness."""
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    result = np.einsum('ilr,lkr,kir->ilk', U_t, V_t, W_t)
    return np.allclose(result, 1.0, atol=1e-9)

def _try_single_elimination(U, V, W, n, m, p, rank, target):
    """Try to eliminate one term via linear combination of others."""
    nmp = n * m * p
    
    target_vec = np.einsum('a,b,c->abc', U[:, target], V[:, target], W[:, target]).reshape(nmp)
    
    other_cols = [r for r in range(rank) if r != target]
    n_others = len(other_cols)
    
    if n_others == 0:
        return None
    
    A_mat = np.zeros((nmp, n_others))
    for idx, r in enumerate(other_cols):
        contrib = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(nmp)
        A_mat[:, idx] = contrib
    
    try:
        c, _, _, _ = np.linalg.lstsq(A_mat, target_vec, rcond=None)
        if np.allclose(A_mat @ c, target_vec, atol=1e-9):
            new_rank = rank - 1
            new_U = np.zeros((n * m, new_rank))
            new_V = np.zeros((m * p, new_rank))
            new_W = np.zeros((p * n, new_rank))
            
            for idx, r in enumerate(other_cols):
                new_U[:, idx] = U[:, r] + c[idx] * U[:, target]
                new_V[:, idx] = V[:, r] + c[idx] * V[:, target]
                new_W[:, idx] = W[:, r] + c[idx] * W[:, target]
            
            return new_U, new_V, new_W, new_rank
    except:
        pass
    
    return None

def _try_pair_elimination(U, V, W, n, m, p, rank, t1, t2):
    """Try to eliminate two terms simultaneously."""
    nmp = n * m * p
    
    v1 = np.einsum('a,b,c->abc', U[:, t1], V[:, t1], W[:, t1]).reshape(nmp)
    v2 = np.einsum('a,b,c->abc', U[:, t2], V[:, t2], W[:, t2]).reshape(nmp)
    
    other_cols = [r for r in range(rank) if r not in (t1, t2)]
    n_others = len(other_cols)
    
    if n_others < 2:
        return None
    
    A_mat = np.zeros((nmp, n_others))
    for idx, r in enumerate(other_cols):
        contrib = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(nmp)
        A_mat[:, idx] = contrib
    
    try:
        c1, _, _, _ = np.linalg.lstsq(A_mat, v1, rcond=None)
        c2, _, _, _ = np.linalg.lstsq(A_mat, v2, rcond=None)
        
        if np.allclose(A_mat @ c1, v1, atol=1e-9) and np.allclose(A_mat @ c2, v2, atol=1e-9):
            new_rank = rank - 2
            new_U = np.zeros((n * m, new_rank))
            new_V = np.zeros((m * p, new_rank))
            new_W = np.zeros((p * n, new_rank))
            
            for idx, r in enumerate(other_cols):
                new_U[:, idx] = U[:, r] + c1[idx] * U[:, t1] + c2[idx] * U[:, t2]
                new_V[:, idx] = V[:, r] + c1[idx] * V[:, t1] + c2[idx] * V[:, t2]
                new_W[:, idx] = W[:, r] + c1[idx] * W[:, t1] + c2[idx] * W[:, t2]
            
            return new_U, new_V, new_W, new_rank
    except:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    U, V, W, rank = _build_strassen_54(n, m, p)
    
    if not _verify(U, V, W, n, m, p, rank):
        pass  # Should always be valid
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Single-term elimination attempts
    for target in range(rank):
        if time.time() > deadline:
            break
        
        result = _try_single_elimination(U, V, W, n, m, p, rank, target)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if _verify(new_U, new_V, new_W, n, m, p, new_rank):
                if new_rank < best_rank:
                    best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                    U, V, W, rank = new_U, new_V, new_W, new_rank
                    break
    
    # If we improved, try again
    if time.time() < deadline and rank < 54:
        for target in range(rank):
            if time.time() > deadline:
                break
            
            result = _try_single_elimination(U, V, W, n, m, p, rank, target)
            if result is not None:
                new_U, new_V, new_W, new_rank = result
                if _verify(new_U, new_V, new_W, n, m, p, new_rank):
                    if new_rank < best_rank:
                        best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                        U, V, W, rank = new_U, new_V, new_W, new_rank
                        break
    
    # Pair elimination attempts on the best found so far
    if time.time() < deadline:
        U2, V2, W2, rank2 = best_U.copy(), best_V.copy(), best_W.copy(), best_rank
        
        # Try pairs of terms that might be redundant
        tried_pairs = set()
        for i in range(rank2):
            if time.time() > deadline:
                break
            for j in range(i + 1, rank2):
                if time.time() > deadline:
                    break
                
                pair_key = (i, j)
                if pair_key in tried_pairs:
                    continue
                tried_pairs.add(pair_key)
                
                result = _try_pair_elimination(U2, V2, W2, n, m, p, rank2, i, j)
                if result is not None:
                    new_U, new_V, new_W, new_rank = result
                    if _verify(new_U, new_V, new_W, n, m, p, new_rank):
                        if new_rank < best_rank:
                            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                            U2, V2, W2, rank2 = new_U, new_V, new_W, new_rank
                            # Restart pair search with new rank
                            tried_pairs = set()
    
    # Convert to list-of-lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            lrow = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # If it's very close to an integer, make it integer
                if abs(rounded - round(rounded)) < 1e-9:
                    lrow.append(int(round(rounded)))
                else:
                    lrow.append(rounded)
            result.append(lrow)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

