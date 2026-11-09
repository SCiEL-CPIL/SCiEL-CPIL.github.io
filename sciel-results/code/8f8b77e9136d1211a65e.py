"""Numerical NMF-based rank reduction for 3x4x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 48x48 matrix (flattened)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _flatten_tensor(T, n, m, p):
    """Flatten the 3D tensor into a 2D matrix for NMF."""
    # T has shape (n*m, m*p, p*n)
    # Flatten first two dims: (n*m * m*p, p*n)
    # Actually, for NMF we want M = U @ V^T where M has shape (n*m, m*p)
    # and the third dimension is the "output" dimension
    # Better: reshape to (n*m * m*p, p*n)
    return T.reshape(n * m * m * p, p * n)

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4_clean():
    """
    Clean construction of Strassen for 2x4x4.
    Returns U, V, W for the 2x4x4 sub-problem.
    """
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4, index i*4+j
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((8, rank))   # C: 4x2, index k*2+i
    
    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,1,1)], 1: [(1,2,-1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(0,3,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,0,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,1,-1)], 1: [(0,2,1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [], 1: [(1,2,1),(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        
        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1
    
    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook for 1x4x4: rank 16."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    
    for j in range(4):
        for k in range(4):
            idx = j * 4 + k
            U[j, idx] = 1
            V[idx, idx] = 1
            W[k, idx] = 1
    
    return U, V, W

def _combine_3x4x4():
    """Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) for total rank 44."""
    U2, V2, W2 = _build_strassen_2x4x4_clean()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Place 2x4x4 part (columns 0-27)
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    # Place 1x4x4 part (columns 28-43)
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank by identifying
    linear dependencies among the rank-1 terms and eliminating them.
    
    Strategy: For each pair of columns, check if one can be expressed
    as a linear combination of others. If so, merge them.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Build the tensor representation for each rank-1 term
    # Term c: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Flatten each term to a vector of size n*m*m*p*p*n
    nm = n * m
    mp = m * p
    pn = p * n
    total_dim = nm * mp * pn
    
    # For efficiency, work with the 48x48 flattened matrix
    # M = sum_c (U_c ⊗ V_c) * W_c^T ... this is complex
    # Simpler: check linear independence of the 44 vectors in R^(48*48)
    
    # Build matrix where each column is a flattened rank-1 tensor
    # T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c], flattened
    terms = []
    for c in range(current_rank):
        term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        terms.append(term.flatten())
    
    terms = np.array(terms).T  # shape: (total_dim, current_rank)
    
    # Try to find linear dependencies using SVD
    # If rank of terms < current_rank, we can reduce
    try:
        U_svd, S, Vt = np.linalg.svd(terms, full_matrices=False)
        effective_rank = np.sum(S > 1e-8 * S[0])
        
        if effective_rank < current_rank:
            # We have dependencies; try to project out
            # Keep only the top `target_rank` components
            if effective_rank >= target_rank:
                # Use truncated SVD to get a new basis
                U_new = U_svd[:, :target_rank]
                S_new = S[:target_rank]
                Vt_new = Vt[:target_rank, :]
                
                # Reconstruct: terms ≈ U_new @ diag(S_new) @ Vt_new
                # This gives a new decomposition with rank target_rank
                # But we need to convert back to U, V, W format
                # This is non-trivial; skip for now
                pass
    except Exception:
        pass
    
    # Alternative: greedy elimination
    # Try removing each column and checking if the tensor is still valid
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for c_remove in range(current_rank):
        if time.time() > deadline:
            break
        
        # Remove column c_remove
        mask = [i for i in range(current_rank) if i != c_remove]
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if _validate(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = current_rank - 1
            current_rank = best_rank
            U, V, W = U_test, V_test, W_test
            break  # Restart with the reduced rank
    
    if best_rank < target_rank:
        return best_U, best_V, best_W
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with the known rank-44 Strassen-based decomposition
    try:
        U, V, W = _combine_3x4x4()
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
            U_s, V_s, W_s = _schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
        
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try to reduce rank from 44 down to 43, 42, etc.
        for target_rank in range(43, 39, -1):
            if time.time() > deadline:
                break
            
            U_red, V_red, W_red = _try_rank_reduction(
                best_U, best_V, best_W, n, m, p, target_rank, deadline
            )
            
            if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
                best_rank = U_red.shape[1]
                best_U, best_V, best_W = U_red, V_red, W_red
                print(f"Reduced rank to {best_rank}")
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    except Exception as e:
        print(f"Error: {e}")
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
