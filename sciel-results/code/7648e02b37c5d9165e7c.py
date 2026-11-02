"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_multiplication_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for (i,j,k) with j matching."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    T = _build_multiplication_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Compute the reconstructed tensor
    # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2_block(n, m, p, block_offset):
    """
    Build Strassen decomposition for 2x2 * 2x2 block.
    block_offset: which 2-column block of the 2x4 matrix (0 or 1).
    Returns (U_block, V_block, W_block) each of shape (n*m, 7), (m*p, 7), (p*n, 7)
    but only the relevant columns are non-zero.
    """
    U_block = np.zeros((4, 7))
    V_block = np.zeros((8, 7))
    W_block = np.zeros((8, 7))
    
    if block_offset == 0:
        # Block 0: k in {0,1}
        # V rows: j*4+k where k=0,1 -> rows 0,1 (j=0), 4,5 (j=1)
        v_rows = [0, 1, 4, 5]  # (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
        # W rows: k*2+i where k=0,1
        # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        w_rows = [0, 1, 2, 3]  # (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    else:
        # Block 1: k in {2,3}
        # V rows: j*4+k where k=2,3 -> rows 2,3 (j=0), 6,7 (j=1)
        v_rows = [2, 3, 6, 7]
        # W rows: k*2+i where k=2,3
        # k=2,i=0 -> 4; k=2,i=1 -> 5; k=3,i=0 -> 6; k=3,i=1 -> 7
        w_rows = [4, 5, 6, 7]
    
    # Strassen factors for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_block[0,0] = 1; U_block[3,0] = 1
    V_block[v_rows[0],0] = 1; V_block[v_rows[3],0] = 1
    W_block[w_rows[0],0] = 1  # C00
    W_block[w_rows[3],0] = 1  # C11
    
    # Term 1: M2 = (A10+A11)B00
    U_block[2,1] = 1; U_block[3,1] = 1
    V_block[v_rows[0],1] = 1
    W_block[w_rows[2],1] = 1  # C10
    W_block[w_rows[3],1] = -1  # C11
    
    # Term 2: M3 = A00(B01-B11)
    U_block[0,2] = 1
    V_block[v_rows[1],2] = 1; V_block[v_rows[3],2] = -1
    W_block[w_rows[1],2] = 1  # C01
    W_block[w_rows[3],2] = 1  # C11
    
    # Term 3: M4 = A11(B10-B00)
    U_block[3,3] = 1
    V_block[v_rows[2],3] = 1; V_block[v_rows[0],3] = -1
    W_block[w_rows[0],3] = 1  # C00
    W_block[w_rows[2],3] = 1  # C10
    
    # Term 4: M5 = (A00+A10)B11
    U_block[0,4] = 1; U_block[2,4] = 1
    V_block[v_rows[3],4] = 1
    W_block[w_rows[1],4] = 1  # C01
    W_block[w_rows[0],4] = -1  # C00
    
    # Term 5: M6 = (A01-A00)(B00+B10)
    U_block[0,5] = -1; U_block[1,5] = 1
    V_block[v_rows[0],5] = 1; V_block[v_rows[2],5] = 1
    W_block[w_rows[3],5] = 1  # C11
    
    # Term 6: M7 = (A10-A11)(B01+B11)
    U_block[2,6] = 1; U_block[3,6] = -1
    V_block[v_rows[1],6] = 1; V_block[v_rows[3],6] = 1
    W_block[w_rows[0],6] = 1  # C00
    
    return U_block, V_block, W_block

def _try_reduce_rank(U, V, W, n, m, p, max_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 tensors.
    If we have rank r, we can try to express one term as a linear combination of others.
    """
    r = U.shape[1]
    if r <= max_rank:
        return U, V, W, r
    
    # Build the tensor from the decomposition
    T = _build_multiplication_tensor(n, m, p)
    
    # Try to find a linear combination of rank-1 tensors that equals zero
    # This is equivalent to finding a null space vector
    # For rank r, we have r rank-1 tensors. If they are linearly dependent,
    # we can reduce the rank.
    
    # Flatten the tensor
    T_flat = T.flatten()
    size = len(T_flat)
    
    # Build the matrix of rank-1 tensors
    # Each column is one rank-1 tensor flattened
    M = np.zeros((size, r))
    for c in range(r):
        M[:, c] = np.outer(np.outer(U[:, c], V[:, c]), W[:, c]).flatten()
    
    # Find null space
    try:
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, r
    
    # Find vectors in the null space
    tol = 1e-10
    null_mask = S < tol
    null_space = Vt[null_mask]
    
    if null_space.shape[0] == 0:
        return U, V, W, r
    
    # For each null space vector, try to eliminate one component
    # We want to find a null vector with a non-zero component we can eliminate
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    for ns in null_space:
        # Find the largest component to eliminate
        idx = np.argmax(np.abs(ns))
        if abs(ns[idx]) < 1e-10:
            continue
        
        # Scale so that the eliminated component has coefficient 1
        scale = ns[idx]
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        # For each other component j, subtract (ns[j]/scale) * component j from component idx
        for j in range(r):
            if j == idx:
                continue
            coeff = ns[j] / scale
            new_U[:, idx] -= coeff * U[:, j]
            new_V[:, idx] -= coeff * V[:, j]
            new_W[:, idx] -= coeff * W[:, j]
        
        # Remove the idx-th component
        keep = [k for k in range(r) if k != idx]
        new_U = new_U[:, keep]
        new_V = new_V[:, keep]
        new_W = new_W[:, keep]
        
        new_r = len(keep)
        if new_r < best_rank:
            # Verify
            if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p, tol=1e-8):
                best_rank = new_r
                best_U, best_V, best_W = new_U, new_V, new_W
    
    return best_U, best_V, best_W, best_rank

def _try_pairwise_cancellation(U, V, W, n, m, p, deadline):
    """
    Try to find pairs of terms from different blocks that can be merged or cancelled.
    If term i from block 0 and term j from block 1 have related structures,
    we might be able to combine them.
    """
    r = U.shape[1]
    if r <= 1:
        return U, V, W, r
    
    T = _build_multiplication_tensor(n, m, p)
    
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    # Try all pairs of terms to see if one can be expressed as a combination of the other
    # This is a simplified approach - try to eliminate one term using another
    for i in range(r):
        if time.time() > deadline - 5:
            break
        for j in range(r):
            if i == j:
                continue
            
            # Try to express term i as a scalar multiple of term j
            # This only works if the rank-1 tensors are proportional
            Ui = U[:, i]
            Uj = U[:, j]
            Vi = V[:, i]
            Vj = V[:, j]
            Wi = W[:, i]
            Wj = W[:, j]
            
            # Check if Ui is proportional to Uj
            if np.linalg.norm(Uj) < 1e-10:
                continue
            
            # Try to find alpha such that Ui = alpha * Uj
            # Use least squares
            try:
                alpha = np.dot(Ui, Uj) / np.dot(Uj, Uj)
                residual = Ui - alpha * Uj
                if np.linalg.norm(residual) > 1e-8:
                    continue
            except:
                continue
            
            # Similarly for V and W
            try:
                beta = np.dot(Vi, Vj) / np.dot(Vj, Vj)
                residual = Vi - beta * Vj
                if np.linalg.norm(residual) > 1e-8:
                    continue
            except:
                continue
            
            try:
                gamma = np.dot(Wi, Wj) / np.dot(Wj, Wj)
                residual = Wi - gamma * Wj
                if np.linalg.norm(residual) > 1e-8:
                    continue
            except:
                continue
            
            # If alpha == beta == gamma, then term i is a scalar multiple of term j
            if abs(alpha - beta) < 1e-8 and abs(beta - gamma) < 1e-8:
                # Merge: subtract term i from term j
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                
                new_U[:, j] -= alpha * Ui
                new_V[:, j] -= alpha * Vi
                new_W[:, j] -= alpha * Wi
                
                keep = [k for k in range(r) if k != i]
                new_U = new_U[:, keep]
                new_V = new_V[:, keep]
                new_W = new_W[:, keep]
                
                new_r = len(keep)
                if new_r < best_rank:
                    if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p, tol=1e-8):
                        best_rank = new_r
                        best_U, best_V, best_W = new_U, new_V, new_W
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Start with rank 14 (two Strassen blocks)
    U0, V0, W0 = _strassen_2x2_block(n, m, p, 0)
    U1, V1, W1 = _strassen_2x2_block(n, m, p, 1)
    
    U = U0 + U1
    V = V0 + V1
    W = W0 + W1
    
    current_rank = 14
    
    # Try to reduce rank via null space analysis
    if time.time() < deadline - 30:
        U, V, W, current_rank = _try_reduce_rank(U, V, W, n, m, p, 13, deadline)
    
    # Try pairwise cancellation
    if time.time() < deadline - 10:
        U, V, W, current_rank = _try_pairwise_cancellation(U, V, W, n, m, p, deadline)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Final verification
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the original rank-14 decomposition
    U_list = [[float(x) for x in row] for row in (U0 + U1)]
    V_list = [[float(x) for x in row] for row in (V0 + V1)]
    W_list = [[float(x) for x in row] for row in (W0 + W1)]
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Ultimate fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
