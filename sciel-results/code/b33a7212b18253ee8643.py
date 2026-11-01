import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 145  # 150s limit, 5s safety margin

    n, m, p = DIMENSIONS
    nA = n * m  # 8
    nB = m * p  # 16
    nC = p * n  # 8

    # Build the matrix multiplication tensor T[i,j,k] = 1 iff C[i,j] depends on A[i,*] and B[*,j]
    # T has shape (nA, nB, nC)
    T = np.zeros((nA, nB, nC))
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i  # W[k*n+i, r] convention
                T[a_idx, b_idx, c_idx] = 1.0

    # Fallback: Schoolbook (Rank 32)
    rank_school = n * m * p
    U_school = np.zeros((nA, rank_school))
    V_school = np.zeros((nB, rank_school))
    W_school = np.zeros((nC, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Iterative sparsification search
    # Start from a known valid decomposition (schoolbook or Strassen-based),
    # then try to reduce rank by merging/zeroing columns via optimization.
    
    # First, build the Strassen rank-28 solution as a starting point
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 0]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, -1, 1]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, -1, 1, 0, -1, 0, 0]
    ], dtype=float)

    # Build rank-28 from Strassen blocks
    U_28 = np.zeros((nA, 28))
    V_28 = np.zeros((nB, 28))
    W_28 = np.zeros((nC, 28))
    
    subproblems = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2)
    ]
    
    for sp_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(subproblems):
        for s in range(7):
            r = sp_idx * 7 + s
            for local_row in range(2):
                for local_col in range(2):
                    global_col = a_col_off + local_col
                    a_flat = local_row * m + global_col
                    local_a = local_row * 2 + local_col
                    U_28[a_flat, r] = U_s[local_a, s]
                    
                    global_row = b_row_off + local_row
                    global_bcol = b_col_off + local_col
                    b_flat = global_row * p + global_bcol
                    local_b = local_row * 2 + local_col
                    V_28[b_flat, r] = V_s[local_b, s]
                    
                    global_ccol = c_col_off + local_col
                    c_flat = global_ccol * n + local_row
                    local_c = local_col * 2 + local_row
                    W_28[c_flat, r] = W_s[local_c, s]

    # Verify rank-28
    def verify_decomp(U, V, W, T, tol=1e-9):
        """Check if sum_s U[:,s] V[:,s] W[:,s] == T"""
        rank = U.shape[1]
        for s in range(rank):
            T -= np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
        return np.max(np.abs(T)) < tol

    T_check = T.copy()
    if verify_decomp(U_28.copy(), V_28.copy(), W_28.copy(), T_check):
        best_U, best_V, best_W = U_28.copy(), V_28.copy(), W_28.copy()
        best_rank = 28

    # Now attempt to reduce rank below 28 using structured search
    # Approach: Try to find a rank-r decomposition for r < best_rank
    # by randomly initializing factors and using iterative hard thresholding
    
    def try_find_decomp(target_rank, max_iters=200, seed=42):
        """Try to find a rank-target_rank decomposition via iterative optimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random sparse factors
        U = rng.randn(nA, target_rank) * 0.5
        V = rng.randn(nB, target_rank) * 0.5
        W = rng.randn(nC, target_rank) * 0.5
        
        # Iterative refinement: alternate fixing two factors and solving for the third
        # This is a form of alternating least squares with sparsity promotion
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                return None
            
            # Solve for W given U, V
            # T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
            # For each k: T[:,:,k] = sum_s (U[:,s] V[j,s]) W[k,s]
            # This is: for each (i,j), T[i,j,k] = sum_s (U[i,s]*V[j,s]) * W[k,s]
            
            # Build the system: T_reshaped[i,j,k] = sum_s M[i,j,s] * W[k,s]
            # where M[i,j,s] = U[i,s]*V[j,s]
            # So for fixed k: T[:,:,k] = W[k,:] @ M[:,:,k]^T ... no
            # T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
            # Let B_s = U[:,s] * V[:,s] (outer product, shape nA x nB)
            # T[:,:,k] = sum_s W[k,s] * B_s
            
            # For each k, we need: T[:,:,k] = sum_s W[k,s] * (U[:,s] ⊗ V[:,s])
            # This is a linear system in W[k,s] for s=0..target_rank-1
            
            # Stack: T_flat[k, idx] = sum_s W[k,s] * B_s_flat[idx]
            # So T_flat = W @ B_flat^T where B_flat has shape (target_rank, nA*nB)
            
            B_flat = np.zeros((target_rank, nA * nB))
            for s in range(target_rank):
                B_flat[s, :] = (U[:, s] * V[:, s]).flatten()
            
            T_flat = T.flatten().reshape(nC, nA * nB)  # T[i,j,k] -> T_flat[k, i*nB+j]
            # Wait, T is indexed [i,j,k], so T.flatten() in C order gives [i0j0k0, i0j0k1, ..., i0j1k0, ...]
            # We need T_flat[k, ij] = T[i,j,k]
            T_reshaped = T.transpose(2, 0, 1).reshape(nC, nA * nB)
            
            # Solve: T_reshaped = W @ B_flat.T
            # W = T_reshaped @ B_flat.T @ (B_flat @ B_flat.T)^{-1}
            try:
                gram = B_flat @ B_flat.T  # (target_rank, target_rank)
                gram += 1e-10 * np.eye(target_rank)
                rhs = T_reshaped @ B_flat  # (nC, target_rank)
                W_new = np.linalg.solve(gram.T, rhs.T).T  # (nC, target_rank)
            except np.linalg.LinAlgError:
                W_new = W
            
            # Sparsify W: zero out small entries
            W = W_new
            W[np.abs(W) < 0.1] = 0.0
            
            # Similarly solve for U and V
            # T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
            # For fixed i: T[i,j,k] = sum_s U[i,s] * (V[j,s]*W[k,s])
            # Let C_s = V[:,s] * W[:,s] (outer, nB x nC)
            # T[i,:,:] = sum_s U[i,s] * C_s
            C_flat = np.zeros((target_rank, nB * nC))
            for s in range(target_rank):
                C_flat[s, :] = (V[:, s] * W[:, s]).flatten()
            
            T_reshaped_i = T.transpose(0, 1, 2).reshape(nA, nB * nC)
            try:
                gram_c = C_flat @ C_flat.T
                gram_c += 1e-10 * np.eye(target_rank)
                rhs_c = T_reshaped_i @ C_flat
                U_new = np.linalg.solve(gram_c.T, rhs_c.T).T
            except np.linalg.LinAlgError:
                U_new = U
            U = U_new
            U[np.abs(U) < 0.1] = 0.0
            
            # For V:
            # T[i,j,k] = sum_s V[j,s] * (U[i,s]*W[k,s])
            D_flat = np.zeros((target_rank, nA * nC))
            for s in range(target_rank):
                D_flat[s, :] = (U[:, s] * W[:, s]).flatten()
            
            T_reshaped_j = T.transpose(1, 0, 2).reshape(nB, nA * nC)
            try:
                gram_d = D_flat @ D_flat.T
                gram_d += 1e-10 * np.eye(target_rank)
                rhs_d = T_reshaped_j @ D_flat
                V_new = np.linalg.solve(gram_d.T, rhs_d.T).T
            except np.linalg.LinAlgError:
                V_new = V
            V = V_new
            V[np.abs(V) < 0.1] = 0.0
        
        # Check if the decomposition is valid
        T_check = T.copy()
        for s in range(target_rank):
            T_check -= np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
        
        if np.max(np.abs(T_check)) < 1e-8:
            # Project to half-integers
            U_proj = np.round(U * 2) / 2
            V_proj = np.round(V * 2) / 2
            W_proj = np.round(W * 2) / 2
            
            T_check2 = T.copy()
            for s in range(target_rank):
                T_check2 -= np.einsum('i,j,k->ijk', U_proj[:, s], V_proj[:, s], W_proj[:, s])
            
            if np.max(np.abs(T_check2)) < 1e-8:
                return U_proj, V_proj, W_proj
            
            # If projection fails, try integer projection
            U_proj = np.round(U)
            V_proj = np.round(V)
            W_proj = np.round(W)
            
            T_check3 = T.copy()
            for s in range(target_rank):
                T_check3 -= np.einsum('i,j,k->ijk', U_proj[:, s], V_proj[:, s], W_proj[:, s])
            
            if np.max(np.abs(T_check3)) < 1e-8:
                return U_proj, V_proj, W_proj
        
        return None

    # Try ranks from 27 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        for seed in range(10):
            if time.time() - start_time > deadline - 5:
                break
            result = try_find_decomp(target_rank, max_iters=100, seed=seed + target_rank * 100)
            if result is not None:
                U_f, V_f, W_f = result
                # Count actual non-zero columns
                U_f = U_f[:, np.any(U_f != 0, axis=0)]
                V_f = V_f[:, np.any(V_f != 0, axis=0)]
                W_f = W_f[:, np.any(W_f != 0, axis=0)]
                
                # Verify
                T_check = T.copy()
                for s in range(U_f.shape[1]):
                    T_check -= np.einsum('i,j,k->ijk', U_f[:, s], V_f[:, s], W_f[:, s])
                if np.max(np.abs(T_check)) < 1e-8:
                    best_U, best_V, best_W = U_f, V_f, W_f
                    best_rank = U_f.shape[1]
                    found = True
                    break
        
        if found:
            # Try even lower ranks
            continue
    
    # Also try a direct structured approach: 
    # The 2x4x4 tensor has a Kronecker-like structure.
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x4) = [[B11, B12], [B21, B22]]
    # C (2x4) = A1*B11 + A2*B21 (for first 2 cols), A1*B12 + A2*B22 (for last 2 cols)
    # This gives 4 independent 2x2x2 multiplications, each rank 7, total 28.
    # But maybe we can share some rank terms between blocks?
    
    # Alternative: use a different block partition or overlapping structure
    # Try rank 27 by allowing one shared term
    
    if time.time() - start_time < deadline - 15:
        # Try a more targeted search: start from rank-28 and try to zero out one column
        # by adjusting others
        U_start = best_U.copy()
        V_start = best_V.copy()
        W_start = best_W.copy()
        
        for col_to_kill in range(28):
            if time.time() - start_time > deadline - 10:
                break
            
            # Try to absorb column col_to_kill into other columns
            # by modifying U, V, W slightly
            U_mod = U_start.copy()
            V_mod = V_start.copy()
            W_mod = W_start.copy()
            
            # Zero out the column
            U_mod[:, col_to_kill] = 0
            V_mod[:, col_to_kill] = 0
            W_mod[:, col_to_kill] = 0
            
            # Try to fix the error by adjusting other columns
            # Error tensor: E = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # We want to redistribute this error across other columns
            
            # Simple approach: add a fraction of the killed column to another column
            for other_col in range(28):
                if other_col == col_to_kill:
                    continue
                
                # Try adding the killed column's contribution to other_col
                U_test = U_mod.copy()
                V_test = V_mod.copy()
                W_test = W_mod.copy()
                
                # Scale: add alpha * U[:,c] to U[:,other], etc.
                # But this changes the product nonlinearly.
                # Instead, just check if simply removing the column still works (it won't)
                # Skip this - too complex for the time budget
        
        # Try a completely different approach: 
        # Use the fact that 2x4x4 can be seen as a batch of 2x2x2 with shared structure
        # and search for a rank-27 decomposition with one shared term
        
        # For now, return what we have
        pass

    if time.time() - start_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
