"""Rewritten program: structural change using algebraic tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    # Helper to verify a decomposition
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T_rec = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_target, atol=tol, rtol=0)
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for val in mat.flatten():
                v2 = val * 2
                if abs(v2 - round(v2)) > tol:
                    return False
        return True
    
    def round_to_half_integer(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Schoolbook decomposition as fallback (rank 40)
    rank_schoolbook = n * m * p  # 40
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strategy: Try to find lower rank decompositions using algebraic methods
    # The key insight: matrix multiplication tensor has special structure.
    # For (n,m,p) = (2,4,5), we know:
    # - Schoolbook rank = 40
    # - Strassen-like algorithms may reduce rank
    # - The bilinear complexity of 2x4 by 4x5 multiplication
    
    # Approach 1: Use the fact that we can decompose into smaller subproblems
    # A 2x4 by 4x5 multiplication can be viewed as:
    # [A1 A2] * B where A1 is 2x4, A2 is 2x4, B is 4x5
    # = A1*B + A2*B (but this doesn't directly help)
    
    # Better approach: Use known results about bilinear complexity
    # For small matrices, we can try to find decompositions by solving
    # the system of equations directly.
    
    # Let's try a systematic search for ranks below 40
    # We'll use a combination of:
    # 1. Randomized ALS with exact verification
    # 2. Structured initialization based on tensor symmetries
    
    def try_rank(rank, num_restarts=20, max_iter=500):
        """Try to find a rank decomposition using ALS with verification."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each row i, solve for U[i,:]
                # T[i, :, :] = sum_c U[i,c] * outer(V[:,c], W[:,c])
                # This is a linear system in U[i,:]
                for i in range(nm):
                    # Build the linear system: U[i,:] * M = T[i,:,:].reshape(-1)
                    # where M[c, (j,k)] = V[j,c] * W[k,c]
                    M = np.zeros((rank, mp * pn))
                    for c in range(rank):
                        M[c, :] = np.outer(V[:, c], W[:, c]).flatten()
                    rhs = T_target[i, :, :].flatten()
                    # Solve least squares
                    try:
                        U[i, :], residuals, rank_sol, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        break
                
                # Update V
                for j in range(mp):
                    M = np.zeros((rank, nm * pn))
                    for c in range(rank):
                        M[c, :] = np.outer(U[:, c], W[:, c]).flatten()
                    rhs = T_target[:, j, :].reshape(nm, pn).T.reshape(-1)
                    # Actually T_target[:, j, :] has shape (nm, pn), we need to reshape properly
                    # T_target[i, j, k] = sum_c U[i,c]*V[j,c]*W[k,c]
                    # For fixed j: T_target[:, j, :] is (nm, pn)
                    # We need: for each (i,k), T_target[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
                    # So the system is: V[j,:] * M' = rhs'
                    # where M'[c, (i,k)] = U[i,c] * W[k,c]
                    M = np.zeros((rank, nm * pn))
                    for c in range(rank):
                        M[c, :] = np.outer(U[:, c], W[:, c]).flatten()
                    rhs = T_target[:, j, :].reshape(nm, pn).T.reshape(-1)  # (pn*nm,)
                    # Wait, this indexing is getting complex. Let me use einsum approach.
                    try:
                        V[j, :], residuals, rank_sol, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        break
                
                # Update W
                for k in range(pn):
                    M = np.zeros((rank, nm * mp))
                    for c in range(rank):
                        M[c, :] = np.outer(U[:, c], V[:, c]).flatten()
                    rhs = T_target[:, :, k].reshape(nm, mp).T.reshape(-1)
                    try:
                        W[k, :], residuals, rank_sol, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        break
                
                if it % 50 == 0:
                    # Check convergence
                    if verify(U, V, W, tol=1e-8):
                        converged = True
                        break
            
            if converged:
                # Check half-integer constraint
                U_hi = round_to_half_integer(U)
                V_hi = round_to_half_integer(V)
                W_hi = round_to_half_integer(W)
                if verify(U_hi, V_hi, W_hi, tol=1e-8):
                    return {"U": U_hi, "V": V_hi, "W": W_hi}
                elif check_half_integer(U, V, W, tol=1e-6):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 35 down to 25
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, num_restarts=10, max_iter=200)
        if result is not None:
            best_result = result
            best_rank = target_rank
            print(f"Found rank {target_rank} decomposition!")
            break
    
    # If no better rank found, try the structured Kronecker product approach
    # mentioned in the strategy: treat as tensor product of 2x1 and 1x4 by 4x5
    if best_rank == rank_schoolbook and time.time() < deadline - 25:
        # For 1x4 by 4x5, the schoolbook rank is 20.
        # But we can potentially do better.
        # Actually, let's try a different structural approach:
        # Decompose the 2x4 matrix as [row1; row2] and use linearity
        
        # The multiplication (2x4) * (4x5) = 2x5 result
        # Let A be 2x4, B be 4x5
        # A = [a11 a12 a13 a14; a21 a22 a23 a24]
        # B = [b11 b12 b13 b14 b15; ...; b41 b42 b43 b44 b45]
        
        # Result C = A*B is 2x5
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Using the "border" method or other algebraic techniques
        # For now, let's try to see if rank 35-39 is achievable with better init
        
        # Try structured initialization based on the tensor's sparsity pattern
        for target_rank in range(39, 34, -1):
            if time.time() > deadline - 20:
                break
            
            # Structured init: initialize based on the support of T_target
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Assign random signs/values to non-zero pattern
            for c in range(target_rank):
                # Randomly select which (i,j,k) triple this term contributes to
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                # Add contribution with random weight
                weight = random.choice([1, -1, 0.5, -0.5])
                U[idx_u, c] = weight
                V[idx_v, c] = 1.0
                W[idx_w, c] = 1.0 / weight if weight != 0 else 1.0
            
            # Refine with ALS
            for it in range(100):
                if time.time() > deadline - 20:
                    break
                # Quick ALS step
                for i in range(nm):
                    M = np.zeros((target_rank, mp * pn))
                    for c in range(target_rank):
                        M[c, :] = np.outer(V[:, c], W[:, c]).flatten()
                    rhs = T_target[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                
                if it % 20 == 0 and verify(U, V, W, tol=1e-8):
                    U_hi = round_to_half_integer(U)
                    V_hi = round_to_half_integer(V)
                    W_hi = round_to_half_integer(W)
                    if verify(U_hi, V_hi, W_hi, tol=1e-8):
                        best_result = {"U": U_hi, "V": V_hi, "W": W_hi}
                        best_rank = target_rank
                        break
                
                for j in range(mp):
                    M = np.zeros((target_rank, nm * pn))
                    for c in range(target_rank):
                        M[c, :] = np.outer(U[:, c], W[:, c]).flatten()
                    rhs = T_target[:, j, :].reshape(-1)
                    V[j, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                
                for k in range(pn):
                    M = np.zeros((target_rank, nm * mp))
                    for c in range(target_rank):
                        M[c, :] = np.outer(U[:, c], V[:, c]).flatten()
                    rhs = T_target[:, :, k].reshape(-1)
                    W[k, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            
            if best_rank < target_rank + 1:
                break
    
    # Return the best result
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
