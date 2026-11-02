"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Flatten T for faster ALS operations
    # T_flat[i] is a vector of length mp*pn
    T_flat = T.reshape(nm, mp * pn)
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_list(mat):
        """Convert numpy array to list of lists with clean values."""
        out = []
        for row in mat:
            r = []
            for x in row:
                if abs(x) < 1e-12:
                    r.append(0)
                elif abs(x - 1) < 1e-12:
                    r.append(1)
                elif abs(x + 1) < 1e-12:
                    r.append(-1)
                elif abs(x - 0.5) < 1e-12:
                    r.append(0.5)
                elif abs(x + 0.5) < 1e-12:
                    r.append(-0.5)
                else:
                    r.append(x)
            out.append(r)
        return out

    def als_step(U, V, W, T_flat, nm, mp, pn, max_iter=50):
        """
        Alternating Least Squares to minimize || T - sum U_c V_c W_c ||_F^2
        U: (nm, r), V: (mp, r), W: (pn, r)
        """
        r = U.shape[1]
        # Precompute Kronecker products for V, W and U, W and U, V
        # M_VW: (mp*pn, r) where col c is V[:,c] x W[:,c]
        M_VW = np.kron(V, W) # shape (mp*pn, r)
        M_UW = np.kron(U, W) # shape (nm*pn, r)
        M_UV = np.kron(U, V) # shape (nm*mp, r)
        
        T_VW = T_flat.reshape(nm, mp * pn) # (nm, mp*pn)
        T_UW = T.reshape(nm, pn, mp).transpose(0, 2, 1).reshape(nm, pn * mp) # Need correct flattening for U update?
        # Let's be careful with indices.
        # T[i,j,k]. 
        # Update U: fix V,W.
        # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed i, we want U[i,:] to approximate T[i,:,:] using basis {V[:,c] W[:,c]}.
        # Basis matrix M_VW has rows indexed by (j,k) and cols by c.
        # T[i, j, k] flattened is T_flat[i].
        # So U[i,:] = T_flat[i] @ M_VW^T (M_VW M_VW^T)^-1
        # We can compute the pseudo-inverse of M_VW once per iteration? 
        # M_VW changes when V or W changes.
        
        # Let's update U, then V, then W.
        
        # 1. Update U
        # M_VW is (mp*pn, r)
        # Compute (M_VW M_VW^T)^-1 M_VW^T ? No, that's for projecting T onto row space of M_VW.
        # We want U[i,:] such that U[i,:] M_VW^T = T_flat[i] (as a row vector)
        # Let A = M_VW. We want u A = t.
        # u = t A^T (A A^T)^-1
        try:
            A_A_T_inv = np.linalg.inv(M_VW @ M_VW.T)
            # U_new[i, :] = T_flat[i] @ M_VW.T @ A_A_T_inv
            U_new = T_flat @ (M_VW.T @ A_A_T_inv)
            U = U_new
        except np.linalg.LinAlgError:
            pass
        
        # 2. Update V
        # Fix U, W.
        # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed j, we want V[j,:] to approximate T[:, j, :] using basis {U[:,c] W[:,c]}.
        # Basis matrix M_UW has rows indexed by (i,k) and cols by c.
        # M_UW = kron(U, W) is (nm*pn, r).
        # T[:, j, :] flattened is T[:, j, :].reshape(-1).
        # Note: T is (nm, mp, pn). T[:, j, :] is (nm, pn). Flatten to (nm*pn).
        # Order: index (i, k) -> i*pn + k.
        # kron(U, W) row index: i*pn + k? 
        # np.kron(U, W) where U is (nm, r), W is (pn, r).
        # Result is (nm*pn, r). Row (i*pn + k) is U[i, :] * W[k, :]? 
        # Yes, kron(A, B) with A (m, n), B (p, q) -> (mp, nq).
        # Row i*p + k, col j*q + l is A[i,j] B[k,l].
        # Here A=U (nm, r), B=W (pn, r).
        # Row i*pn + k, col c is U[i,c] W[k,c]. Correct.
        
        M_UW = np.kron(U, W)
        # T_slice_j: for each j, get T[:, j, :] and flatten.
        # T shape (nm, mp, pn).
        # T[:, j, :] shape (nm, pn).
        # We need a matrix of shape (mp, nm*pn) where row j is T[:, j, :].flatten().
        T_for_V = np.zeros((mp, nm * pn))
        for j in range(mp):
            T_for_V[j, :] = T[:, j, :].flatten()
        
        try:
            A_A_T_inv = np.linalg.inv(M_UW @ M_UW.T)
            V_new = T_for_V @ (M_UW.T @ A_A_T_inv)
            V = V_new
        except np.linalg.LinAlgError:
            pass
            
        # 3. Update W
        # Fix U, V.
        # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed k, we want W[k,:] to approximate T[:, :, k] using basis {U[:,c] V[:,c]}.
        # Basis matrix M_UV has rows indexed by (i,j) and cols by c.
        # M_UV = kron(U, V) is (nm*mp, r).
        # Row i*mp + j, col c is U[i,c] V[j,c]. Correct.
        
        M_UV = np.kron(U, V)
        # T_slice_k: for each k, get T[:, :, k] and flatten.
        # T shape (nm, mp, pn).
        # T[:, :, k] shape (nm, mp).
        # Flatten to (nm*mp).
        T_for_W = np.zeros((pn, nm * mp))
        for k in range(pn):
            T_for_W[k, :] = T[:, :, k].flatten()
            
        try:
            A_A_T_inv = np.linalg.inv(M_UV @ M_UV.T)
            W_new = T_for_W @ (M_UV.T @ A_A_T_inv)
            W = W_new
        except np.linalg.LinAlgError:
            pass
            
        return U, V, W

    def try_rank(target_rank):
        # Try to find a decomposition of target_rank
        # Strategy 1: Merging initialization
        for attempt in range(100):
            if time.time() > deadline:
                return False
            
            # Random partition of 0..41 into target_rank groups
            indices = list(range(rank_school))
            random.shuffle(indices)
            # Assign indices to groups
            groups = [[] for _ in range(target_rank)]
            for idx in indices:
                # Simple round-robin or random assignment
                g = random.randint(0, target_rank - 1)
                groups[g].append(idx)
            
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            for c in range(target_rank):
                if groups[c]:
                    U[:, c] = np.sum(U_school[:, groups[c]], axis=1)
                    V[:, c] = np.sum(V_school[:, groups[c]], axis=1)
                    W[:, c] = np.sum(W_school[:, groups[c]], axis=1)
                else:
                    # If a group is empty, add small noise to avoid singularity
                    U[:, c] = np.random.randn(nm) * 0.01
                    V[:, c] = np.random.randn(mp) * 0.01
                    W[:, c] = np.random.randn(pn) * 0.01
            
            # Run ALS
            for _ in range(50):
                if time.time() > deadline:
                    return False
                U, V, W = als_step(U, V, W, T_flat, nm, mp, pn)
                
                # Check for convergence/exactness
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-10:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r):
                        nonlocal best_rank, best_U, best_V, best_W
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        return True
            
        # Strategy 2: Pure random initialization
        for attempt in range(50):
            if time.time() > deadline:
                return False
            
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for _ in range(50):
                if time.time() > deadline:
                    return False
                U, V, W = als_step(U, V, W, T_flat, nm, mp, pn)
                
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-10:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r):
                        nonlocal best_rank, best_U, best_V, best_W
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        return True
                        
        return False

    # Try ranks from 41 down to 20
    for r in range(41, 19, -1):
        if time.time() > deadline:
            break
        if try_rank(r):
            break

    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
