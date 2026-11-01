"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    
    # Target tensor: T[i,j,k] = 1 for all (i,j,k)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook fallback (rank 18)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def tensor_residual(U, V, W):
        """Compute residual using vectorized operations."""
        # Residual[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # Build using broadcasting
        rank = U.shape[1]
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Reshape for computation
        # U_idx: (n, m, rank), V_idx: (m, p, rank), W_idx: (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # = einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        residual = result - 1.0
        return np.sum(residual ** 2)
    
    def is_half_integer(arr, tol=1e-9):
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-6):
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol
    
    def als_solve(U, V, W, target_rank, max_iters=500, tol=1e-10):
        """ALS solver for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_res = tensor_residual(U, V, W)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve linear system
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ (V[j,:] * W[:,i]).T
            # A[k,r] = V[j*p+k,r] * W[k*n+i,r], shape (p, rank)
            # U_row @ A.T = T[i,j,:]
            # U_row = T[i,j,:] @ pinv(A.T)
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U_row @ A.T = ones(p)
                    # U_row = ones(p) @ pinv(A.T)
                    try:
                        A_T = A.T  # (rank, p)
                        U[i * m + j, :] = np.linalg.lstsq(A_T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ (U[:,j,:] * W[k*n,:]).T
            # B[i,r] = U[i*m+j,r] * W[k*n+i,r], shape (n, rank)
            # V_row @ B.T = ones(n)
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        B_T = B.T  # (rank, n)
                        V[j * p + k, :] = np.linalg.lstsq(B_T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ (U[i*m,:,:] * V[:,k,:]).T
            # C[j,r] = U[i*m+j,r] * V[j*p+k,r], shape (m, rank)
            # W_row @ C.T = ones(m)
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        C_T = C.T  # (rank, m)
                        W[k * n + i, :] = np.linalg.lstsq(C_T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            curr_res = tensor_residual(U, V, W)
            if abs(prev_res - curr_res) < tol and curr_res < 1e-6:
                break
            prev_res = curr_res
        
        return U, V, W
    
    def polish(U, V, W, target_rank, max_iters=100):
        """High-precision polish using more iterations."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        A_T = A.T
                        U[i * m + j, :] = np.linalg.lstsq(A_T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        B_T = B.T
                        V[j * p + k, :] = np.linalg.lstsq(B_T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        C_T = C.T
                        W[k * n + i, :] = np.linalg.lstsq(C_T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            res = tensor_residual(U, V, W)
            if res < 1e-12:
                break
        
        return U, V, W
    
    def try_rank(target_rank, max_restarts=20):
        """Try to find a valid decomposition for given target_rank."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return False
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Random Gaussian
                np.random.seed(42 + target_rank * 1000 + restart)
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
            elif restart % 4 == 1:
                # Near-schoolbook perturbation: start with first target_rank columns of schoolbook
                if target_rank <= rank_sb:
                    U = U_sb[:, :target_rank].copy()
                    V = V_sb[:, :target_rank].copy()
                    W = W_sb[:, :target_rank].copy()
                    # Add small random perturbation to escape local minima
                    np.random.seed(1000 + target_rank * 100 + restart)
                    U += 0.1 * np.random.randn(nm, target_rank)
                    V += 0.1 * np.random.randn(mp, target_rank)
                    W += 0.1 * np.random.randn(pn, target_rank)
                else:
                    np.random.seed(42 + target_rank * 1000 + restart)
                    U = np.random.randn(nm, target_rank)
                    V = np.random.randn(mp, target_rank)
                    W = np.random.randn(pn, target_rank)
            elif restart % 4 == 2:
                # Structured: exploit symmetry between i=0 and i=1
                np.random.seed(2000 + target_rank * 100 + restart)
                U = np.random.randint(-2, 3, size=(nm, target_rank)) / 2.0
                V = np.random.randint(-2, 3, size=(mp, target_rank)) / 2.0
                W = np.random.randint(-2, 3, size=(pn, target_rank)) / 2.0
            else:
                # Sparse random
                np.random.seed(3000 + target_rank * 100 + restart)
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    for _ in range(np.random.randint(2, 5)):
                        U[np.random.randint(nm), r] = np.random.choice([-1, 1]) / 2
                    for _ in range(np.random.randint(2, 5)):
                        V[np.random.randint(mp), r] = np.random.choice([-1, 1]) / 2
                    for _ in range(np.random.randint(2, 5)):
                        W[np.random.randint(pn), r] = np.random.choice([-1, 1]) / 2
            
            # Run ALS
            U_sol, V_sol, W_sol = als_solve(U, V, W, target_rank, max_iters=300)
            
            # Check if close to solution
            res = tensor_residual(U_sol, V_sol, W_sol)
            
            if res < 1e-4:
                # Polish
                U_pol, V_pol, W_pol = polish(U_sol, V_sol, W_sol, target_rank, max_iters=50)
                res_pol = tensor_residual(U_pol, V_pol, W_pol)
                
                if res_pol < 1e-8:
                    # Check half-integer
                    if verify_solution(U_pol, V_pol, W_pol, 1e-6):
                        best = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
                        best_rank = target_rank
                        return True
                    else:
                        # Try rounding
                        U_r = round_to_half(U_pol)
                        V_r = round_to_half(V_pol)
                        W_r = round_to_half(W_pol)
                        if verify_solution(U_r, V_r, W_r, 1e-6):
                            best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                            best_rank = target_rank
                            return True
                        else:
                            # Try snapping to nearest half-integer and re-polishing
                            U_snap = np.round(U_pol * 2) / 2
                            V_snap = np.round(V_pol * 2) / 2
                            W_snap = np.round(W_pol * 2) / 2
                            # One more polish from snapped
                            U_p2, V_p2, W_p2 = polish(U_snap, V_snap, W_snap, target_rank, max_iters=30)
                            if verify_solution(U_p2, V_p2, W_p2, 1e-6):
                                best = {"U": U_p2.tolist(), "V": V_p2.tolist(), "W": W_p2.tolist()}
                                best_rank = target_rank
                                return True
        
        return False
    
    # Try ranks from 14 upward (known minimum for 2x3x3 is 14)
    for target_rank in range(14, rank_sb + 1):
        if time.time() > deadline - 15:
            break
        found = try_rank(target_rank, max_restarts=15)
        if found:
            # Found a solution at this rank, can't do better
            break
    
    return best
# EVOLVE-BLOCK-END
