"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 130  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_structured_search(target_rank, max_attempts=200):
        """
        Uses the decoupled linear system structure.
        For fixed U, V, the system for W decouples into n*p independent sub-systems.
        Each sub-system is M @ w = 1, where M is m x rank.
        We solve for w and check if it's close to integer.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate random U, V with entries in {0, 1} or {-1, 0, 1}
            # Using {0, 1} first as it corresponds to F2
            val_set = [0, 1] if attempt % 2 == 0 else [-1, 0, 1]
            
            U = [[random.choice(val_set) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice(val_set) for _ in range(target_rank)] for _ in range(mp)]
            
            # Convert to numpy for easier manipulation
            U_np = np.array(U).reshape(n, m, target_rank)
            V_np = np.array(V).reshape(m, p, target_rank)
            
            W_solution = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for each (i, k)
            for i in range(n):
                for k in range(p):
                    # Construct M: m x target_rank
                    # M[j, c] = U[i, j, c] * V[j, k, c]
                    M = U_np[i, :, :] * V_np[:, k, :]
                    
                    # Solve M @ w = 1
                    # If M has rank < m, there might be no solution or infinite.
                    # lstsq gives min-norm solution.
                    try:
                        w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if w is close to integers
                    w_int = np.round(w)
                    if np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                        # Store W[k, i, c]
                        for c in range(target_rank):
                            W_solution[k * n + i][c] = int(w_int[c])
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W_solution}
        
        return None

    def try_als_search(target_rank, max_restarts=50, max_iters=100):
        """
        ALS-based search for low-rank decomposition.
        Alternates between solving for U, V, W given the others,
        then projects to integer/half-integer and verifies.
        """
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random sparse matrices
            U = np.random.randint(0, 2, size=(n, m, target_rank)).astype(float)
            V = np.random.randint(0, 2, size=(m, p, target_rank)).astype(float)
            W = np.random.randint(0, 2, size=(p, n, target_rank)).astype(float)
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U: for each (i,j), U[i,j,:] should satisfy
                # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                # This is a linear system for U[i,j,:]
                for i in range(n):
                    for j in range(m):
                        # A[c, k] = V[j,k,c] * W[k,i,c]
                        A = np.zeros((target_rank, p))
                        for k in range(p):
                            A[:, k] = V[j, k, :] * W[k, i, :]
                        # Solve A @ u = ones(p)
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i, j, :] = u
                        except:
                            pass
                
                # Solve for V: for each (j,k), V[j,k,:] should satisfy
                # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i, j, :] * W[k, i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j, k, :] = v
                        except:
                            pass
                
                # Solve for W: for each (k,i), W[k,i,:] should satisfy
                # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for j in range(m):
                            A[:, j] = U[i, j, :] * V[j, k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k, i, :] = w
                        except:
                            pass
            
            # After ALS, project to integers and verify
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            # Verify
            T = np.einsum('ijc,jkc,kic->ijk', U_int, V_int, W_int)
            if np.all(T == 1):
                U_list = U_int.reshape(nm, target_rank).tolist()
                V_list = V_int.reshape(mp, target_rank).tolist()
                W_list = W_int.reshape(pn, target_rank).tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 15:
            break
        
        # First try structured search
        result = try_structured_search(target_rank, max_attempts=50)
        if result:
            best = result
            best_rank = target_rank
            continue
        
        # Then try ALS
        result = try_als_search(target_rank, max_restarts=20, max_iters=50)
        if result:
            best = result
            best_rank = target_rank
            continue
    
    return best
# EVOLVE-BLOCK-END
