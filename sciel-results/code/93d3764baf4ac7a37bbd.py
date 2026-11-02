"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W decompose the matrix multiplication tensor exactly.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Entries must be integer or half-integer.
    """
    rank = len(U[0])
    
    # Check if entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # val is float
                # Check if 2*val is an integer
                if abs(2 * val - round(2 * val)) > 1e-9:
                    return False

    # Check tensor decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We only need to check non-zero entries of the standard tensor, which are all 1s?
    # No, the tensor T_{mm} has T[i, j, k] = 1 if the indices correspond to a valid multiplication term?
    # Actually, the tensor for matrix multiplication is defined such that 
    # <T, A \otimes B \otimes C^T> = trace(C^T (AB))
    # The components T[i, j, k] are 1 if the indices align in the summation?
    # Let's construct the target tensor T explicitly.
    # T is a 3D array of shape (n, m, p) ? No, the indices are (i, j, k) corresponding to A[i,j], B[j,k], C[k,i].
    # The tensor $T$ satisfies $C = A B \iff \sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = \text{trace}(C^T A B)$.
    # The non-zero entries of $T$ are $T[i,j,k] = 1$ for all $i, j, k$?
    # Let's check: $\sum_{i,j,k} A[i,j] B[j,k] C[k,i] = \text{trace}(C A B)$.
    # Yes, the tensor has 1s at all positions $(i, j, k)$ in the index space $(0..n-1, 0..m-1, 0..p-1)$.
    # Wait, the decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$.
    # So we need $\sum_r U[i, r] V[j, r] W[k, r] = 1$ for all $i \in [0, n), j \in [0, m), k \in [0, p)$.
    
    # Let's flatten indices to match U, V, W shapes.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We can check this by reconstructing the tensor or checking specific slices.
    # Efficient check:
    # For each r, the outer product U[:, r] \otimes V[:, r] \otimes W[:, r] contributes to the tensor.
    # We can accumulate the sum.
    
    T_reconstructed = np.zeros((n, m, p), dtype=np.float64)
    
    for r in range(rank):
        u_vec = np.array(U)[:, r]
        v_vec = np.array(V)[:, r]
        w_vec = np.array(W)[:, r]
        
        # Reshape u_vec to (n, m), v_vec to (m, p), w_vec to (p, n)
        u_mat = u_vec.reshape((n, m))
        v_mat = v_vec.reshape((m, p))
        w_mat = w_vec.reshape((p, n))
        
        # The contribution to T[i, j, k] is u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # This is not a simple outer product of 1D vectors in a way that np.einsum handles directly without care.
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # We can compute this term by term.
        # Since n, m, p are small (3, 5, 7), we can just loop.
        for i in range(n):
            for j in range(m):
                u_val = u_vec[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_vec[j * p + k]
                    if v_val == 0: continue
                    w_val = w_vec[k * n + i]
                    if w_val == 0: continue
                    T_reconstructed[i, j, k] += u_val * v_val * w_val

    # Check if T_reconstructed is all 1s
    # The target tensor for matrix multiplication has 1s at all (i, j, k) positions?
    # Let's verify:
    # Sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = trace(C A B)
    # If T[i,j,k] = 1 for all i,j,k, then Sum A[i,j] B[j,k] C[k,i] = trace(C A B).
    # Yes.
    
    # So we need T_reconstructed[i, j, k] == 1 for all i, j, k.
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9):
        return False
        
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # We will try ranks from best_rank - 1 downwards.
    # Given the time limit, we can't try too many.
    # Let's try a few random restarts for ranks 104, 103, ..., down to maybe 80.
    # Or just try to find *any* improvement.
    
    # Strategy: 
    # 1. Start with rank R.
    # 2. Initialize U, V, W with random small values.
    # 3. Use ALS to minimize error.
    # 4. If error is close to 0, snap to half-integers and verify.
    
    # To be efficient, we can try to use the structure.
    # However, random search is safer for "exact" integer constraints if we snap carefully.
    
    # Let's try ranks in decreasing order.
    # We will limit the number of restarts per rank based on time.
    
    for R in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        max_restarts = 5
        if R < 90:
            max_restarts = 20
        if R < 70:
            max_restarts = 50
        if R < 50:
            max_restarts = 100

        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers/half-integers
            # Scaling is important. Start with small values.
            scale = 1.0
            U = np.random.randint(-2, 3, (n * m, R)) * 0.5
            V = np.random.randint(-2, 3, (m * p, R)) * 0.5
            W = np.random.randint(-2, 3, (p * n, R)) * 0.5
            
            # ALS optimization
            # We want to minimize || T_target - sum(U_r V_r W_r) ||^2
            # This is not a simple linear problem for all variables simultaneously.
            # But fixing two, the third is linear.
            
            # Reshape for easier manipulation
            # U_mat: (n, m, R)
            # V_mat: (m, p, R)
            # W_mat: (p, n, R)
            
            U_arr = U.reshape((n, m, R))
            V_arr = V.reshape((m, p, R))
            W_arr = W.reshape((p, n, R))
            
            # Target tensor T is all 1s of shape (n, m, p)
            T_target = np.ones((n, m, p))
            
            # ALS steps
            for step in range(50): # Fixed number of steps per restart
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each (i, j), we want U[i, j, :] to satisfy:
                # sum_r U[i, j, r] * (V[j, :, r] * W[:, i, r]) = T_target[i, j, :] (which is 1 for all k? No)
                # Wait, the equation is:
                # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all k.
                # This is a linear system for U[i, j, :] for each fixed i, j.
                # Let M_U[i, j] be the matrix where M_U[i, j][k, r] = V[j, k, r] * W[k, i, r].
                # Then M_U[i, j] @ U[i, j, :] = ones(p).
                
                # We can solve this least squares problem.
                for i in range(n):
                    for j in range(m):
                        # Construct M_U for this (i, j)
                        # Shape (p, R)
                        # M_U[k, r] = V_arr[j, k, r] * W_arr[k, i, r]
                        M_U = V_arr[j, :, :] * W_arr[:, i, :] # Broadcasting: (p, 1) * (p, R) -> (p, R)
                        # Solve M_U @ x = ones(p)
                        try:
                            # Add regularization to avoid singularity
                            sol, residuals, rank, sv = np.linalg.lstsq(M_U, np.ones(p), rcond=None)
                            U_arr[i, j, :] = sol
                        except:
                            pass

                # Update V
                # For each (j, k), V[j, k, :]
                # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i.
                # M_V[i, r] = U_arr[i, j, r] * W_arr[k, i, r]
                for j in range(m):
                    for k in range(p):
                        M_V = U_arr[:, j, :] * W_arr[k, :, :] # (n, R) * (n, R) -> (n, R)
                        try:
                            sol, residuals, rank, sv = np.linalg.lstsq(M_V, np.ones(n), rcond=None)
                            V_arr[j, k, :] = sol
                        except:
                            pass

                # Update W
                # For each (k, i), W[k, i, :]
                # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all j.
                # M_W[j, r] = U_arr[i, j, r] * V_arr[j, k, r]
                for k in range(p):
                    for i in range(n):
                        M_W = U_arr[i, :, :] * V_arr[:, k, :] # (m, R) * (m, R) -> (m, R)
                        try:
                            sol, residuals, rank, sv = np.linalg.lstsq(M_W, np.ones(m), rcond=None)
                            W_arr[k, i, :] = sol
                        except:
                            pass

            # After ALS, check error
            U_flat = U_arr.reshape((n * m, R))
            V_flat = V_arr.reshape((m * p, R))
            W_flat = W_arr.reshape((p * n, R))
            
            # Snap to half-integers
            U_snap = np.round(U_flat * 2) / 2.0
            V_snap = np.round(V_flat * 2) / 2.0
            W_snap = np.round(W_flat * 2) / 2.0
            
            # Verify exactness
            if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                best_rank = R
                # If we found a valid one, we can continue to try for lower rank,
                # or break if we think this is good enough.
                # Let's continue to try for lower rank.
                continue

    return best_result
# EVOLVE-BLOCK-END
