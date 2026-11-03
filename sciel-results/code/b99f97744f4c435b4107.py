"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build schoolbook as fallback
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s limit
    
    best_rank = rank_school
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # ============================================================
    # STRATEGY: Block decomposition with Strassen for 2x2x2 blocks
    # ============================================================
    # Split the 3x5 matrix A into blocks: A1 (3x2), A2 (3x3)
    # Split the 5x5 matrix B into blocks: B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # Output C blocks: C1 (3x2) = A1*B11 + A2*B21, C2 (3x3) = A1*B12 + A2*B22
    
    # For each block multiplication, we need a rank decomposition.
    # The 2x2x2 blocks can use Strassen (rank 7).
    # The 3x3x3 blocks use schoolbook (rank 9) or potentially better.
    
    # Block 1: A1(3x2) * B11(2x2) -> C1_part1(3x2)
    # This is a 3x2x2 multiplication. Rank = 3*2*2 = 12 schoolbook.
    # Can we do better? The tensor is T[i,j,k] where i in [0,3), j in [0,2), k in [0,2).
    # This is a 3x2x2 tensor. Known lower bound is max(3*2, 2*2, 2*3) = 6.
    # Actually for small sizes, let's just use schoolbook.
    
    # Block 2: A2(3x3) * B21(3x2) -> C1_part2(3x2)
    # This is a 3x3x2 multiplication. Rank = 3*3*2 = 18 schoolbook.
    
    # Block 3: A1(3x2) * B12(2x3) -> C2_part1(3x3)
    # This is a 3x2x3 multiplication. Rank = 3*2*3 = 18 schoolbook.
    
    # Block 4: A2(3x3) * B22(3x3) -> C2_part2(3x3)
    # This is a 3x3x3 multiplication. Rank = 3*3*3 = 27 schoolbook.
    # Strassen for 3x3x3: We can split 3x3 into 2x2 + 1x1 blocks.
    # Actually, let's try a different approach for the 3x3x3 block.
    
    # Total schoolbook rank: 12 + 18 + 18 + 27 = 75 (same as full schoolbook)
    
    # The key insight: For the 2x2x2 sub-blocks within the larger blocks,
    # we can apply Strassen to reduce rank.
    
    # Let me reconsider the block structure more carefully.
    # A is 3x5, B is 5x5, C is 3x5.
    # Split column dimension of A and row dimension of B at position 2:
    #   A = [A1 | A2] where A1 is 3x2, A2 is 3x3
    #   B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = A1*B11 + A2*B21 (3x2) + A1*B12 + A2*B22 (3x3)
    
    # For the 3x2x2 block (A1*B11):
    #   We can further split A1 into rows. But 3 is small.
    #   Let's just use schoolbook: rank 12.
    
    # For the 3x3x2 block (A2*B21):
    #   Schoolbook: rank 18.
    
    # For the 3x2x3 block (A1*B12):
    #   Schoolbook: rank 18.
    
    # For the 3x3x3 block (A2*B22):
    #   We can apply Strassen-like decomposition.
    #   Split 3x3 into 2x2 + 1x1.
    #   A2 = [A21 A22; A23 A24] where A21 is 2x2, A22 is 2x1, A23 is 1x2, A24 is 1x1
    #   B22 = [B2211 B2212; B2221 B2222] where B2211 is 2x2, B2212 is 2x1, B2221 is 1x2, B2222 is 1x1
    #   C22 = A21*B2211 + A22*B2221 + A23*B2212 + A24*B2222
    #   The 2x2x2 block uses Strassen (rank 7).
    #   The other blocks are smaller.
    
    # Actually, let me try a direct approach: use the known rank-7 Strassen for 2x2x2,
    # and combine with schoolbook for the remaining parts.
    
    # Let me try a simpler approach: just attempt random ALS for various ranks
    # and see if we can find a valid decomposition below 75.
    
    def compute_error(U_curr, V_curr, W_curr):
        """Compute sum of squared errors for all (i,j,k) triples."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                    err += (val - 1)**2
        return err
    
    # Try target ranks from lower to higher
    # Lower bound is max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But realistically, we need at least the rank of the matrix multiplication tensor.
    # For 3x5x5, the tensor rank is unknown but likely > 25.
    
    target_ranks = list(range(48, 76))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 10000 + target_rank * 7 + 42)
            U_curr = np.random.randn(n * m, target_rank) * 0.3
            V_curr = np.random.randn(m * p, target_rank) * 0.3
            W_curr = np.random.randn(p * n, target_rank) * 0.3
            
            converged = False
            max_iterations = 50
            prev_err = float('inf')
            
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # --- Update U ---
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((target_rank, target_rank))
                        for k in range(p):
                            A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            U_new[i*m+j, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            U_new[i*m+j, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                U_curr = U_new
                
                # --- Update V ---
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((target_rank, target_rank))
                        for i in range(n):
                            A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            V_new[j*p+k, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            V_new[j*p+k, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                V_curr = V_new
                
                # --- Update W ---
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((target_rank, target_rank))
                        for j in range(m):
                            A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            W_new[k*n+i, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            W_new[k*n+i, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                W_curr = W_new
                
                # Check convergence
                err = compute_error(U_curr, V_curr, W_curr)
                
                if err < 1e-12:
                    converged = True
                    break
                if abs(prev_err - err) < 1e-14 and err < 1e-8:
                    converged = True
                    break
                prev_err = err
            
            if converged:
                # Polish pass with tighter tolerance
                for polish_iter in range(10):
                    if time.time() > deadline:
                        break
                    
                    U_new = np.zeros_like(U_curr)
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((target_rank, target_rank))
                            for k in range(p):
                                A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                U_new[i*m+j, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                U_new[i*m+j, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    U_curr = U_new
                    
                    V_new = np.zeros_like(V_curr)
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((target_rank, target_rank))
                            for i in range(n):
                                A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                V_new[j*p+k, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                V_new[j*p+k, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    V_curr = V_new
                    
                    W_new = np.zeros_like(W_curr)
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((target_rank, target_rank))
                            for j in range(m):
                                A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                W_new[k*n+i, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                W_new[k*n+i, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    W_curr = W_new
                
                if verify(U_curr, V_curr, W_curr):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    break
    
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
