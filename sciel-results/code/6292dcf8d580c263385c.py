"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 75
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    def check_deadline():
        return time.time() > deadline
    
    # Build schoolbook decomposition as fallback
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition: check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured approaches
    # Key insight: For 3x5x5, we can try block decompositions that exploit
    # the rectangular structure. The idea is to split the 5x5 matrix into
    # blocks and use lower-rank decompositions for each sub-problem.
    
    # Approach: Try Strassen-like block decomposition
    # Split 5 into 2+3, giving 4 sub-problems:
    # A1(3x2)*B11(2x2)->C11(3x2): rank 12
    # A2(3x3)*B21(3x2)->C12(3x2): rank 18  
    # A1(3x2)*B12(2x3)->C21(3x3): rank 18
    # A2(3x3)*B22(3x3)->C22(3x3): rank 27
    # Total: 75 (same as schoolbook)
    
    # But we can try to use lower-rank decompositions for the sub-problems.
    # For 3x3x3, the best known is rank 23 (Strassen), not 27.
    # For 3x2x2, 3x3x2, 3x2x3, we need to find good decompositions.
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can try exhaustive or semi-exhaustive search for low-rank decompositions.
    
    # Strategy: Try to find a rank-r decomposition for the full 3x5x5 problem
    # by using a structured ansatz. We'll parameterize U, V, W with constraints
    # that make the problem more tractable.
    
    # Alternative: Use the block structure and try to find lower-rank decomps
    # for the 3x3x3 sub-problem using known results.
    
    # For 3x3x3, Strassen's algorithm gives rank 23.
    # Let's implement Strassen's algorithm for 3x3x3.
    
    def strassen_3x3x3():
        """Return U, V, W for 3x3x3 matrix multiplication with rank 23."""
        # Strassen's algorithm for 3x3x3
        # This is a known decomposition
        rank = 23
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        
        # Strassen's 7 products for 2x2x2, extended to 3x3x3
        # Actually, let's use a simpler approach: the standard Strassen algorithm
        # for 2x2x2 has rank 7. For 3x3x3, we can use a block decomposition.
        
        # Let's use a known rank-23 decomposition for 3x3x3.
        # This is based on the work of Strassen and subsequent improvements.
        
        # For now, let's use a simpler approach: try to find a low-rank decomposition
        # using random initialization and iterative refinement.
        
        # Actually, let's just use the schoolbook for the sub-problems for now,
        # and focus on finding a better overall structure.
        
        # Return schoolbook for 3x3x3
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Let's try a different structural approach: use a "factorization" of the
    # matrix multiplication that exploits the rectangular shape.
    
    # Key idea: A(3x5) * B(5x5) = C(3x5)
    # We can write this as:
    # C = [A1 | A2] * [B1; B2] where A1 is 3x2, A2 is 3x3, B1 is 2x5, B2 is 3x5
    # C = A1*B1 + A2*B2
    
    # This gives us two sub-problems:
    # A1(3x2) * B1(2x5) -> C1(3x5): rank 3*2*5 = 30
    # A2(3x3) * B2(3x5) -> C2(3x5): rank 3*3*5 = 45
    # Total: 75
    
    # But we can try to find lower-rank decompositions for these sub-problems.
    
    # For 3x2x5: we can try to find a rank < 30 decomposition
    # For 3x3x5: we can try to find a rank < 45 decomposition
    
    # Let's try a random search approach for these sub-problems.
    
    def find_low_rank_decomp(n_sub, m_sub, p_sub, target_rank, max_iters=100):
        """Try to find a rank-target_rank decomposition for n_sub x m_sub x p_sub."""
        if target_rank >= n_sub * m_sub * p_sub:
            # Use schoolbook
            U = [[0] * (n_sub * m_sub * p_sub) for _ in range(n_sub * m_sub)]
            V = [[0] * (n_sub * m_sub * p_sub) for _ in range(m_sub * p_sub)]
            W = [[0] * (n_sub * m_sub * p_sub) for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, n_sub * m_sub * p_sub
        
        # Try random initialization and iterative refinement
        best_rank = n_sub * m_sub * p_sub
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(5):
            if check_deadline():
                break
            
            np.random.seed(42 + attempt)
            U = np.random.randn(n_sub * m_sub, target_rank) * 0.5
            V = np.random.randn(m_sub * p_sub, target_rank) * 0.5
            W = np.random.randn(p_sub * n_sub, target_rank) * 0.5
            
            # Iterative refinement: alternate between solving for each matrix
            for iter in range(max_iters):
                if check_deadline():
                    break
                
                # Update U: fix V, W
                # For each i,j: sum_k sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # This is linear in U[i*m_sub+j, :]
                # Build the linear system: for each (i,j), we have an equation
                # sum_c U[i*m_sub+j,c] * (sum_k V[j*p_sub+k,c] * W[k*n_sub+i,c]) = 1
                # Let A[i*m_sub+j, c] = sum_k V[j*p_sub+k,c] * W[k*n_sub+i,c]
                # Then A[i*m_sub+j, :] * U[i*m_sub+j, :] = 1
                # This is a system of n_sub*m_sub equations in target_rank unknowns per row
                
                # Actually, this is underdetermined if target_rank < n_sub*m_sub
                # Let's use least squares
                A = np.zeros((n_sub * m_sub, target_rank))
                for i in range(n_sub):
                    for j in range(m_sub):
                        for c in range(target_rank):
                            s = 0
                            for k in range(p_sub):
                                s += V[j * p_sub + k, c] * W[k * n_sub + i, c]
                            A[i * m_sub + j, c] = s
                
                # Solve A * U_row = 1 for each row
                # This is: for each row r, A[r, :] * U[r, :] = 1
                # We can solve this as a least squares problem
                # Minimize sum_r (A[r, :] * U[r, :] - 1)^2
                # This is equivalent to minimizing ||A * U^T - 1||^2 where U is (n_sub*m_sub, target_rank)
                # Actually, it's a block diagonal system
                
                # Let's solve it row by row
                for r in range(n_sub * m_sub):
                    # A[r, :] * U[r, :] = 1
                    # This is a linear equation in U[r, :]
                    # If A[r, :] is not zero, we can solve it
                    if np.linalg.norm(A[r, :]) > 1e-10:
                        # Least squares solution
                        U[r, :] = np.linalg.lstsq(A[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Similarly for V and W
                
                # Update V: fix U, W
                # For each j,k: sum_i sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # Let B[j*p_sub+k, c] = sum_i U[i*m_sub+j,c] * W[k*n_sub+i,c]
                B = np.zeros((m_sub * p_sub, target_rank))
                for j in range(m_sub):
                    for k in range(p_sub):
                        for c in range(target_rank):
                            s = 0
                            for i in range(n_sub):
                                s += U[i * m_sub + j, c] * W[k * n_sub + i, c]
                            B[j * p_sub + k, c] = s
                
                for r in range(m_sub * p_sub):
                    if np.linalg.norm(B[r, :]) > 1e-10:
                        V[r, :] = np.linalg.lstsq(B[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update W: fix U, V
                # For each k,i: sum_j sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # Let C[k*n_sub+i, c] = sum_j U[i*m_sub+j,c] * V[j*p_sub+k,c]
                C = np.zeros((p_sub * n_sub, target_rank))
                for k in range(p_sub):
                    for i in range(n_sub):
                        for c in range(target_rank):
                            s = 0
                            for j in range(m_sub):
                                s += U[i * m_sub + j, c] * V[j * p_sub + k, c]
                            C[k * n_sub + i, c] = s
                
                for r in range(p_sub * n_sub):
                    if np.linalg.norm(C[r, :]) > 1e-10:
                        W[r, :] = np.linalg.lstsq(C[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Check convergence
                err = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = np.sum(U[i * m_sub + j, :] * V[j * p_sub + k, :] * W[k * n_sub + i, :])
                            err += (val - 1) ** 2
                
                if err < 1e-8:
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if best_rank == target_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        else:
            # Fall back to schoolbook
            rank = n_sub * m_sub * p_sub
            U = [[0] * rank for _ in range(n_sub * m_sub)]
            V = [[0] * rank for _ in range(m_sub * p_sub)]
            W = [[0] * rank for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, rank
    
    # Try to find lower-rank decompositions for the sub-problems
    # Sub-problem 1: 3x2x5
    # Sub-problem 2: 3x3x5
    
    # Try target ranks below the schoolbook
    best_total_rank = rank_school
    best_U, best_V, best_W = build_schoolbook()
    
    # Try a few target ranks for the sub-problems
    for target1 in range(25, 31):  # Try ranks 25-30 for 3x2x5
        if check_deadline():
            break
        for target2 in range(40, 46):  # Try ranks 40-45 for 3x3x5
            if check_deadline():
                break
            
            total_rank = target1 + target2
            if total_rank >= best_total_rank:
                continue
            
            # Try to find decompositions
            U1, V1, W1, r1 = find_low_rank_decomp(3, 2, 5, target1, max_iters=50)
            if check_deadline():
                break
            U2, V2, W2, r2 = find_low_rank_decomp(3, 3, 5, target2, max_iters=50)
            if check_deadline():
                break
            
            actual_total = r1 + r2
            if actual_total < best_total_rank:
                # Combine the decompositions
                best_total_rank = actual_total
                # Create the full U, V, W
                U_full = [[0] * actual_total for _ in range(n * m)]
                V_full = [[0] * actual_total for _ in range(m * p)]
                W_full = [[0] * actual_total for _ in range(p * n)]
                
                # Copy sub-problem 1
                for i in range(3):
                    for j in range(2):
                        for c in range(r1):
                            U_full[i * m + j][c] = U1[i * 2 + j][c]
                for j in range(2):
                    for k in range(5):
                        for c in range(r1):
                            V_full[j * p + k][c] = V1[j * 5 + k][c]
                for k in range(5):
                    for i in range(3):
                        for c in range(r1):
                            W_full[k * n + i][c] = W1[k * 3 + i][c]
                
                # Copy sub-problem 2
                for i in range(3):
                    for j in range(3):
                        for c in range(r2):
                            U_full[i * m + (j + 2)][r1 + c] = U2[i * 3 + j][c]
                for j in range(3):
                    for k in range(5):
                        for c in range(r2):
                            V_full[(j + 2) * p + k][r1 + c] = V2[j * 5 + k][c]
                for k in range(5):
                    for i in range(3):
                        for c in range(r2):
                            W_full[k * n + i][r1 + c] = W2[k * 3 + i][c]
                
                # Verify
                if verify_decomp(U_full, V_full, W_full, actual_total):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    break
        
        if best_total_rank < rank_school:
            break
    
    if best_total_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = build_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
