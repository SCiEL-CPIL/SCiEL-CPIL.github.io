import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strategy:
    # 1. Decompose the 2x4 multiplication into a lower rank structure (Strassen-like or direct).
    #    Standard 2x4 is rank 8. Strassen for 2x2 is 7, but we have 2x4.
    #    We can view 2x4 as two 2x2 blocks. Rank 2x2 is 7 (Strassen) or 4 (Schoolbook).
    #    If we use Strassen for the 2x2 blocks, we might save operations.
    #    However, the tensor is T_{i,j,k}.
    #    
    #    Let's try a different approach:
    #    The tensor T is in C^{2x4} x C^{4x7} x C^{7x2}.
    #    We can fix a basis for the first two modes and solve for the third, or use a mixed approach.
    #    
    #    Improvement over current:
    #    The current program uses random U, V and solves for W.
    #    This is good but random.
    #    
    #    New Strategy:
    #    Use a structured initialization for U and V based on known fast matrix multiplication algorithms for small sizes, 
    #    then refine using local search or linear algebra.
    #    
    #    For 2x4 * 4x7:
    #    We can split the 4 columns of the first matrix into two 2-column blocks.
    #    Let A = [A1, A2] where A1, A2 are 2x2.
    #    Let B = [B1, B2] where B1, B2 are 4x? No, B is 4x7.
    #    C = A1 * B_left + A2 * B_right? No, the split must be consistent.
    #    
    #    Actually, let's stick to the "Linear Algebra Rank Reduction" idea.
    #    We know the rank is at most 56.
    #    We suspect it might be lower.
    #    
    #    Let's try to construct a decomposition with rank 50, 49, etc.
    #    But instead of pure random, we use a "greedy" or "structured" random.
    #    
    #    Specifically, we can use the fact that the tensor is linear in each mode.
    #    If we fix U and V, W is determined (if the system is consistent).
    #    The condition for consistency is that the tensor lies in the span of the rank-1 tensors formed by U and V columns.
    #    
    #    Let's try a specific structured search:
    #    1. Try ranks from 55 down to 40.
    #    2. For each rank, generate U and V using a mix of random and structured patterns.
    #    3. Solve for W using least squares.
    #    4. Check if W is close to half-integers.
    #    5. If yes, verify exactly.
    #    
    #    To improve the current program, we will:
    #    a) Increase the number of attempts for lower ranks.
    #    b) Use a smarter initialization for U and V (e.g., based on Kronecker products or block structures).
    #    c) Add a local search step: if we find a valid decomposition with rank r, try to reduce it to r-1 by null-space projection.
    
    # Helper to verify
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to snap to half-integers
    def snap_to_half(W):
        W_half = W * 2
        W_rounded = np.round(W_half)
        error = np.max(np.abs(W_half - W_rounded))
        if error < 1e-6:
            return W_rounded / 2.0, True
        return W, False

    # Search for lower rank
    # We will try ranks from 55 down to 45.
    for r in range(55, 44, -1):
        if time.time() > deadline - 10:
            break
        
        attempts = 200 if r > 50 else 400
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U and V
            # Using entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            
            # Construct linear system for W
            num_eqs = n * m * p
            num_unknowns = p * n * r
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row = k * n + i
                        for c in range(r):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                A_mat[eq_idx, w_row * r + c] = coeff
                        eq_idx += 1
            
            # Solve A_mat * W_flat = b_vec
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(p * n, r)
            
            # Snap to half-integers
            W_candidate, is_half = snap_to_half(W)
            
            if is_half:
                # Verify exactly
                if verify(U, V, W_candidate):
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_candidate.tolist()
                    
                    # Check if we can reduce rank further by removing zero columns or dependent columns
                    # For now, return this valid decomposition
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    return best_result
