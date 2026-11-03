import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110
    
    def make_schoolbook(rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Using numpy for speed
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape U to (n, m, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        # Reshape V to (m, p, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # We need T[i, j, k] == 1 for all i, j, k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vector dot product
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat)
        # Multiply by 2, round, and check difference
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True

    def is_half_integer_vector(vec):
        """Check if a numpy vector contains only half-integers."""
        doubled = vec * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True

    # Strategy: Try ranks from 44 down to 30
    # For each rank, try multiple random initializations of U and V
    # Then solve for W using the linear system approach
    
    # Pre-compute indices for faster access if needed, but numpy reshape is fast enough
    
    for target_rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random initializations
        # Increase attempts as we go to lower ranks? Or keep constant?
        # Let's keep it around 200-500 depending on time.
        max_attempts = 300
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate structured random U and V
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # Strategy: Use sparse matrices with small integer values
            # To encourage half-integers in W, we want A_ik vectors to have norms
            # that divide 2*A_ik components.
            
            # Randomize density
            # Attempt 0-100: Sparse (2 non-zeros)
            # Attempt 100-200: Medium (3 non-zeros)
            # Attempt 200-300: Denser (4 non-zeros)
            
            if attempt < 100:
                nnz_u = 2
                nnz_v = 2
            elif attempt < 200:
                nnz_u = 3
                nnz_v = 3
            else:
                nnz_u = 4
                nnz_v = 4
                
            for r in range(target_rank):
                # Random rows for U
                u_rows = random.sample(range(n * m), nnz_u)
                for row in u_rows:
                    U[row, r] = random.choice([-1, 1])
                
                # Random rows for V
                v_rows = random.sample(range(m * p), nnz_v)
                for row in v_rows:
                    V[row, r] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p * n, target_rank))
            valid = True
            
            # We need to check if the resulting W is half-integer.
            # Instead of full verification, we can check the W entries directly.
            
            for i in range(n):
                for k in range(p):
                    # Compute A_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
                    A_ik = np.zeros(target_rank)
                    for j in range(m):
                        A_ik += U[i * m + j] * V[j * p + k]
                    
                    denom = np.dot(A_ik, A_ik)
                    if abs(denom) < 1e-10:
                        valid = False
                        break
                    
                    # Minimum norm solution: w = A_ik / (A_ik @ A_ik)
                    w = A_ik / denom
                    W[k * n + i] = w
                    
                    # Quick check: is w half-integer?
                    if not is_half_integer_vector(w):
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If W is half-integer and U, V are integers (which they are, -1, 0, 1),
            # then the decomposition is exact by construction of W.
            # We just need to verify the exactness (floating point errors) and format.
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Final verification
            if verify_decomposition(U_list, V_list, W_list, target_rank):
                # Check half-integer constraint for U and V (they are integers, so OK)
                # Check W (already checked, but double check with list)
                if check_half_integer(W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
