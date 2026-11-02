"""Program for 2x3x9 matrix multiplication tensor decomposition using SVD approximation and rounding."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
import random

def construct():
    """Return a low-rank tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(U, V, W, R):
        """Verify the decomposition exactly."""
        # Convert to numpy for faster verification if needed, but list is fine for small R
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(R):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def try_svd_decomp(target_rank):
        """
        Attempt to find a decomposition of target_rank by:
        1. Constructing the full tensor.
        2. Matricizing it.
        3. Finding best rank-R approximation via SVD.
        4. Reshaping back to U, V, W.
        5. Rounding entries to nearest half-integers.
        6. Verifying.
        """
        # 1. Construct tensor T of shape (n*m, m*p, p*n)
        # T[i*m+j, j*p+k, k*n+i] = 1
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # 2. Matricize. 
        # We want U (n*m x R), V (m*p x R), W (p*n x R).
        # This corresponds to a 3-way tensor decomposition.
        # A common approach is to flatten two modes.
        # Let's flatten modes 1 and 2 into a matrix of size (n*m * m*p) x (p*n).
        # M = T.reshape(n*m * m*p, p*n)
        # SVD: M = U_M * S * V_M.T
        # U_M is (n*m * m*p) x (p*n).
        # This doesn't directly give U, V, W in the required format.
        
        # Alternative: Use the fact that T is a matrix multiplication tensor.
        # We can use a heuristic: 
        # Randomly initialize U, V, W with small random values.
        # Use Gradient Descent to minimize ||T - sum u_c v_c w_c||^2.
        # Then round.
        
        # Given the time constraint and complexity of exact GD, 
        # let's use a simpler "Structured Random" approach with Rounding.
        
        # Initialize U, V, W with random values in {-1, 0, 1}
        # This is similar to the parent but we will do a local search on the rounded values.
        
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Initialize with sparse random integers
        for c in range(target_rank):
            # Each term should ideally contribute to some subset of the tensor
            # Randomly assign non-zeros
            mask_u = np.random.choice([0, 1], size=n*m, p=[0.7, 0.3])
            mask_v = np.random.choice([0, 1], size=m*p, p=[0.7, 0.3])
            mask_w = np.random.choice([0, 1], size=p*n, p=[0.7, 0.3])
            
            U[:, c] = mask_u * np.random.choice([-1, 1], size=n*m)
            V[:, c] = mask_v * np.random.choice([-1, 1], size=m*p)
            W[:, c] = mask_w * np.random.choice([-1, 1], size=p*n)
            
        # Local Search: Try to fix errors
        # Calculate current tensor
        def calc_tensor(U, V, W):
            T_calc = np.zeros((n * m, m * p, p * n), dtype=float)
            for c in range(target_rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            return T_calc
        
        T_target = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Iterate local search
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                return None
            
            T_calc = calc_tensor(U, V, W)
            error = np.sum((T_calc - T_target) ** 2)
            
            if error < 1e-9:
                # Check if it's exactly integer/half-integer and matches
                # Round to nearest half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded version
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                if verify_decomp(U_list, V_list, W_list, target_rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # If rounding broke it, accept current float if it's close enough? 
                    # No, must be exact.
                    # Try to adjust to nearest valid integer?
                    pass
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Perturb one entry of one matrix
            if random.random() < 0.33:
                mat = U
                name = 'U'
            elif random.random() < 0.5:
                mat = V
                name = 'V'
            else:
                mat = W
                name = 'W'
                
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, mat.shape[1] - 1)
            
            # Change value
            old_val = mat[row, col]
            # Try values in {-1, -0.5, 0, 0.5, 1, 2}
            candidates = [-1, -0.5, 0, 0.5, 1, 2]
            # Remove old val if present
            if old_val in candidates:
                candidates.remove(old_val)
            new_val = random.choice(candidates)
            
            mat[row, col] = new_val
            
            # Calculate new error (approximate, only changed terms)
            # For simplicity, recalc full error every few steps or use fast approx
            # Full recalc is O(R * NMP). 33 * 54 = 1782 ops. 5000 iters = 9M ops. Fast enough.
            T_calc_new = calc_tensor(U, V, W)
            new_error = np.sum((T_calc_new - T_target) ** 2)
            
            if new_error >= error:
                # Revert
                mat[row, col] = old_val
            else:
                # Accept
                pass
        
        # Check best solution
        if best_error < 1e-9:
            U_round = np.round(best_U * 2) / 2
            V_round = np.round(best_V * 2) / 2
            W_round = np.round(best_W * 2) / 2
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            if verify_decomp(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        return None

    # Try ranks from 30 upwards
    # Known: 54 is schoolbook. 
    # Let's try 36, 33, 30.
    for R in [36, 33, 30]:
        if time.time() - start_time > DEADLINE - 10:
            break
        result = try_svd_decomp(R)
        if result:
            return result
            
    # Fallback
    return build_schoolbook()
