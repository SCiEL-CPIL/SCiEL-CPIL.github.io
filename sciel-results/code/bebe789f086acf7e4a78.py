import numpy as np
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We want to find a decomposition of rank r < 18
    # Try ranks 12, 13, 14, 15
    # If all fail, fallback to 18
    
    # Target tensor T[i, j, k, i', j', k'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # Indices:
    # U: (n*m, r) -> 6 x r
    # V: (m*p, r) -> 9 x r
    # W: (p*n, r) -> 6 x r
    
    # Equation: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = T[i,j,j',k,k',i']
    
    # We will iterate through candidate ranks
    for rank in [12, 13, 14, 15]:
        # Number of trials
        # Rank 12 is hard, Rank 15 is easy
        trials = 5000 if rank <= 13 else 2000
        
        for trial in range(trials):
            # Generate random sparse U and V
            # Use -1, 0, 1 with probability 0.5 for non-zero
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < 0.5:
                        U[i, c] = random.choice([-1, 1])
                        
            for i in range(m * p):
                for c in range(rank):
                    if random.random() < 0.5:
                        V[i, c] = random.choice([-1, 1])
            
            # Build linear system for W
            # A * w = b
            # w is flattened W (p*n*rank)
            # Equations: for each (i, j, j', k, k', i')
            # Coefficient of W[k'*n+i', c] is U[i*m+j, c] * V[j'*p+k, c]
            
            num_eq = n * m * m * p * p * n # 2*3*3*3*3*2 = 324
            num_unk = p * n * rank # 6 * rank
            
            A = np.zeros((num_eq, num_unk))
            b = np.zeros(num_eq)
            
            # Fill A and b
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                for i_prime in range(n):
                                    # Target value
                                    if j == j_prime and i == i_prime and k == k_prime:
                                        b[eq_idx] = 1.0
                                    else:
                                        b[eq_idx] = 0.0
                                    
                                    # Coefficients
                                    # W index for this equation is k_prime*n + i_prime
                                    # We need to add U[i*m+j, c] * V[j_prime*p+k, c] to A[eq_idx, (k_prime*n+i_prime)*rank + c]
                                    
                                    u_row = U[i * m + j, :]
                                    v_row = V[j_prime * p + k, :]
                                    
                                    # product is element-wise
                                    prod = u_row * v_row
                                    
                                    # Add to A
                                    # The column block for this (k_prime, i_prime) is from (k_prime*n+i_prime)*rank to (k_prime*n+i_prime)*rank + rank
                                    col_start = (k_prime * n + i_prime) * rank
                                    A[eq_idx, col_start:col_start+rank] += prod
                                    
                                    eq_idx += 1
            
            # Solve
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check residuals
            if residuals.size > 0 and residuals[0] > 1e-6:
                continue
                
            # Check if w_sol is close to half-integers
            w_flat = w_sol
            is_half_int = True
            for val in w_flat:
                # Distance to nearest half-integer
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_half) > 1e-4:
                    is_half_int = False
                    break
                    
            if not is_half_int:
                continue
                
            # Construct W
            W = w_flat.reshape((p * n, rank))
            
            # Clean W to exact half-integers
            W_clean = np.round(W * 2) / 2.0
            
            # Verify exact decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                for i_prime in range(n):
                                    lhs = 0.0
                                    for c in range(rank):
                                        lhs += U[i * m + j, c] * V[j_prime * p + k, c] * W_clean[k_prime * n + i_prime, c]
                                    rhs = 1.0 if (j == j_prime and i == i_prime and k == k_prime) else 0.0
                                    if abs(lhs - rhs) > 1e-4:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            # If valid, return
            # Convert to lists of numbers (int or float)
            def to_list(mat):
                return [[float(x) for x in row] for row in mat]
                
            return {
                "U": to_list(U),
                "V": to_list(V),
                "W": to_list(W_clean)
            }
            
    # Fallback to schoolbook rank 18
    rank = n * m * p
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
# EVOLVE-BLOCK-END
