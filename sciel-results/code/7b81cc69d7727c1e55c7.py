import numpy as np
import scipy.linalg

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 12
    
    # Tensor T indices:
    # U: (n*m, rank) -> 6 x r
    # V: (m*p, rank) -> 9 x r
    # W: (p*n, rank) -> 6 x r
    
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j, j',k, k',i']
    # T is 1 if j==j' and i==i' and k==k', else 0.
    
    # Build the linear system matrix A and vector b
    # Unknowns are W flattened: W[k*n+i, c] -> index (k*n+i)*r + c
    num_unknowns = (p * n) * target_rank
    num_equations = (n * m) * (m * p) * (p * n) # 6 * 9 * 6 = 324
    
    A = np.zeros((num_equations, num_unknowns))
    b = np.zeros(num_equations)
    
    # Fill b (the target tensor)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m * p): # j', k
                j_prime = jp // p
                k = jp % p
                for kp in range(p * n): # k', i'
                    k_prime = kp // n
                    i_prime = kp % n
                    # Check if this equation is non-zero
                    if j == j_prime and i == i_prime and k == k_prime:
                        b[eq_idx] = 1
                    eq_idx += 1
    
    # Random search for U and V
    # We will try a few random configurations
    for _ in range(1000):
        # Generate random U and V with small integers
        # Sparse to help with conditioning?
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        
        # Build A for this U, V
        A_current = np.zeros((num_equations, num_unknowns))
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m * p):
                    j_prime = jp // p
                    k = jp % p
                    for kp in range(p * n):
                        k_prime = kp // n
                        i_prime = kp % n
                        
                        # Coefficient for W[k_prime*n + i_prime, c] is U[i*m+j, c] * V[j_prime*p+k, c]
                        # We sum over c
                        for c in range(target_rank):
                            coeff = U[i * m + j, c] * V[j_prime * p + k, c]
                            if coeff != 0:
                                # Index in W vector
                                w_idx = (k_prime * n + i_prime) * target_rank + c
                                A_current[eq_idx, w_idx] += coeff
                        
                        eq_idx += 1
        
        # Solve A_current * w = b
        # Use least squares
        try:
            w_solution, residuals, rank, s = np.linalg.lstsq(A_current, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check if solution is close to integer or half-integer
        # And if it actually satisfies the equations (residuals small)
        if residuals.size > 0 and residuals[0] > 1e-6:
            continue
            
        # Check if w_solution entries are close to rational numbers with small denom
        is_good = True
        W_flat = w_solution
        for val in W_flat:
            # Check distance to nearest integer and half-integer
            dist_int = abs(val - round(val))
            dist_half = abs(val - (round(val * 2) / 2.0))
            if dist_int > 1e-4 and dist_half > 1e-4:
                is_good = False
                break
        if not is_good:
            continue
            
        # Construct W matrix
        W = W_flat.reshape((p * n, target_rank))
        
        # Verify the decomposition exactly (using integers if possible)
        # Convert to fractions or just check with floats
        valid = True
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                lhs = 0
                                for c in range(target_rank):
                                    lhs += U[i * m + j, c] * V[j_prime * p + k, c] * W[k_prime * n + i_prime, c]
                                rhs = 1 if (j == j_prime and i == i_prime and k == k_prime) else 0
                                if abs(lhs - rhs) > 1e-4:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid:
            continue
            
        # Return the solution
        # Ensure entries are integers or half-integers
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        
        # Clean up values to be exact integers or half-integers
        def clean(val):
            r2 = round(val * 2)
            if abs(val * 2 - r2) < 1e-4:
                return r2 / 2.0
            else:
                return round(val)
                
        U_out = [[clean(val) for val in row] for row in U_out]
        V_out = [[clean(val) for val in row] for row in V_out]
        W_out = [[clean(val) for val in row] for row in W_out]
        
        return {"U": U_out, "V": V_out, "W": W_out}
        
    # Fallback to schoolbook if search fails
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
