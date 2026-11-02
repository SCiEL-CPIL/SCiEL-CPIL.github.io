import numpy as np
import scipy.linalg
import random
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 18
    # We try to find a decomposition with rank < 18
    # Known lower bound is max(n*m, m*p, p*n) = 9
    # Let's try ranks from 9 to 17
    
    # Precompute the target tensor T as a 3D array T[i, j, k] for the "diagonal" part
    # The tensor T_{i,j,k}^{i',j',k'} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    # We model this as: for each rank-1 component c, we have:
    # U_c: vector of length n*m (index i*m+j)
    # V_c: vector of length m*p (index j'*p+k)
    # W_c: vector of length p*n (index k'*n+i')
    # Such that sum_c U_c[i*m+j] * V_c[j'*p+k] * W_c[k'*n+i'] = T[i,j,j',k,k',i']
    
    # This is equivalent to: for each (i,j,j',k,k',i'),
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j==j' and i==i' and k==k' else 0
    
    # Let's try a structured approach using GF(2) or small integers
    # First, let's try to find a decomposition with rank 12 (a common target for 2x3x3)
    
    # Actually, let's try a different approach: use the fact that for 2x3x3,
    # there might be a decomposition with rank 12 or lower.
    
    # Let's try ranks from 9 to 17, starting from the most likely to work
    for target_rank in [12, 11, 10, 9, 13, 14, 15, 16, 17]:
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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

def _try_rank(n, m, p, target_rank):
    """Try to find a decomposition with the given target rank."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Number of equations: n*m * m*p * p*n = 6*9*6 = 324
    # Number of unknowns: p_n * target_rank = 6 * target_rank
    # We fix U and V randomly, then solve for W
    
    num_equations = n_m * m_p * p_n
    num_unknowns = p_n * target_rank
    
    # Build the linear system
    # For each equation (i, j, j_prime, k, k_prime, i_prime),
    # the coefficient of W[k_prime*n + i_prime, c] is U[i*m+j, c] * V[j_prime*p+k, c]
    
    # We'll try multiple random U, V configurations
    max_attempts = 500
    
    for attempt in range(max_attempts):
        # Generate random sparse U and V with entries in {-1, 0, 1}
        # Use sparsity to help with conditioning
        density = 0.5
        U = np.zeros((n_m, target_rank))
        V = np.zeros((m_p, target_rank))
        
        for i in range(n_m):
            for c in range(target_rank):
                if random.random() < density:
                    U[i, c] = random.choice([-1, 1])
        
        for i in range(m_p):
            for c in range(target_rank):
                if random.random() < density:
                    V[i, c] = random.choice([-1, 1])
        
        # Build A matrix and b vector
        A = np.zeros((num_equations, num_unknowns))
        b = np.zeros(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m * p):
                    j_prime = jp // p
                    k = jp % p
                    for kp in range(p * n):
                        k_prime = kp // n
                        i_prime = kp % n
                        
                        # RHS
                        if j == j_prime and i == i_prime and k == k_prime:
                            b[eq_idx] = 1
                        
                        # LHS coefficients for W
                        for c in range(target_rank):
                            coeff = U[i * m + j, c] * V[j_prime * p + k, c]
                            if coeff != 0:
                                w_idx = (k_prime * n + i_prime) * target_rank + c
                                A[eq_idx, w_idx] += coeff
                        
                        eq_idx += 1
        
        # Solve the linear system
        try:
            w_solution, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is close to half-integers
        W_flat = w_solution
        is_good = True
        for val in W_flat:
            dist_int = abs(val - round(val))
            dist_half = abs(val - (round(val * 2) / 2.0))
            if dist_int > 1e-6 and dist_half > 1e-6:
                is_good = False
                break
        
        if not is_good:
            continue
        
        # Check residual
        if residuals.size > 0 and residuals[0] > 1e-6:
            continue
        
        # Construct W matrix
        W = W_flat.reshape((p_n, target_rank))
        
        # Verify the decomposition
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
                                if abs(lhs - rhs) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Clean up values
        def clean(val):
            r2 = round(val * 2)
            if abs(val * 2 - r2) < 1e-6:
                return r2 / 2.0
            else:
                return round(val)
        
        U_out = [[clean(val) for val in row] for row in U.tolist()]
        V_out = [[clean(val) for val in row] for row in V.tolist()]
        W_out = [[clean(val) for val in row] for row in W.tolist()]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None
# EVOLVE-BLOCK-END
