import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 26
    
    # Define the target tensor T
    # T[i, j, j', k, k', i'] = 1 if j==j' and k==k' and i==i' else 0
    # We only need to check the non-zero parts for efficiency in residual calculation?
    # No, least_squares needs all residuals.
    
    def residuals(params):
        # params is a flat vector of length 3 * 9 * target_rank
        size = n * m
        U = params[0:size*target_rank].reshape(size, target_rank)
        V = params[size*target_rank:2*size*target_rank].reshape(size, target_rank)
        W = params[2*size*target_rank:3*size*target_rank].reshape(size, target_rank)
        
        # Compute the tensor product sum
        # Sum_c U[i, j, c] * V[j', k, c] * W[k', i', c]
        # We need to compare with T
        
        # Efficient computation:
        # For each c, compute outer products?
        # Or use einsum.
        # Residuals shape: (n, m, m, p, p, n) -> 3,3,3,3,3,3 = 729
        
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # Let's reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        
        # Sum over r: U[n,m,r] * V[m,p,r] * W[p,n,r]
        # This is a tensor contraction.
        # Result shape: (n, m, m, p, p, n) ?
        # Indices: i, j, j', k, k', i'
        # U[i, j, c] * V[j', k, c] * W[k', i', c]
        # Wait, V index is (j', k). In V3, row is j', col is k?
        # V shape is (m*p, r). Index is j'*p + k.
        # So V3[j', k, c].
        # W shape is (p*n, r). Index is k'*n + i'.
        # So W3[k', i', c].
        
        # Using einsum:
        # 'ijc,jkc,kic -> ijjkkii'? No.
        # U: i, j, c
        # V: j', k, c  (note: j' is the first index of V)
        # W: k', i', c (note: k' is first, i' is second)
        
        # We want result[i, j, j', k, k', i']
        # sum_c U[i, j, c] * V[j', k, c] * W[k', i', c]
        
        # einsum('ijc,jkc,kic->ijjjkkii')? No, indices must be unique in output.
        # Let's name indices:
        # U: a, b, c
        # V: d, e, c
        # W: f, g, c
        # Output: a, b, d, e, f, g
        
        # U3 is (n, m, r) -> a, b, c
        # V3 is (m, p, r) -> d, e, c  (d is j', e is k)
        # W3 is (p, n, r) -> f, g, c  (f is k', g is i')
        
        # So: np.einsum('abc,dec,fec->abdefg', U3, V3, W3) ?
        # Wait, V3 indices are (d, e, c). W3 indices are (f, g, c).
        # The c index is summed.
        
        result = np.einsum('abc,dec,fec->abdefg', U3, V3, W3)
        
        # Now construct the target tensor T
        # T[a, b, d, e, f, g] = 1 if b==d and e==f and a==g else 0
        # Note: 
        # a=i, b=j
        # d=j', e=k
        # f=k', g=i'
        # Condition: j==j' (b==d), k==k' (e==f), i==i' (a==g)
        
        # We can create T efficiently.
        # T = np.zeros((n, m, m, p, p, n))
        # for i in range(n):
        #   for j in range(m):
        #     for k in range(p):
        #       T[i, j, j, k, k, i] = 1
        
        # But creating T every time is slow.
        # We can precompute T.
        
        return (result - T_target).flatten()

    # Precompute T_target
    T_target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, k, i] = 1.0

    # Optimization
    best_rank = n * m * p
    best_params = None
    
    # Try rank 26
    r = 26
    size = n * m
    num_vars = 3 * size * r
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize
        # Random initialization
        params = np.random.randn(num_vars) * 0.1
        
        # Solve
        try:
            res = least_squares(residuals, params, method='lm', max_nfev=1000)
            if res.cost < 1e-8:
                # Found a solution
                # Verify and extract
                U = res.x[0:size*r].reshape(size, r)
                V = res.x[size*r:2*size*r].reshape(size, r)
                W = res.x[2*size*r:3*size*r].reshape(size, r)
                
                # Convert to list of lists as required
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
            
    # If no solution found, return seed
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
