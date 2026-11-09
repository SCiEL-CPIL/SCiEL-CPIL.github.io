import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def _precompute_target(n, m, p):
    """Precompute the target tensor T[i, j, jp, k, ip, kp] = delta(i,ip)*delta(j,jp)*delta(k,kp)"""
    # T has shape (n, m, m, p, n, p)
    # Indices: i, j, jp, k, ip, kp
    # Value is 1 if i==ip, j==jp, k==kp, else 0
    
    # Using broadcasting
    # i: (n, 1, 1, 1, 1, 1)
    # jp: (1, 1, m, 1, 1, 1) -> wait, jp is index 2
    # k: (1, 1, 1, p, 1, 1) -> k is index 3
    # ip: (1, 1, 1, 1, n, 1) -> ip is index 4
    # j: (1, m, 1, 1, 1, 1) -> j is index 1
    # kp: (1, 1, 1, 1, 1, p) -> kp is index 5
    
    # Let's construct the condition directly
    # We want T[i, j, jp, k, ip, kp]
    # Condition: i==ip AND j==jp AND k==kp
    
    # Meshgrid approach is safer for clarity
    I, J, JP, K, IP, KP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(n), range(p),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    return target

def _verify_exact(U, V, W, n, m, p, target):
    """Verify if U, V, W exactly decompose the tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # T[i, j, jp, k, ip, kp] = sum_r U[i, j, r] * V[jp, k, r] * W[k, ip, r]
    # Note: W index is k*n + ip, so W3[k, ip, r]
    # einsum: U[i, j, r], V[jp, k, r], W[k, ip, r] -> T[i, j, jp, k, ip, kp]?
    # Wait, the target indices are (i, j, jp, k, ip, kp).
    # In the decomposition:
    # U depends on i, j
    # V depends on jp, k
    # W depends on k, ip (and implicitly kp? No, W is (p*n, R), so W[k, ip, r]. 
    # Wait, the standard tensor for matmul C=AB is T[i, j, k, i', j', k']?
    # Let's re-read the task contract carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]"
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = ?
    
    # In the schoolbook solution:
    # For a specific (i, j, k), we have a column r where:
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # And for all other (i', j', k'), the product is 0?
    # Let's check the product for arbitrary (i, j, j', k, i', k')?
    # No, the indices in the product are coupled.
    # The product is sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r].
    # Wait, the V index in the product usually matches the inner dimension.
    # If we are computing C[i', k] = sum_j A[i', j] B[j, k],
    # The tensor T maps (A, B) -> C.
    # T[i, j, j', k, i', k'] is not quite right.
    
    # Let's look at the seed program's verification:
    # T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    # U3: (n, m, R) -> indices i, j, r
    # V3: (m, p, R) -> indices j, k, r  (Wait, seed uses 'jkr' for V3? 
    #   V3 is reshaped from V (m*p, R) to (m, p, R). So V3[j, k, r].
    #   The einsum string 'ijr,jkr,pir' implies:
    #   U[i, j, r]
    #   V[j, k, r]  <-- Here the second index of U (j) is summed with the first index of V (j).
    #   W[p, i, r]  <-- Here the second index of V (k) is summed with the first index of W (p)?
    #   Wait, W3 is (p, n, R). So W3[k, i, r]?
    #   The einsum output is 'ijjkpi'.
    #   Indices: i, j, j, k, p, i.
    #   This looks like it's summing over r, but keeping i, j (from U), j (from V), k (from V), p (from W? No, W is p*n), i (from W).
    #   
    # Let's re-examine the seed's `W` reshape: `W3 = W.reshape(p, n, R)`.
    # So W3[k, i, r] corresponds to W[k*n + i, r].
    # The einsum 'ijr,jkr,pir' is invalid because 'p' is not in U or V.
    # Ah, the seed code in the prompt has:
    # `T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)`
    # This implies W3 has indices p, i, r.
    # So W3 is (p, n, R).
    # And the output indices are i, j, j, k, p, i.
    # This means the tensor T has indices (i, j, j', k, p, i').
    # Wait, the last index is 'i' again? That would mean i == i'.
    # And the first 'j' and second 'j' (j') are separate?
    # The target in seed:
    # `target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)`
    # Meshgrid: I (n), J (m), JP (m), K (p), KP (p), IP (n).
    # So target indices are (i, j, jp, k, kp, ip).
    # Condition: i==ip, j==jp, k==kp.
    #
    # So the tensor T[i, j, jp, k, kp, ip] should be 1 if i==ip, j==jp, k==kp.
    #
    # Let's check the einsum again.
    # U3: (n, m, R) -> i, j, r
    # V3: (m, p, R) -> jp, k, r  (Using jp as the first index of V)
    # W3: (p, n, R) -> kp, ip, r (Using kp as the first index of W, ip as second)
    #
    # Product: sum_r U[i, j, r] * V[jp, k, r] * W[kp, ip, r]
    # Output indices: i, j, jp, k, kp, ip.
    # This matches the target indices (i, j, jp, k, kp, ip).
    #
    # So the correct einsum is:
    # `np.einsum('ijr,jkr,kir->ijjkkir', U3, V3, W3)`? No.
    # U: i, j, r
    # V: jp, k, r
    # W: kp, ip, r
    # We want output: i, j, jp, k, kp, ip
    # So: `np.einsum('ijr,jkr,kir->ijjkkir', U3, V3, W3)` is wrong because 'k' is used in V and W?
    # V has indices (jp, k). W has indices (kp, ip).
    # So V's second index is k. W's first index is kp.
    # They are different indices in the target: k and kp.
    # So the einsum should be:
    # `np.einsum('ijr,ab r,cd r->ijabcd', U3, V3, W3)` where:
    # a=jp, b=k
    # c=kp, d=ip
    # Output: i, j, jp, k, kp, ip.
    #
    # Let's write it out:
    # `np.einsum('ijr,jkr,kir->ijjkkir', ...)` No, 'k' is repeated.
    # Use distinct letters:
    # U: i, j, r
    # V: a, b, r  (a=jp, b=k)
    # W: c, d, r  (c=kp, d=ip)
    # Output: i, j, a, b, c, d
    # `np.einsum('ijr,abr,cdr->ijabcd', U3, V3, W3)`
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    T = np.einsum('ijr,abr,cdr->ijabcd', U3, V3, W3, optimize=True)
    
    # Target indices: i, j, jp, k, kp, ip
    # T indices:      i, j, a,  b, c,  d
    # So a=jp, b=k, c=kp, d=ip.
    # Target condition: i==d (ip), j==a (jp), b==c (k==kp).
    
    # Let's verify with the target array generated by meshgrid.
    # I, J, JP, K, KP, IP
    # target[i, j, jp, k, kp, ip] = 1 if i==ip and j==jp and k==kp.
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _solve_rank(n, m, p, R, target, time_limit=10.0):
    """Try to find a decomposition of rank R."""
    start_time = time.time()
    num_vars = R * (n * m + m * p + p * n)
    
    def residual(params):
        U = params[:R * n * m].reshape(n * m, R)
        V = params[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
        W = params[R * (n * m + m * p):].reshape(p * n, R)
        
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        T = np.einsum('ijr,abr,cdr->ijabcd', U3, V3, W3, optimize=True)
        
        return (T - target).flatten()
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        params = np.random.randint(-2, 3, size=num_vars)
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=2000)
            
            # Check if residual is close to zero
            if np.linalg.norm(result.fun) < 1e-6:
                U = result.x[:R * n * m].reshape(n * m, R)
                V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
                W = result.x[R * (n * m + m * p):].reshape(p * n, R)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exact solution
                if _verify_exact(U_rounded, V_rounded, W_rounded, n, m, p, target):
                    return U_rounded, V_rounded, W_rounded
                    
                # If rounding failed, try to polish the original solution with higher precision
                # Or just accept if it's extremely close? No, must be exact.
                # Try a few more random restarts if this one failed rounding.
                
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    target = _precompute_target(n, m, p)
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 24
    # We try ranks from 47 down to 40
    for target_rank in range(47, 39, -1):
        if time.time() > DEADLINE - 15:
            break
        
        result = _solve_rank(n, m, p, target_rank, target, time_limit=12.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break # Found a solution, return it
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
