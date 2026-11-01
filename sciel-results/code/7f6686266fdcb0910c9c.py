"""
Rewritten program for Matrix Multiplication Tensor Decomposition (3,3,6).
Aim: Find a rank-minimal exact tensor decomposition.
Target rank: < 54 (Schoolbook). Likely target: 46 (2 * Rank(3x3x3)) or lower.
"""
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if the indices correspond to the multiplication term
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term is A[i*m+j] * B[j*p+k] * C[k*n+i]^* (dual)
    
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    T_target = np.zeros((dim_u, dim_v, dim_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        r = U.shape[1]
        recon = np.tensordot(U, V, axes=1)  # shape (dim_u, dim_v, r)
        recon = np.tensordot(recon, W, axes=2) # shape (dim_u, dim_v, dim_w)
        return np.allclose(recon, T_target, atol=tol)

    def solve_sne(rank, max_iter=500, num_trials=10):
        from scipy.optimize import minimize
        
        # Objective: sum(T - sum U_t otimes V_t otimes W_t)^2
        # We can speed up by precomputing structures or using efficient einsum
        
        # Flatten parameters: U (9, rank), V (18, rank), W (18, rank)
        # Total params = rank * (9 + 18 + 18) = rank * 45
        
        best_solution = None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            # Deterministic-ish random initialization with different seeds
            np.random.seed(42 + trial * 7)
            
            # Sparse-ish initialization to escape bad local minima?
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            # Normalize to prevent overflow/underflow in early steps
            # Not strictly necessary but helps stability
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective_and_grad(x):
                # Unpack
                U = x[:dim_u*rank].reshape(dim_u, rank)
                V = x[dim_u*rank : dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                W = x[dim_u*rank + dim_v*rank : ].reshape(dim_w, rank)
                
                # Compute T_recon = sum_{t} U[:,t] otimes V[:,t] otimes W[:,t]
                # Using einsum for efficiency
                # Term_t = U[:,t, None, None] * V[:,None,t,None] * W[:,None,None,t]
                # This creates a huge intermediate. 
                # Better: 
                # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
                
                T_recon = np.einsum('at,bt,ct->abc', U, V, W)
                
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                # Gradient
                # d(loss)/dU = 2 * error : (V, W)
                # dU_{a,t} = 2 * sum_{b,c} error_{a,b,c} V_{b,t} W_{c,t}
                dU = 2 * np.einsum('abc,bt,ct->at', error, V, W)
                dV = 2 * np.einsum('abc,at,ct->bt', error, U, W)
                dW = 2 * np.einsum('abc,at,bt->ct', error, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                
                return loss, grad
            
            def objective(x):
                return objective_and_grad(x)[0]
            
            def grad(x):
                return objective_and_grad(x)[1]
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter, 'gtol': 1e-10, 'maxfun': max_iter*10}
                )
            except Exception as e:
                #print(f"Trial {trial} failed: {e}")
                continue
            
            if time.time() > deadline:
                break
                
            U_final = res.x[:dim_u*rank].reshape(dim_u, rank)
            V_final = res.x[dim_u*rank : dim_u*rank + dim_v*rank].reshape(dim_v, rank)
            W_final = res.x[dim_u*rank + dim_v*rank : ].reshape(dim_w, rank)
            
            # Check exactness
            if verify(U_final, V_final, W_final, tol=1e-7):
                # Round to nearest integer/half-integer?
                # The tensor has integer entries. The decomposition coefficients are often integers.
                # Let's check if we can snap to integers.
                U_int = np.round(U_final * 2) / 2.0
                V_int = np.round(V_final * 2) / 2.0
                W_int = np.round(W_final * 2) / 2.0
                
                if verify(U_int, V_int, W_int, tol=1e-6):
                    return U_int, V_int, W_int, rank
                
                # If integer snapping fails, try 0.25 steps?
                U_q = np.round(U_final * 4) / 4.0
                V_q = np.round(V_final * 4) / 4.0
                W_q = np.round(W_final * 4) / 4.0
                if verify(U_q, V_q, W_q, tol=1e-6):
                    return U_q, V_q, W_q, rank

                # Return raw float if verified with high precision
                return U_final, V_final, W_final, rank

        return None

    # Fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb

    # Search for lower ranks
    # Start from 30 up to 53
    # Lavin-Szekely 3x3x3 is 23. 2 * 23 = 46.
    # Try 30 to 53
    
    # Order: try higher ranks first (easier to find), then lower?
    # Or try specific promising values?
    # Let's try 45, 40, 35, 30.
    
    ranks_to_try = [45, 42, 39, 36, 33, 30]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        res = solve_sne(r, max_iter=300, num_trials=5)
        if res is not None:
            U, V, W, found_rank = res
            best_U, best_V, best_W, best_rank = U, V, W, found_rank
            # Don't break immediately, might find even lower?
            # But we should prioritize valid low ranks.
            # The list is descending in difficulty (rank). Lower rank is harder.
            # If we find rank 45, we stop? No, we want min rank.
            # But finding rank 30 is rare.
            # Let's just return the best found so far.
            break # One valid solution is enough to beat 54. We can try to improve further if time permits.
            # Actually, let's just return the first one found to save time.
            # If we find 45, score is 45.
            # If we find 30, score is 30.
            # Since 30 is much better, we should maybe try 30 first?
            # But 30 is very hard to find.
            # Strategy: Try 45, 44, ...
            # If we find one, we can try to search lower only if we have lots of time.
            
    # Convert to lists
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for v in row:
                # Try to keep as int or simple float
                f = Fraction(v).limit_denominator(10)
                if abs(f - v) < 1e-6:
                    if f.denominator == 1:
                        r.append(int(f.numerator))
                    else:
                        r.append(f.numerator / f.denominator)
                else:
                    r.append(float(v))
            res.append(r)
        return res

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

# EVOLVE-BLOCK-END
