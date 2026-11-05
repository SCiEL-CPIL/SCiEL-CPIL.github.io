import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T = 1 iff j=j', k=k', i=i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank 53 first
    for target_rank in [53, 54]:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Use the structured initialization from the 54-rank solution
        # and try to optimize down to target_rank
        
        # Initialize with the Strassen-based solution
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            if col >= target_rank:
                return False
            
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            if col >= target_rank:
                return False
            
            # M1: (A10+A11)B00
            U[(i_base+1) * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] -= 1
            col += 1
            if col >= target_rank:
                return False
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            if col >= target_rank:
                return False
            
            # M3: A11(B10-B00)
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            if col >= target_rank:
                return False
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            if col >= target_rank:
                return False
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base+1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            if col >= target_rank:
                return False
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base+1)][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] -= 1
            V[(j_base+1) * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            return True
        
        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            for di in range(2):
                for dk in range(2):
                    if col >= target_rank:
                        return False
                    U[(i_base+di) * m + j_base][col] += 1
                    V[j_base * p + (k_base+dk)][col] += 1
                    W[(k_base+dk) * n + (i_base+di)][col] += 1
                    col += 1
            return True
        
        # Fill with Strassen blocks
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # Now we've used 42 columns. Need 53-42=11 more for the last part.
        # Naive would use 12. Can we do 11?
        # The remaining is A[:,4] × B[4,:] which is 2×1 × 1×6 → 2×6
        # This needs at least 12 rank-1 terms (outer product of 2 and 6 vectors)
        # So 11 is impossible for this block alone.
        
        # But maybe we can share terms across blocks!
        # Let's try to optimize the last few terms.
        
        # For now, fill with naive
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        # Check if we have exactly target_rank columns
        if col != target_rank:
            # Adjust: if we have too many, truncate; if too few, this won't work
            if col > target_rank:
                U = U[:, :target_rank]
                V = V[:, :target_rank]
                W = W[:, :target_rank]
            else:
                continue
        
        # Verify the decomposition
        # Compute T_reconstructed
        T_rec = np.zeros((n*m, m*p, p*n))
        for t in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        
        if np.allclose(T_rec, T, atol=1e-10):
            # Found exact decomposition!
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try optimization: adjust coefficients to make it exact
        # Flatten the problem
        # T_rec[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # This is nonlinear. Use iterative refinement.
        
        # Try a simple approach: fix W, solve for U and V
        # Or use alternating least squares
        
        for iteration in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Update U: for each t, U[:,t] should satisfy
            # sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            # Fix V and W, solve for U
            # This is: for each (b,c), U[:,t]*V[b,t]*W[c,t] summed over t = T[:,b,c]
            # Let M_bc = outer(V[:,t]*W[c,t] for t) ... this is complex
            
            # Simpler: use gradient descent on the loss
            # Loss = sum_{a,b,c} (T_rec[a,b,c] - T[a,b,c])^2
            
            # Compute gradient
            # dL/dU[a,t] = 2 * sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,t] * W[c,t]
            # dL/dV[b,t] = 2 * sum_{a,c} (T_rec[a,b,c] - T[a,b,c]) * U[a,t] * W[c,t]
            # dL/dW[c,t] = 2 * sum_{a,b} (T_rec[a,b,c] - T[a,b,c]) * U[a,t] * V[b,t]
            
            # This is expensive. Let's use a different approach.
            # Instead, try to find if the current solution can be made exact by adjusting
            # only the last few columns.
            
            break  # Skip optimization for now
        
        # Try random perturbation of the last few columns
        for attempt in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Perturb the last 3 columns
            for t in range(target_rank-3, target_rank):
                U_try[:, t] += np.random.randn(n*m) * 0.1
                V_try[:, t] += np.random.randn(m*p) * 0.1
                W_try[:, t] += np.random.randn(p*n) * 0.1
            
            T_rec = np.zeros((n*m, m*p, p*n))
            for t in range(target_rank):
                T_rec += np.einsum('a,b,c->abc', U_try[:,t], V_try[:,t], W_try[:,t])
            
            if np.allclose(T_rec, T, atol=1e-10):
                return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    # Fallback: return the rank-54 solution
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
