import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # Build the schoolbook decomposition
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # The convention is:
    #   U has index (i,j) -> row = i*m + j
    #   V has index (j,k) -> row = j*p + k  
    #   W has index (k,i) -> row = k*n + i
    # and the condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # Strategy: Try to find a decomposition of rank r by exploiting the structure.
    # For 2x3 * 3x5, the schoolbook rank is 30.
    # We can try to reduce this by using linear combinations.
    
    # Key insight: We can group terms by the j index (the inner dimension).
    # For a fixed j, we have n*p = 2*5 = 10 terms.
    # These 10 terms share the same U column pattern (only U[i*m+j, c] can be nonzero)
    # and the same V column pattern (only V[j*p+k, c] can be nonzero).
    
    # Let's think about this differently. The tensor is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    # This is a sum of 30 rank-1 tensors. The question is whether we can do better.
    
    # One approach: use the fact that for fixed j, the sub-tensor involving only
    # terms with that j is a matrix multiplication of size n x 1 times 1 x p,
    # which is just an outer product, and has rank 1 in some sense... 
    # Actually no, for fixed j, we have n*p terms, each being a rank-1 tensor.
    
    # Let me try a direct algebraic approach:
    # Suppose we have r terms. Then for each (i,j,k):
    #   sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 30 equations in the unknowns U, V, W.
    # The unknowns are: nm*r + mp*r + pn*r = (6+15+10)*r = 31*r
    # But we only have 30 constraints, so for r >= 1 we have more unknowns than equations.
    # However, the constraints are nonlinear (products of three variables).
    
    # Let me try a specific construction:
    # Group by j. For j=0: terms (i,0,k) for i in {0,1}, k in {0,1,2,3,4}
    #   U[i*3+0, c] * V[0*5+k, c] * W[k*2+i, c]
    # For j=1: terms (i,1,k) for i in {0,1}, k in {0,1,2,3,4}
    #   U[i*3+1, c] * V[1*5+k, c] * W[k*2+i, c]
    # For j=2: terms (i,2,k) for i in {0,1}, k in {0,1,2,3,4}
    #   U[i*3+2, c] * V[2*5+k, c] * W[k*2+i, c]
    
    # Idea: Can we share some of the rank-1 terms across different j values?
    # If we allow U[i*m+j, c] to be nonzero for multiple j for the same c,
    # then a single term c can contribute to multiple (i,j,k) equations.
    
    # Let me try r = 20, 21, ..., 30 and see if we can find a solution.
    
    # Actually, let me think about this more carefully.
    # The tensor M_{n,m,p} has a known lower bound. For 2x3x5, 
    # the Strassen-style lower bound might be useful.
    
    # Let me try a constructive approach:
    # Use the fact that 3 = 2 + 1, so we can split the 3x5 part into 2x5 and 1x5.
    # M_{2,3,5} = M_{2,2,5} + M_{2,1,5} (in some sense, not exactly)
    # Actually this isn't quite right because the j index couples everything.
    
    # Let me just try to find a decomposition by solving the nonlinear system.
    # I'll use a simple iterative approach:
    
    # For a given rank r, initialize U, V, W randomly and use gradient descent
    # to minimize the residual.
    
    def try_decomposition(r, max_iters=1000, lr=0.01, seed=42):
        """Try to find a decomposition of rank r using gradient descent."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        # Precompute the indices
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((u_idx, v_idx, w_idx))
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Compute residual and gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            residual_sq = 0
            
            for u_idx, v_idx, w_idx in indices:
                # Compute sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
                val = 0
                for c in range(r):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                res = val - 1.0
                residual_sq += res * res
                
                # Gradients
                for c in range(r):
                    U_grad[u_idx, c] += res * V[v_idx, c] * W[w_idx, c]
                    V_grad[v_idx, c] += res * U[u_idx, c] * W[w_idx, c]
                    W_grad[w_idx, c] += res * U[u_idx, c] * V[v_idx, c]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if residual_sq < 1e-12:
                break
        
        # Verify
        if _verify_decomp(n, m, p, r, U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Try ranks from 20 to 30
    best_decomp = None
    best_rank = 31
    
    # First, get the schoolbook decomposition as fallback
    def get_schoolbook():
        U = np.zeros((nm, 30))
        V = np.zeros((mp, 30))
        W = np.zeros((pn, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition
    for r in range(20, 31):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            result = try_decomposition(r, max_iters=500, lr=0.05, seed=seed)
            if result is not None:
                best_decomp = result
                best_rank = r
                break
        
        if best_decomp is not None:
            break
    
    if best_decomp is None:
        best_decomp = get_schoolbook()
    
    return best_decomp

def _verify_decomp(n, m, p, r, U, V, W, tol=1e-8):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0
                for c in range(r):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
