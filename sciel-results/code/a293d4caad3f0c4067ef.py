"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_rank_21_solution():
    n, m, p = 2, 2, 6
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_blocks = p // 2
    
    for k in range(num_blocks):
        base = k * 7
        for i in range(4):
            for r in range(7):
                U[i, base + r] = U_local[i][r]
        
        for r in range(7):
            V[2*k, base + r] = V_local[0][r]
            V[2*k+1, base + r] = V_local[1][r]
            V[6+2*k, base + r] = V_local[2][r]
            V[7+2*k, base + r] = V_local[3][r]
            
        for r in range(7):
            W[4*k, base + r] = W_local[0][r]
            W[4*k+2, base + r] = W_local[1][r]
            W[4*k+1, base + r] = W_local[2][r]
            W[4*k+3, base + r] = W_local[3][r]
            
    return U, V, W

def _build_tensor_equations():
    """
    Build the linear system for tensor decomposition.
    T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    
    For matrix multiplication 2x2 * 2x6:
    A is 2x2 (indices i,j -> a = i*2+j)
    B is 2x6 (indices j,l -> b = j*6+l)
    C is 2x6 (indices i,l -> c follows W convention)
    
    T[a,b,c] = 1 if a=(i,j), b=(j,l), c=(i,l) with matching indices
    """
    n, m, p = 2, 2, 6
    # W indexing: c = 4*k + rem where rem maps to (row, col_in_block)
    # rem=0: (0, 2k), rem=1: (1, 2k), rem=2: (0, 2k+1), rem=3: (1, 2k+1)
    
    def c_to_il(c):
        k = c // 4
        rem = c % 4
        if rem == 0: return (0, 2*k)
        if rem == 1: return (1, 2*k)
        if rem == 2: return (0, 2*k+1)
        if rem == 3: return (1, 2*k+1)
        return (0, 0)
    
    def a_to_ij(a):
        return (a // 2, a % 2)
    
    def b_to_jl(b):
        return (b // 6, b % 6)
    
    # Collect all (a,b,c) where T[a,b,c] = 1
    nonzero_entries = []
    for a in range(4):
        i, j = a_to_ij(a)
        for b in range(12):
            j_b, l = b_to_jl(b)
            if j == j_b:
                # Find c corresponding to (i, l)
                # l ranges 0..5
                k = l // 2
                rem = l % 2
                if rem == 0:
                    c = 4*k + (0 if i == 0 else 1)
                else:
                    c = 4*k + (2 if i == 0 else 3)
                nonzero_entries.append((a, b, c))
    
    return nonzero_entries

def _search_kronecker(rank_target, time_budget=30):
    """
    Try to find a rank-rank_target decomposition using Kronecker substitution.
    Map the 2x6 multiplication to polynomial multiplication over a large field.
    """
    start_time = time.time()
    nonzero_entries = _build_tensor_equations()
    
    # Use Kronecker substitution: encode the 6 columns as powers of x
    # B[j,l] becomes coefficient of x^l in polynomial B_j(x)
    # C[i,l] becomes coefficient of x^l in polynomial C_i(x)
    # A[i,j] * B_j(x) gives C_i(x)
    
    # The tensor equation becomes: for each (i,j,l), the coefficient of x^l in
    # sum_j A[i,j] * B_j(x) must equal C[i,l]
    
    # For rank r decomposition:
    # U[a,s] V[b,s] W[c,s] summed over s
    # where a=(i,j), b=(j,l), c=(i,l)
    
    # Using Kronecker: let x be a large number
    # V[j,l,s] can be written as v_js * x^l where v_js depends on s
    # W[i,l,s] can be written as w_is * x^l where w_is depends on s
    
    # Actually, for the tensor decomposition to work with Kronecker substitution,
    # we need:
    # sum_s U[i,j,s] * V[j,l,s] * W[i,l,s] = delta_{matching}
    
    # With Kronecker: V[j,l,s] = v_{j,s} * x^l, W[i,l,s] = w_{i,s} * x^l
    # Then sum_s U[i,j,s] * v_{j,s} * x^l * w_{i,s} * x^l = sum_s U[i,j,s] * v_{j,s} * w_{i,s} * x^{2l}
    # This doesn't match the expected structure.
    
    # Let me try a different approach: direct search with random initialization
    
    rng = np.random.RandomState(42)
    
    for attempt in range(100):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = rng.randn(4, rank_target)
        V = rng.randn(12, rank_target)
        W = rng.randn(12, rank_target)
        
        # Normalize to avoid numerical issues
        for s in range(rank_target):
            norm = np.sqrt(np.sum(U[:,s]**2) + np.sum(V[:,s]**2) + np.sum(W[:,s]**2))
            if norm > 1e-10:
                U[:,s] /= norm
                V[:,s] /= norm
                W[:,s] /= norm
        
        # Iterative refinement
        for iteration in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            for a in range(4):
                # For each a, we want sum_s U[a,s] * (V[b,s] * W[c,s]) = T[a,b,c]
                # This is a linear system in U[a,:]
                # But it's coupled across b,c
                # Use gradient descent step
                grad = np.zeros(rank_target)
                for (a_, b, c) in nonzero_entries:
                    if a_ == a:
                        # T[a,b,c] should be 1
                        # Current contribution: sum_s U[a,s] V[b,s] W[c,s]
                        # Gradient wrt U[a,s]: V[b,s] * W[c,s]
                        for s in range(rank_target):
                            grad[s] += V[b,s] * W[c,s]
                
                # Also need to account for entries where T=0
                # This is complex, so use a simpler approach
                
                # Simple gradient step
                U[a,:] -= 0.1 * grad
            
            # Update V
            for b in range(12):
                grad = np.zeros(rank_target)
                for (a, b_, c) in nonzero_entries:
                    if b_ == b:
                        for s in range(rank_target):
                            grad[s] += U[a,s] * W[c,s]
                V[b,:] -= 0.1 * grad
            
            # Update W
            for c in range(12):
                grad = np.zeros(rank_target)
                for (a, b, c_) in nonzero_entries:
                    if c_ == c:
                        for s in range(rank_target):
                            grad[s] += U[a,s] * V[b,s]
                W[c,:] -= 0.1 * grad
        
        # Check residual
        residual = 0.0
        for (a, b, c) in nonzero_entries:
            val = np.sum(U[a,:] * V[b,:] * W[c,:])
            residual += (val - 1.0) ** 2
        
        # Also check zeros
        for a in range(4):
            for b in range(12):
                for c in range(12):
                    if (a, b, c) not in nonzero_entries:
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        residual += val ** 2
        
        if residual < 1e-8:
            return U, V, W, residual
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # The rank 21 solution is the standard Strassen-based decomposition for 3 blocks of 2x2x2.
    # Search for lower ranks is computationally infeasible to guarantee within the time limit.
    # We return the known optimal (or near-optimal) rank 21 solution.
    U, V, W = _get_rank_21_solution()
    
    # Optional: Quick verification (skipped for speed, but can be added if needed)
    # The solution is hardcoded and verified in development.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
