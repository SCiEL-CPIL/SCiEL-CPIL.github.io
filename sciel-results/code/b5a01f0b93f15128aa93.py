import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE = 150  # seconds, keep 10s safety margin

def _verify(U, V, W, n, m, p, rank, seed=12345):
    """Verify the decomposition by checking a few random integer matrices."""
    np.random.seed(seed)
    A = np.random.randint(0, 10, size=(n, m))
    B = np.random.randint(0, 10, size=(m, p))
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # For each term c, compute u_c = sum_{i,j} U[i*m+j, c] * A[i,j]
    # v_c = sum_{j,k} V[j*p+k, c] * B[j,k]
    # C_decomp = sum_c u_c * v_c * W_c  where W_c is the outer product matrix
    
    # Reshape A to (n*m,) and B to (m*p,)
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # u_c = U[:, c] @ A_flat
    u_vals = U_arr @ A_flat  # shape (rank,)
    v_vals = V_arr @ B_flat  # shape (rank,)
    
    # C_decomp[i, k] = sum_c u_c * v_c * W[k*n + i, c]
    # W is (p*n, rank), so W[k*n+i, c]
    W_reshaped = W_arr.reshape(p, n, rank)  # (p, n, rank)
    
    # For each c: contribution to C[i,k] is u_c * v_c * W[k, i, c]
    # C_decomp[i, k] = sum_c u_c * v_c * W[k, i, c]
    # = sum_k? No, for each (i,k): sum_c u_c * v_c * W[k, i, c]
    
    # Reshape to (p, n, rank) and compute
    uv = u_vals * v_vals  # (rank,)
    # C_decomp[i, k] = sum_c uv[c] * W[k, i, c]
    # = (W_reshaped @ uv)[k, i]  if we do W_reshaped.T @ uv? 
    # W_reshaped[k, i, c], sum over c: this is np.einsum('kic,c->ki', W_reshaped, uv)
    C_decomp = np.einsum('kic,c->ki', W_reshaped, uv)
    C_decomp = C_decomp.T  # now (n, p)
    
    return np.allclose(C_expected, C_decomp, atol=1e-9)

def _try_rank(n, m, p, target_rank, time_budget, rng_seed):
    """Try to find a decomposition of target_rank. Returns (U, V, W) or None."""
    if target_rank < n + m + p - 2:
        return None  # Strassen lower bound
    
    deadline = time.time() + time_budget
    rng = np.random.RandomState(rng_seed)
    
    # We'll use a simple approach: try to find sparse integer decompositions
    # by iteratively improving from a random initialization.
    
    # Strategy: start with random sparse matrices with small integer entries,
    # then use coordinate descent / local search to satisfy the tensor equation.
    
    # The tensor equation: for all (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # For all other combinations (where the triple doesn't correspond to a valid matmul term):
    # The decomposition only needs to satisfy the matmul constraint.
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 if the indices
    # correspond to a valid matmul term, and the decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,j,k} (matmul delta)
    
    # Wait, let me re-read the contract. The tensor is for multiplying (n x m) by (m x p).
    # The standard tensor has entries T[i, j, k] where i in [0,n), j in [0,m), k in [0,p).
    # The matmul tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for the "multiplication" pattern.
    
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # the tensor entry. For matmul, the tensor entry for indices (i,j,k) is... 
    
    # Let me think about this more carefully. The tensor for matrix multiplication M_{n,m,p}
    # is a 3-tensor of shape (n*m, m*p, p*n) or equivalently indexed by (i,j), (j',k), (k',i').
    # The constraint is: sum_c U[(i,j), c] * V[(j',k), c] * W[(k',i'), c] 
    # should equal delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # So for each c, we pick a rank-1 term, and the sum over c must give the identity.
    
    # Let's set up the problem: we need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i, j, j', k, k', i':
    #   sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # This is a system of polynomial equations in the entries of U, V, W.
    
    # Simple approach: use a greedy/iterative method.
    # 1. Start with random initialization
    # 2. For each term c, try to assign values to minimize the residual
    
    # Let's try a different approach: use the fact that for small ranks, we can
    # use a structured initialization.
    
    # For rank = n + m + p - 2 (Strassen-like), this is generally hard.
    # Let's try rank = n*m*p first (schoolbook) and see if we can reduce.
    
    # Actually, let's try a randomized local search approach:
    # - Initialize U, V, W with random sparse patterns
    # - Iteratively update one entry at a time to reduce the residual
    
    # But this might be too slow. Let's try a more direct approach.
    
    # For the 3x5x6 case, known bounds:
    # Lower bound: max(n, m, p) * min(n, m, p) = 3 * 5 = 15 (trivial)
    # Strassen-like: n + m + p - 2 = 12 (but this is for square matrices)
    # Upper bound: n*m*p = 90
    
    # Let's try to find a decomposition with rank < 90.
    
    # Approach: use a sparse random initialization and then optimize.
    
    # Let's define the residual function:
    # R = sum over all (i,j,j',k,k',i') of (sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c] - delta)^2
    
    # This is a high-dimensional nonlinear optimization problem.
    
    # For now, let's just try a few random sparse initializations and see if any work.
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Random sparse initialization
        density = max(0.1, 3.0 / target_rank)  # expected non-zeros per row
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Choose random positions for non-zeros
            n_u = max(1, int(rng.binomial(n * m, density)))
            n_v = max(1, int(rng.binomial(m * p, density)))
            n_w = max(1, int(rng.binomial(p * n, density)))
            
            u_idx = rng.choice(n * m, size=n_u, replace=False)
            v_idx = rng.choice(m * p, size=n_v, replace=False)
            w_idx = rng.choice(p * n, size=n_w, replace=False)
            
            U[u_idx, c] = rng.choice([-1, 1], size=n_u)
            V[v_idx, c] = rng.choice([-1, 1], size=n_v)
            W[w_idx, c] = rng.choice([-1, 1], size=n_w)
        
        # Check if this works
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce the rank
    # Try ranks from a lower bound upward
    lower_bound = max(n, m, p) * min(n, m, p)
    # Also try Strassen-like
    strassen_bound = n + m + p - 2
    
    # Try a few candidate ranks
    candidate_ranks = sorted(set([lower_bound, strassen_bound, 
                                   int(rank * 0.75), int(rank * 0.5),
                                   int(rank * 0.33), rank]))
    candidate_ranks = [r for r in candidate_ranks if r >= lower_bound and r < rank]
    candidate_ranks.append(rank)  # fallback
    
    for target_rank in candidate_ranks:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_budget = min(30, DEADLINE - 10 - (time.time() - start_time))
        if time_budget < 2:
            break
        
        result = _try_rank(n, m, p, target_rank, time_budget, seed=int(time.time() * 1000) % 100000)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    print(f"Time: {time.time() - start:.2f}s, Rank: {rank}")
    
    # Verify
    assert _verify(U, V, W, n, m, p, rank)
    print("Decomposition verified.")
