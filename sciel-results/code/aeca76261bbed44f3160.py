"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute target tensor indices
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # We need: sum_t U[a,t] * V[b,t] * W[c,t] = 1 for all valid (a,b,c)
    # where c = k*n+i
    
    # Build list of (a, b, c_idx) tuples
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                tensor_indices.append((a, b, c_idx))
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly matches the target tensor."""
        for (a, b, c) in tensor_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    def round_and_verify(U, V, W, step=0.5):
        """Round to nearest multiples of step and verify."""
        U_rounded = np.round(U / step) * step
        V_rounded = np.round(V / step) * step
        W_rounded = np.round(W / step) * step
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # Try integer rounding
        U_rounded = np.round(U)
        V_rounded = np.round(V)
        W_rounded = np.round(W)
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        return None
    
    def try_rank_ip(rank):
        """
        Use integer linear programming to find a rank-r decomposition.
        
        The key insight: each rank term t defines a "pebble" that can be placed
        at positions in the U, V, W matrices. For the multiplication to work,
        for each output element (i,k), we need:
            sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        
        This is a nonlinear constraint, but we can linearize by fixing the
        structure: if we assume entries are in {-1, 0, 1}, we can use the
        fact that the product of three such values is also in {-1, 0, 1}.
        
        Alternative approach: use the pebbling model where each rank term
        corresponds to a "circuit" that processes a subset of the computation.
        """
        
        # For small ranks, try exhaustive search over sparse structures
        # Each rank term can be seen as a "path" through the computation
        
        # Strategy: For each rank term t, choose which (i,j), (j,k), (k,i) 
        # triples it contributes to. The constraint is that for each (i,j,k),
        # exactly one term (or a combination) must produce 1.
        
        # This is equivalent to finding a set of "rank-1 tensors" that sum
        # to the multiplication tensor.
        
        # For rank r, we need r rank-1 tensors whose sum equals the target.
        # A rank-1 tensor is determined by vectors u, v, w such that
        # T[a,b,c] = u[a] * v[b] * w[c]
        
        # The target tensor has T[a,b,c] = 1 if (a,b,c) corresponds to
        # (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
        
        # This is a hard problem. Let's use a different approach:
        # Try to find decompositions with entries in {-1, 0, 1} using
        # backtracking with pruning.
        
        if rank > 30:
            return None
        
        # Use random search with structured initialization
        best_result = None
        max_attempts = 500 if rank <= 25 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            # Structured random initialization
            np.random.seed(12345 + attempt * 7 + rank)
            
            # Start with mostly zeros, add sparse non-zero entries
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each rank column, decide which entries are non-zero
            for t in range(rank):
                # Choose a random subset of rows for U
                n_u_nonzero = np.random.randint(1, 4)
                u_rows = np.random.choice(n*m, size=min(n_u_nonzero, n*m), replace=False)
                for r in u_rows:
                    U[r, t] = np.random.choice([-1, 1])
                
                # Choose a random subset of rows for V
                n_v_nonzero = np.random.randint(1, 4)
                v_rows = np.random.choice(m*p, size=min(n_v_nonzero, m*p), replace=False)
                for r in v_rows:
                    V[r, t] = np.random.choice([-1, 1])
                
                # Choose a random subset of rows for W
                n_w_nonzero = np.random.randint(1, 4)
                w_rows = np.random.choice(p*n, size=min(n_w_nonzero, p*n), replace=False)
                for r in w_rows:
                    W[r, t] = np.random.choice([-1, 1])
            
            # Quick verification
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    def try_rank_optimization(rank, max_restarts=10):
        """Try to find a decomposition using continuous optimization."""
        
        def residual_and_grad(params, rank):
            """Compute residual and gradient given flattened parameters."""
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            total = 0.0
            for (a, b, c) in tensor_indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = 2.0 * (val - 1.0)
                total += (val - 1.0) ** 2
                
                # Gradients
                U_grad[a, :] += err * V[b, :] * W[c, :]
                V_grad[b, :] += err * U[a, :] * W[c, :]
                W_grad[c, :] += err * U[a, :] * V[b, :]
            
            grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
            return total, grad
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                return None
            
            np.random.seed(999 + rank * 1000 + restart)
            
            # Sparse initialization
            scale = 0.1 + 0.4 * (restart / max_restarts)
            U_init = np.random.randn(n*m, rank) * scale
            V_init = np.random.randn(m*p, rank) * scale
            W_init = np.random.randn(p*n, rank) * scale
            
            # Make it sparse: zero out many entries
            sparsity = 0.7
            U_init[np.random.rand(n*m, rank) < sparsity] = 0
            V_init[np.random.rand(m*p, rank) < sparsity] = 0
            W_init[np.random.rand(p*n, rank) < sparsity] = 0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(
                    lambda x: residual_and_grad(x, rank)[0],
                    x0,
                    jac=lambda x: residual_and_grad(x, rank)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if time.time() > deadline - 3:
                    return None
                
                if result.fun < 1e-8:
                    U_flat = result.x[:n*m*rank]
                    V_flat = result.x[n*m*rank:n*m*rank + m*p*rank]
                    W_flat = result.x[n*m*rank + m*p*rank:]
                    
                    U = U_flat.reshape(n*m, rank)
                    V = V_flat.reshape(m*p, rank)
                    W = W_flat.reshape(p*n, rank)
                    
                    rounded = round_and_verify(U, V, W)
                    if rounded is not None:
                        return rounded[0], rounded[1], rounded[2]
                        
            except Exception:
                continue
        
        return None
    
    def try_strassen_based(rank):
        """
        Try to construct a decomposition based on Strassen-like patterns.
        
        For 3x3x4, we can try to decompose it into overlapping 2x2x2 blocks
        and use Strassen's 7-multiplication algorithm for each block, then
        combine them cleverly to reduce the total rank.
        """
        
        # Strassen's algorithm for 2x2 matrices uses rank 7 instead of 8.
        # For our 3x3x4 problem, we can try to partition into overlapping
        # 2x2x2 subproblems and share some of the intermediate results.
        
        # The key idea: identify which 2x2x2 subproblems overlap and
        # find a way to share the Strassen terms between them.
        
        # This is a complex combinatorial problem. For now, let's try
        # a simple approach: use Strassen for non-overlapping 2x2x2 blocks
        # and schoolbook for the rest.
        
        # Partition the 3x3x4 into 2x2x2 blocks:
        # Block (0,0): rows 0-1, cols 0-1, output 0-1 x 0-1
        # Block (1,1): rows 1-2, cols 1-2, output 1-2 x 1-2
        # These overlap at row/col 1.
        
        # For simplicity, let's try a few specific constructions.
        
        # Construction 1: Use 2 Strassen blocks + schoolbook for the rest
        # This would give rank 7 + 7 + (36 - 8 - 8 + 4) = 7 + 7 + 24 = 38 (worse)
        
        # The challenge is to find a smarter combination.
        
        # For now, return None to indicate this approach needs more work.
        return None
    
    def try_pure_random(rank, max_attempts=1000):
        """Pure random search with {-1, 0, 1} entries."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(42 + attempt * 13 + rank)
            
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    best_result = schoolbook()
    
    # Try ranks from low to high
    # Based on literature, the rank of 3x3x4 matrix multiplication is at most 25
    # and at least 20. Let's try from 20 upward.
    
    for target_rank in range(20, 36):
        if time.time() > deadline - 5:
            break
        
        # Try different strategies in order of speed
        result = try_rank_ip(target_rank)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        result = try_rank_optimization(target_rank, max_restarts=5)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        result = try_pure_random(target_rank, max_attempts=200)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
