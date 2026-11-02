import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = delta[i,k] for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 if and only if the (i,j) of A and (j,k) of B contribute to (i,k) of C
    # Actually, the tensor is: T[i,j,k] = 1 for all i,j,k where the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The multiplication tensor T satisfies:
    # C[i,k] = sum_{j} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need this to equal sum_j A[i,j]*B[j,k]
    # So for each (i,j,k), the coefficient should be 1 (since A[i,j]*B[j,k] appears once)
    # Actually T[i,j,k] = 1 for all i,j,k - it's the full tensor of 1s? No...
    
    # Let me reconsider. The matrix multiplication tensor T is a 3-tensor where
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give the (i,k) entry of C.
    # Since each pair (i,j) in A and (j,k) in B uniquely determines contribution to (i,k) in C,
    # and each such product appears exactly once, T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Check a sample of entries for speed, then check all if sample passes
    # First check all entries (n*m*p = 105 checks, each with rank multiplications)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_r_search(n, m, p, target_rank, time_limit):
    """Try to find a rank-`target_rank` decomposition using structured approaches."""
    start_time = time.time()
    nmp = n * m * p
    
    # Approach: Try to find U, V, W of rank target_rank such that
    # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations in n*m*rank + m*p*rank + p*n*rank unknowns.
    # For small rank reductions, we can try to solve this linear system.
    
    # Let's set up the linear system. Let x be the vector of all unknowns.
    # Unknowns: U has n*m*rank vars, V has m*p*rank vars, W has p*n*rank vars.
    # Total unknowns: rank * (n*m + m*p + p*n)
    # Equations: n*m*p = 105
    
    total_unknowns = target_rank * (n*m + m*p + p*n)
    
    # For target_rank close to n*m*p, this is underdetermined.
    # We can use a greedy/structured approach: start with schoolbook and try to merge terms.
    
    # Alternative: Use the fact that if we can find a set of rank-1 tensors that sum to T,
    # we can try random linear combinations.
    
    # Let's try a different approach: parameterize U, V, W with structure.
    # For example, try circulant or Kronecker product structures.
    
    # Simple approach: try to find a solution by solving the linear system
    # with a minimum-norm solution (least squares / pseudoinverse).
    
    # Build the system: for each (i,j,k), we have an equation.
    # The unknowns are indexed by (matrix, row, col) where matrix is U/V/W.
    
    # Let's build the constraint matrix A such that A @ x = 1 (vector of ones of length n*m*p)
    # where x is the vector of all U, V, W entries.
    
    # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is nonlinear in x! The product of three variables.
    
    # So this is not a linear system. We need a different approach.
    
    # Approach: Fix two of the three matrices and solve for the third.
    # If we fix U and V, then for each (i,j,k):
    #   sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
    # This is linear in W!
    
    # So: fix U and V (randomly or with some structure), then solve for W.
    # Then verify. If it works, great. If not, try different U, V.
    
    # Let's try this approach.
    
    np.random.seed(42)
    
    # Start with schoolbook-like U and V, then perturb
    # Actually, let's try random U and V with small values
    
    best_solution = None
    
    num_attempts = 50000  # Will be limited by time
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random U and V with entries from {-1, 0, 1} or small integers
        # Scale to keep values reasonable
        U_rand = np.random.randint(-2, 3, size=(n*m, target_rank), dtype=np.float64)
        V_rand = np.random.randint(-2, 3, size=(m*p, target_rank), dtype=np.float64)
        
        # For each (i,j,k), we need: sum_r W[k*n+i, r] * U_rand[i*m+j, r] * V_rand[j*p+k, r] = 1
        # This is a linear system in W.
        # The system is: for each (i,j,k), one equation involving W[k*n+i, :]
        # But W[k*n+i, :] appears in multiple equations (for different j).
        
        # Let's group by the W index. For a fixed (k,i), the W row is W[k*n+i, :].
        # It appears in equations for all j.
        # For fixed (k,i): sum_j [equation for (i,j,k)]
        # Each equation: sum_r W[k*n+i, r] * U_rand[i*m+j, r] * V_rand[j*p+k, r] = 1
        
        # So for fixed (k,i), we have m equations (one for each j) and target_rank unknowns (W[k*n+i, :]).
        # If m >= target_rank, we can solve. If m < target_rank, we have a least-squares problem.
        
        # m = 5, so if target_rank <= 5, we can potentially solve exactly.
        # But target_rank is likely much larger than 5 for this problem.
        
        # Actually, the W[k*n+i, :] row is shared across all j for fixed (k,i).
        # So we have a coupled system. Let's set up the full linear system in W.
        
        # Unknowns: W has p*n*target_rank = 7*3*target_rank = 21*target_rank variables.
        # Equations: n*m*p = 105 equations.
        
        # For equation (i,j,k): sum_r W[k*n+i, r] * (U_rand[i*m+j, r] * V_rand[j*p+k, r]) = 1
        # Let C[i,j,k,r] = U_rand[i*m+j, r] * V_rand[j*p+k, r]
        # Then: sum_r W[k*n+i, r] * C[i,j,k,r] = 1
        
        # This is linear in W. Let's build the system.
        
        num_W_vars = p * n * target_rank
        num_eqs = n * m * p
        
        # Build A matrix: A has shape (num_eqs, num_W_vars)
        # A[e, w] = C[i,j,k,r] where e corresponds to (i,j,k) and w corresponds to (k,i,r)
        
        A = np.zeros((num_eqs, num_W_vars), dtype=np.float64)
        b = np.ones(num_eqs, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        w_idx = (k * n + i) * target_rank + r
                        A[eq_idx, w_idx] = U_rand[i*m+j, r] * V_rand[j*p+k, r]
                    eq_idx += 1
        
        # Solve A @ W_vec = b using least squares
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is good
        residual = np.max(np.abs(A @ W_vec - b))
        if residual > 1e-8:
            continue
        
        # Check if W entries are integer or half-integer (within tolerance)
        W_arr = W_vec.reshape(p * n, target_rank)
        
        # Check if entries are close to integers or half-integers
        # Multiply by 2 and check if close to integers
        W_doubled = W_arr * 2
        W_rounded = np.round(W_doubled)
        if np.max(np.abs(W_doubled - W_rounded)) > 1e-6:
            continue
        
        # Also check U and V for integer/half-integer
        U_doubled = U_rand * 2
        U_rounded = np.round(U_doubled)
        if np.max(np.abs(U_doubled - U_rounded)) > 1e-6:
            continue
        
        V_doubled = V_rand * 2
        V_rounded = np.round(V_doubled)
        if np.max(np.abs(V_doubled - V_rounded)) > 1e-6:
            continue
        
        # Convert to lists
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = (W_rounded / 2.0).tolist()
        
        # Verify
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            best_solution = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_solution

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    # Start from rank 104 and go down
    for target_rank in range(rank_sb - 1, max(rank_sb - 50, 1), -1):
        if time.time() > deadline - 10:
            break
        
        time_limit = min(20, deadline - time.time() - 5)
        if time_limit <= 0:
            break
        
        solution = _try_rank_r_search(n, m, p, target_rank, time_limit)
        if solution is not None:
            best_solution = solution
            best_rank = target_rank
            # Keep searching for even lower rank
            continue
        else:
            # No solution found at this rank, try lower
            continue
    
    return best_solution
# EVOLVE-BLOCK-END
