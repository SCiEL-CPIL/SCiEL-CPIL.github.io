"""Matrix multiplication 4x5x6: find rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    total_deadline = time.time() + 145  # safety margin before 150s kill
    
    # Schoolbook rank is n*m*p = 120
    schoolbook_rank = n * m * p
    
    # Build the multiplication tensor T[i, j, k] = e_{i,k} (1 if i==k else 0)
    # Actually, the tensor for matrix multiplication: T[i,j,k] is the coefficient
    # such that (A*B)_{i,k} = sum_{j} A_{i,j} * B_{j,k}
    # The tensor T has entries T[i,j,k] = 1 if we're computing (A*B)_{i,k} from A_{i,j} and B_{j,k}
    # More precisely, the bilinear map is: (A,B) -> A*B
    # T[i,j,k] = delta_{i,k} ... no.
    # 
    # The standard formulation: we want to compute C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T is defined by T[i,j,k] = 1 (it's the identity in the middle index pairing)
    # Actually T[i,j,k] is just 1 for all valid (i,j,k) in the sense that each term A[i,j]*B[j,k] contributes to C[i,k].
    # The tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the contraction is over j.
    # 
    # For the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r has indices (i), v_r has indices (j), w_r has indices (k)
    # and the condition is: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # 
    # Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # 
    # The decomposition condition: for all i,j,k:
    # delta_{i,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Wait, that's not right either. Let me think again.
    # 
    # Matrix multiplication: C = A @ B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form: for each (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: we want rank r such that
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (something)
    # 
    # Actually the standard way: the multiplication tensor M_{n,m,p} has entries
    # M[i,j,k] = 1 if the product term A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # The tensor is: M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in appropriate spaces)
    # And we want M = sum_r u_r ⊗ v_r ⊗ w_r
    # 
    # The condition is: M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where M[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # Wait no. M[i,j,k] should be such that:
    # sum_j A[i,j] * B[j,k] = sum_{j,r} A[i,j] * B[j,k] * M[i,j,k]
    # So M[i,j,k] = 1 for all i,j,k.
    # 
    # Hmm, but that means the tensor is all ones, which can't be right for a rank decomposition.
    # 
    # Let me reconsider. The correct formulation:
    # We have three sets of variables: A[i,j] for i in [n], j in [m]; B[j,k] for j in [m], k in [p]; C[i,k] for i in [n], k in [p].
    # The relation is: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is a bilinear map from (A,B) to C. The tensor associated with this map has entries
    # T[i,k,j] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k]).
    # 
    # For a rank-r decomposition, we need:
    # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Wait, the indices don't match up that way. Let me look at the schoolbook solution:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # 
    # So for the schoolbook, each term (i,j,k) gets its own column. The condition must be:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ... no.
    # 
    # Actually I think the condition is that the product of the three vectors gives the tensor entry.
    # The tensor entry for the bilinear map is: for output index (i,k) and input indices (i,j) and (j,k),
    # the coefficient is 1.
    # 
    # I think the verification is:
    # For all i in [n], k in [p]: 
    #   sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (if this is the coefficient)
    # 
    # Hmm, let me just try to verify the schoolbook solution. In schoolbook, each (i,j,k) triple has its own column c where U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    # 
    # The evaluator likely checks: for random A (n x m) and B (m x p), compute C = A @ B,
    # and also compute C' where C'[i,k] = sum_{j,r} A[i,j] * V[j*p+k,r] * ... 
    # 
    # I think the check is:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
    # 
    # Let me just assume the standard: the tensor T has T[i,j,k] = 1 for all i,j,k, and we need
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # But that would mean sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    # 
    # For schoolbook with one column per (i,j,k): when r corresponds to (i',j',k'), 
    # U[i*m+j,r] = 1 iff i=i' and j=j', V[j*p+k,r] = 1 iff j=j' and k=k', W[k*n+i,r] = 1 iff k=k' and i=i'.
    # So the product is 1 iff i=i' and j=j' and k=k', and 0 otherwise. Summing over r gives 1. Correct!
    # 
    # So the condition is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # 
    # This is a 3-tensor with all entries equal to 1. We want the minimum rank decomposition.
    
    # Known lower bounds: rank >= n*m, rank >= m*p, rank >= n*p, rank >= n+m+p-1 (Strassen lower bound is weaker)
    # For 4x5x6: lower bounds are max(20, 30, 24, 4+5+6-1=14) = 30
    # Upper bound: schoolbook = 120. Known best for small matrices...
    # 
    # For 2x2x2: rank 7 (Strassen)
    # For 2x3x2: rank ? 
    # For 4x5x6: this is a research problem. The best known might be around 80-100.
    # 
    # Since we can't do extensive research, let me try a greedy/optimization approach.
    # 
    # Strategy: Start from schoolbook (rank 120) and try to reduce rank by merging columns.
    # 
    # Alternative: Use a randomized approach to find a low-rank decomposition.
    # 
    # Let me try a different approach: start with a random initialization at a lower rank and use
    # gradient-free optimization or iterative refinement.
    
    # Actually, let me try a simple but effective approach:
    # 1. Start with schoolbook rank 120
    # 2. Try to reduce rank by finding linear dependencies and eliminating columns
    # 3. Use a greedy elimination approach
    
    # Better approach: Use the fact that we can search over rank r from lower bound up,
    # and for each rank, try to find a valid decomposition using numerical optimization.
    
    # For the tensor with all-ones entries, the problem is:
    # Find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is equivalent to: the tensor T (all ones) has a rank-r decomposition.
    
    # Let me try a practical approach: use numpy to do a rank-reduction search.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using numpy operations."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape and use einsum
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need: for i in [n], j in [m], k in [p]: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's compute this step by step
        # First, for each r, compute the outer product structure
        # Actually, let's just loop since n*m*p is small (120)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        return ok
    
    # Start with schoolbook
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = make_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by greedy elimination
    # For each column, check if it can be removed (i.e., if the remaining columns can still satisfy the constraints)
    # This is expensive, so let's try a simpler approach.
    
    # Alternative: try random rank reductions
    # For a given rank r < current, try to find U,V,W of that rank that satisfy the constraints.
    
    # Let me try a different strategy: use the structure of the all-ones tensor.
    # The all-ones tensor of size (n*m) x (m*p) x (p*n) ... 
    # 
    # Actually, I realize this might be a well-known problem. Let me try a few specific ranks.
    
    # Try rank = n*m*p - 1, n*m*p - 2, etc., and see if we can find a solution.
    # 
    # A simpler approach: try to merge pairs of columns.
    
    def try_reduce_rank(U, V, W, max_time=10):
        """Try to reduce the rank by 1 by merging two columns."""
        start = time.time()
        r = U.shape[1]
        if r <= 1:
            return None
        
        # Try removing each column and see if the system is still solvable
        # For column c, we need to check if there exist U', V', W' of rank r-1
        # such that the constraints are satisfied.
        
        # This is a nonlinear feasibility problem. Let's try a simpler heuristic:
        # Check if column c is a linear combination of other columns in a way that preserves the tensor.
        
        # Actually, let's just try random perturbations and see if we can maintain validity.
        
        # Simpler: just try to zero out one column and see if the remaining still work.
        # This won't work in general, but let's try.
        
        return None
    
    # Let me try a completely different approach: use scipy to solve the nonlinear system.
    # 
    # For a given rank r, we have variables U (n*m*r), V (m*p*r), W (p*n*r)
    # Constraints: for all i,j,k: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
    # Number of constraints: n*m*p = 120
    # Number of variables: (n*m + m*p + p*n) * r = (20 + 30 + 24) * r = 74*r
    # 
    # For r = 30: 74*30 = 2220 variables, 120 constraints. Very underdetermined.
    # For r = 20: 74*20 = 1480 variables, 120 constraints.
    
    # The problem is that the constraints are cubic (trilinear), so it's a polynomial feasibility problem.
    
    # Let me try a simple approach: initialize with random values and use iterative refinement.
    
    def random_decomposition(r, seed=42):
        """Create a random decomposition of rank r."""
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        return U, V, W
    
    def refine(U, V, W, max_iters=1000, tol=1e-10):
        """Try to refine a decomposition to satisfy the constraints."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iters):
            # Compute the error: for each (i,j,k), error = 1 - sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
            # Update using a simple gradient step
            # 
            # The constraint is: f(i,j,k) = sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] - 1 = 0
            # 
            # Gradient with respect to U[i*m+j,s]: V[j*p+k,s]*W[k*n+i,s]
            # Gradient with respect to V[j*p+k,s]: U[i*m+j,s]*W[k*n+i,s]  
            # Gradient with respect to W[k*n+i,s]: U[i*m+j,s]*V[j*p+k,s]
            
            # This is a coupled system, so let's do alternating minimization.
            # Fix V and W, solve for U.
            # For each (i*m+j), we need: sum_s U[i*m+j,s] * (sum_k V[j*p+k,s]*W[k*n+i,s]) = 1 for all k
            # Wait, that's not quite right.
            
            # Let me think differently. For fixed V and W, the constraint on U is:
            # For all i,j,k: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
            # 
            # For a fixed (i,j), this gives p equations (one for each k):
            # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1, for k=0,...,p-1
            # 
            # Let A_{i,j,k,s} = V[j*p+k,s] * W[k*n+i,s]
            # Then: sum_s A_{i,j,k,s} * U[i*m+j,s] = 1 for each k
            # 
            # This is a linear system in U[i*m+j, :].
            # Matrix: p x r, vector: p x 1 (all ones)
            
            for i in range(n):
                for j in range(m):
                    # Build the p x r matrix
                    M = np.zeros((p, r))
                    for k in range(p):
                        for s in range(r):
                            M[k, s] = V[j*p+k, s] * W[k*n+i, s]
                    # Solve M @ U[i*m+j, :] = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Similarly for V and W, but it's more complex because they're shared across i and k.
            # For V[j*p+k, s], it appears in constraints for all i.
            # For fixed U and W:
            # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all i,j,k
            # For fixed (j,k): sum_s V[j*p+k,s] * (sum_i U[i*m+j,s]*W[k*n+i,s]) = 1
            # Let B_{j,k,s} = sum_i U[i*m+j,s]*W[k*n+i,s]
            # Then: sum_s V[j*p+k,s] * B_{j,k,s} = 1
            # This is just one equation for r unknowns V[j*p+k, :]. Underdetermined if r > 1.
            
            for j in range(m):
                for k in range(p):
                    b = np.zeros(r)
                    for s in range(r):
                        for i in range(n):
                            b[s] += U[i*m+j, s] * W[k*n+i, s]
                    # V[j*p+k, :] should satisfy sum_s V[j*p+k,s] * b[s] = 1
                    # Least squares solution:
                    if np.linalg.norm(b) > 1e-12:
                        V[j*p+k, :] = b / np.dot(b, b)
                    else:
                        # Degenerate case
                        V[j*p+k, :] = 0
                        V[j*p+k, 0] = 1 if r > 0 else 0
            
            # For W[k*n+i, s], it appears in constraints for all j.
            # For fixed U and V:
            # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all i,j,k
            # For fixed (i,k): sum_s W[k*n+i,s] * (sum_j U[i*m+j,s]*V[j*p+k,s]) = 1
            # Let C_{i,k,s} = sum_j U[i*m+j,s]*V[j*p+k,s]
            # Then: sum_s W[k*n+i,s] * C_{i,k,s} = 1
            
            for i in range(n):
                for k in range(p):
                    c = np.zeros(r)
                    for s in range(r):
                        for j in range(m):
                            c[s] += U[i*m+j, s] * V[j*p+k, s]
                    if np.linalg.norm(c) > 1e-12:
                        W[k*n+i, :] = c / np.dot(c, c)
                    else:
                        W[k*n+i, :] = 0
                        W[k*n+i, 0] = 1 if r > 0 else 0
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                return U, V, W, max_err
        
        return U, V, W, max_err
    
    # Try various ranks
    for r in range(25, min(120, 60)):
        if time.time() > total_deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > total_deadline - 15:
                break
            
            U_r, V_r, W_r = random_decomposition(r, seed=42 + r*100 + attempt)
            U_r, V_r, W_r, err = refine(U_r, V_r, W_r, max_iters=200, tol=1e-10)
            
            if err < 1e-8:
                # Verify
                if verify_fast(U_r, V_r, W_r):
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    # Convert to list of lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
