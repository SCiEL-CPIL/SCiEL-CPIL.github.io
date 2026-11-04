"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    The tensor T has indices (i,j,k) with i in [0,3), j in [0,4), k in [0,4).
    We seek the minimum rank r and matrices U (12 x r), V (16 x r), W (12 x r)
    such that for all (i,j,k):
        T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*3+i, c]
    
    Since T[i,j,k] = 1 for all valid indices, we need:
        sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*3+i, c] = 1
    
    Strategy: Use a structured decomposition based on the Strassen-like approach
    or direct search for low-rank decompositions.
    
    For 3x4 * 4x4, the trivial rank is 48. We search for better decompositions.
    
    Key insight: We can use the fact that the multiplication tensor for m×n * p×q
    can sometimes be decomposed with rank less than m*n*p.
    
    Let's try a systematic approach:
    1. Start with known good decompositions
    2. Try to improve using linear algebra / optimization
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds, leaving 10s margin
    
    # The trivial schoolbook decomposition has rank 48
    # Let's try to find a better one
    
    # Approach: Use a parametric family of decompositions
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is T[i,j,k] = 1 (after appropriate indexing)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try rank-structured decompositions.
    # One idea: use the fact that we can factor through intermediate representations.
    
    # For now, let's construct a valid decomposition and try to minimize rank.
    # 
    # A key observation: if we can write the multiplication as a sum of r rank-1
    # tensors, each rank-1 tensor corresponds to a "multiplication circuit" term.
    
    # Let's try a direct construction. For 3x4 * 4x4:
    # We can potentially use a decomposition based on polynomial multiplication
    # or other algebraic identities.
    
    # Actually, let's try a simpler approach first: verify if there's a known
    # lower rank for this specific case.
    
    # The rank of matrix multiplication tensor M_{n,m,p} is known for small cases.
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{3,3,3}, the rank is between 19 and 23 (not exactly known in general,
    # but specific decompositions exist).
    
    # For 3x4 * 4x4, let's try to construct a decomposition.
    
    # Strategy: Use a block decomposition.
    # A is 3x4, B is 4x4.
    # We can think of this as:
    #   C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # One approach: use the identity that allows us to reduce the number of
    # multiplications. For example, if we group terms.
    
    # Let's try a direct search for a low-rank decomposition using the
    # following ansatz:
    #
    # U[i*m+j, c] = u_{i,j,c}
    # V[j*p+k, c] = v_{j,k,c}  
    # W[k*n+i, c] = w_{k,i,c}
    #
    # Constraint: sum_c u_{i,j,c} * v_{j,k,c} * w_{k,i,c} = 1 for all i,j,k
    
    # This is a system of nonlinear equations. Let's try to find a solution
    # with small r.
    
    # For r = 1: u_{i,j,0} * v_{j,k,0} * w_{k,i,0} = 1 for all i,j,k
    # This requires u_{i,j,0} = a_i, v_{j,k,0} = b_j, w_{k,i,0} = c_k * d_i
    # so a_i * b_j * c_k * d_i = 1, which means (a_i * d_i) * b_j * c_k = 1
    # This can work if all products are 1, e.g., a_i=1, d_i=1, b_j=1, c_k=1.
    # But wait, that gives rank 1! Let me check...
    #
    # If U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U*V*W = 1*1*1 = 1. Yes!
    #
    # But wait, that's not right. The tensor decomposition requires that the
    # product represents the actual multiplication. Let me reconsider.
    
    # The matrix multiplication tensor T has entries T[i,j,k] which is 1 if
    # we're computing C[i,k] = sum_j A[i,j] B[j,k]. The "tensor" is the
    # bilinear form, and we're looking for a rank decomposition of the
    # associated 3-tensor.
    
    # Actually, I think I was overcomplicating. Let me re-read the task.
    
    # The task says: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n,m,p) has entries indexed by
    # (i, j, k) where i in [0,n), j in [0,m), k in [0,p), and the entry is
    # 1 (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k]
    # in C[i,k] is 1).
    
    # A rank-1 decomposition term is u_{i,j} * v_{j,k} * w_{k,i} (with appropriate
    # index conventions). The sum over all terms should give T[i,j,k] = 1.
    
    # Wait, I need to be more careful about the index conventions.
    
    # From the seed program:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # And the constraint is that for the tensor to be correct:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give the
    # appropriate coefficient.
    
    # For matrix multiplication, the "tensor" T[i,j,k] = 1 for all valid
    # (i,j,k) because each product A[i,j]*B[j,k] contributes with coefficient 1
    # to C[i,k].
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    
    # With n=3, m=4, p=4:
    # U is 12 x r, V is 16 x r, W is 12 x r.
    
    # If we set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all
    # indices, then the sum is 1. This gives rank 1!
    
    # But this can't be right because the rank of matrix multiplication tensor
    # is known to be greater than 1 for non-trivial cases.
    
    # Let me reconsider. I think the issue is that the tensor decomposition
    # for matrix multiplication is not just about the scalar entries being 1,
    # but about the bilinear structure.
    
    # Actually, I think I'm confusing two different things. Let me look at this
    # more carefully.
    
    # The matrix multiplication map is:
    #   (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}.
    #
    # The associated tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} has entries
    # T[(i,j), (j',k), (i',k')] which is 1 if i=i', j=j', k=k' and 0 otherwise.
    #
    # Wait, no. Let me think again.
    #
    # The bilinear map is: f(A, B)[i,k] = sum_j A[i,j] B[j,k]
    #
    # The tensor T is defined such that f(A,B)[i,k] = sum_{j,j'} A[i,j] B[j',k] T[(i,j),(j',k),(i,k)]
    #
    # So T[(i,j),(j',k),(i,k)] = 1 if j=j' and 0 otherwise.
    #
    # In the indexing used in the problem:
    # U has shape (n*m, rank), indexed by i*m+j
    # V has shape (m*p, rank), indexed by j*p+k
    # W has shape (p*n, rank), indexed by k*n+i
    #
    # The constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[(i,j),(j,k),(k,i)]
    #
    # And T[(i,j),(j,k),(k,i)] = 1 if the indices match appropriately.
    #
    # For matrix multiplication, T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # But in the W indexing, we have k*n+i, so the third index is (k,i).
    #
    # So T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # In the sum, we're summing over c, and for fixed (i,j,k) in the output:
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to equal
    # T[(i,j),(j,k),(k,i)]
    #
    # But T[(i,j),(j,k),(k,i)] = 1 (since j=j, i=i, k=k).
    #
    # Hmm, but that would mean the same issue. Let me look at the seed program
    # more carefully.
    
    # The seed program sets for each (i,j,k):
    #   U[i*m+j][column] = 1
    #   V[j*p+k][column] = 1
    #   W[k*n+i][column] = 1
    #
    # So for a given column c corresponding to (i0,j0,k0):
    #   U[i*m+j, c] = 1 if (i,j) = (i0,j0)
    #   V[j*p+k, c] = 1 if (j,k) = (j0,k0)
    #   W[k*n+i, c] = 1 if (k,i) = (k0,i0)
    #
    # So the product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 only if
    # (i,j) = (i0,j0), (j,k) = (j0,k0), (k,i) = (k0,i0), which means
    # i=i0, j=j0, k=k0.
    #
    # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if there exists a
    # column with (i0,j0,k0) = (i,j,k), which is true for exactly one column.
    #
    # So the sum equals 1 for all (i,j,k). This matches T[(i,j),(j,k),(k,i)] = 1.
    
    # So the question is: can we do this with fewer than 48 columns?
    
    # The answer depends on the tensor rank. For the matrix multiplication
    # tensor M_{n,m,p}, the rank is the minimum number of rank-1 tensors
    # needed to express it.
    
    # For M_{2,2,2}, the rank is 7 (Strassen).
    # For M_{3,3,3}, the rank is at most 23 and at least 19.
    #
    # For M_{3,4,4}, I'm not sure of the exact rank, but it's likely less than 48.
    
    # Let's try to find a better decomposition.
    
    # One approach: use a parametric search. We can set up the problem as:
    # Find r and matrices U, V, W such that the constraints are satisfied.
    
    # For small r, we can try to solve this system.
    
    # Let's try r from some lower bound up to 48 and see if we can find a solution.
    
    # Lower bound: The rank is at least max(n*m, m*p, n*p) = max(12, 16, 12) = 16.
    # Actually, the rank is at least the border rank, which is at least the
    # geometric rank. For M_{n,m,p}, the rank is at least max(n*m, m*p, n*p).
    
    # So the lower bound is 16. Let's try to find a decomposition with rank
    # between 16 and 48.
    
    # This is a hard problem. Let's try a few strategies:
    
    # Strategy 1: Use a known decomposition for a sub-problem and extend it.
    # Strategy 2: Use a randomized search.
    # Strategy 3: Use a structured ansatz.
    
    # Let's try Strategy 3 first. We can use the fact that the problem has
    # some structure.
    
    # For 3x4 * 4x4, we can think of it as:
    #   C = A * B where A is 3x4 and B is 4x4.
    #
    # One idea: split B into two 4x2 blocks: B = [B1 | B2] where B1, B2 are 4x2.
    # Then C = [A*B1 | A*B2].
    #
    # This doesn't immediately help.
    
    # Another idea: use the identity for 2x2 multiplication (Strassen) and
    # extend it.
    
    # Actually, let's just try to construct a valid decomposition with rank
    # less than 48 and see if we can find one.
    
    # For now, let's return the schoolbook decomposition as a fallback, and
    # try to improve it.
    
    def schoolbook():
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
    
    # Let's try to find a better decomposition.
    # 
    # Idea: Use a decomposition based on the following observation.
    # 
    # For matrix multiplication, we can use the following identity:
    #   (a+b)(c+d) = ac + ad + bc + bd
    # 
    # This doesn't reduce the number of multiplications. But Strassen's trick
    # for 2x2 is:
    #   p1 = (a00+a11)(b00+b11)
    #   p2 = (a10+a11)b00
    #   p3 = a00(b10-b11)
    #   p4 = a11(b01-b00)
    #   p5 = (a00+a01)b11
    #   p6 = (a10-a00)(b00+b10)
    #   p7 = (a01-a11)(b01+b11)
    # 
    # This uses 7 multiplications instead of 8.
    
    # For 3x4 * 4x4, let's try to use a similar approach.
    
    # Actually, let's try a different approach. Let's use the fact that we can
    # write the multiplication as a sum of rank-1 terms, where each term
    # corresponds to a "circuit" for a subset of the products.
    
    # Let's try to find a decomposition with rank r by solving a system of
    # equations.
    
    # For a given r, we have:
    #   U: 12 x r
    #   V: 16 x r
    #   W: 12 x r
    # 
    # Total unknowns: 12r + 16r + 12r = 40r
    # Constraints: 3*4*4 = 48
    #
    # So for r >= 2, we have more unknowns than constraints, which suggests
    # solutions may exist. But the equations are nonlinear (trilinear).
    
    # Let's try r = 16 (the lower bound) and see if we can find a solution.
    
    # This is a hard nonlinear optimization problem. Let's try a few approaches.
    
    # Approach: Use a randomized initialization and gradient descent.
    
    # But first, let's try a simpler structured approach.
    
    # Idea: Use a decomposition where U, V, W have a block structure.
    
    # For example, suppose we can write:
    #   U[i*m+j, c] = sum of products of some basis functions
    #
    # Let's try a specific ansatz.
    
    # Actually, let's just try to find a valid decomposition with rank < 48
    # using a search.
    
    # For now, let's return the schoolbook decomposition. We can try to improve
    # it later.
    
    # But wait, let's try one more idea. What if we use the following:
    #
    # For each i in [0,3), we have C[i,:] = A[i,:] * B
    #
    # A[i,:] is a 1x4 row vector. So C[i,:] is a 1x4 row vector computed as
    # a matrix-vector product.
    #
    # A matrix-vector product of a 1x4 vector with a 4x4 matrix requires
    # 16 multiplications in the naive approach. But we can potentially do
    # better.
    #
    # However, the three rows of C share the same B, so there might be
    # opportunities for sharing.
    
    # Let's try a different approach: use a known result.
    
    # The rank of M_{3,4,4} is not well-known to me, but let's try to find
    # a decomposition with rank less than 48.
    
    # Let's try rank 36 or so and see if we can find a solution.
    
    # Actually, let's just return the schoolbook decomposition for now, since
    # finding a better one is a hard problem.
    
    # But the task asks for rank-minimal, so we should try harder.
    
    # Let's try a simple improvement: can we reduce the rank by using the
    # fact that some of the products can be shared?
    
    # For example, consider the following:
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
    #   ...
    
    # There's no obvious sharing here.
    
    # Let's try a completely different approach: use a MILP or SAT solver.
    # But we don't have those available.
    
    # Let's try a randomized search for a low-rank decomposition.
    
    # For a given rank r, we can try to find U, V, W by:
    # 1. Initialize U, V, W randomly.
    # 2. Use gradient descent to minimize the error.
    
    # But the equations are trilinear, which makes this hard.
    
    # Let's try a different strategy: use the fact that the problem is
    # equivalent to finding a low-rank decomposition of a 3-tensor.
    
    # We can use the following approach:
    # 1. Fix U and V, and solve for W.
    # 2. Fix U and W, and solve for V.
    # 3. Fix V and W, and solve for U.
    # 4. Iterate.
    
    # This is similar to the ALS (Alternating Least Squares) approach for
    # tensor decomposition.
    
    # Let's try this.
    
    def try_rank(r, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to avoid overflow/underflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # for all k.
            
            # This is: U[i*m+j, :] * (V[j*p+0, :] * W[0*n+i, :]) = 1
            #           U[i*m+j, :] * (V[j*p+1, :] * W[1*n+i, :]) = 1
            #           ...
            
            # Let M be a p x r matrix where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # Then we need U[i*m+j, :] * M = [1, 1, ..., 1] (a row vector of 1s)
            
            # This is a linear system: M^T * U[i*m+j, :]^T = [1, 1, ..., 1]^T
            # Wait, U[i*m+j, :] is 1 x r, M is p x r.
            # U[i*m+j, :] * M^T is 1 x p, and we want it to be [1, 1, ..., 1].
            
            # So: U[i*m+j, :] * M^T = ones(1, p)
            # This is: M * U[i*m+j, :]^T = ones(p, 1)
            
            for i in range(n):
                for j in range(m):
                    # Build M
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve M * U[i*m+j, :]^T = ones(p, 1)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        return None
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # for all i.
            
            # Let M be a n x r matrix where M[i, c] = U[i*m+j, c] * W[k*n+i, c]
            # Then we need V[j*p+k, :] * M^T = [1, 1, ..., 1]
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        return None
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # for all j.
            
            # Let M be a m x r matrix where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # Then we need W[k*n+i, :] * M^T = [1, 1, ..., 1]
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        return None
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        error += (val - 1.0) ** 2
            
            if error < 1e-10:
                # Convert to fractions for exactness
                U_frac = [[Fraction(U[i*m+j, c]).limit_denominator(1000000) for c in range(r)] for i in range(n) for j in range(m)]
                V_frac = [[Fraction(V[j*p+k, c]).limit_denominator(1000000) for c in range(r)] for j in range(m) for k in range(p)]
                W_frac = [[Fraction(W[k*n+i, c]).limit_denominator(1000000) for c in range(r)] for k in range(p) for i in range(n)]
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                            if val != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    # Try ranks from the lower bound up to 48
    best_result = None
    for r in range(16, 49):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=50, seed=r)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
