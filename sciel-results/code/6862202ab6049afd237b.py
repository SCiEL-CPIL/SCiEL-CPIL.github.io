"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    The matrix multiplication tensor T of shape (2*4, 4*4, 4*2) = (8, 16, 8)
    satisfies: T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
    contribute to output entry, where output index is (k*n+i, r) with r being
    the row of A's column and i being the row of output.
    
    Actually, let me be more careful about the tensor structure.
    
    For multiplying A (2x4) by B (4x4), the result C is 2x4.
    C[i, r] = sum_j A[i, j] * B[j, r] for i in 0..1, r in 0..3.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Wait, the task says: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So the tensor is indexed as:
    - U has shape (n*m, rank) = (8, rank), indexed by (i*m+j) where i is row of A, j is column of A
    - V has shape (m*p, rank) = (16, rank), indexed by (j*p+k) where j is column of A = row of B, k is column of B
    - W has shape (p*n, rank) = (8, rank), indexed by (k*n+i) where k is column of B = column of C, i is row of C
    
    The tensor T[i*m+j, j*p+k, k*n+i] should be 1 for all valid (i,j,k), 0 otherwise.
    
    We want to find rank-minimal U, V, W such that:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    For the schoolbook algorithm, each (i,j,k) triple gets its own rank-1 term.
    """
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Start with schoolbook
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
        return U, V, W
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(W[0])
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Compute T[a,b,c]
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    # Expected: T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
                    # i = a // m, j = a % m
                    i = a // m
                    j = a % m
                    # j = b // p, k = b % p  
                    j2 = b // p
                    k = b % p
                    # k = c // n, i = c % n
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                    if val != expected:
                        return False
        return True
    
    # Try Strassen-like approach
    # For 2x4 * 4x4, we can view A as having 2 rows of 4 elements each
    # B as 4x4
    # 
    # Let's think about this differently. We can partition A into its two rows:
    # A = [a0; a1] where a0, a1 are 1x4 vectors
    # C = [a0*B; a1*B]
    #
    # Each row of C is a 1x4 vector which is the product of a 1x4 vector and a 4x4 matrix.
    # This is equivalent to 4 separate dot products for each row.
    #
    # For a 1x4 * 4x4 multiplication, we need 16 scalar multiplications naively.
    # For two such, we need 32.
    #
    # But we can share some computations. Let's think about using Strassen on blocks.
    #
    # Actually, let me try a direct search for rank-28 or lower decompositions.
    # This is hard, so let's try a few known tricks.
    
    # One approach: use the fact that we only have 2 output rows.
    # C[0, r] = sum_j A[0,j] * B[j,r]
    # C[1, r] = sum_j A[1,j] * B[j,r]
    
    # Let's try to find a rank-28 decomposition using algebraic manipulation.
    # 
    # Actually, let's just return the schoolbook for now and try to improve.
    
    start_time = time.time()
    deadline = 140  # seconds, with margin for the 150s kill
    
    # Try to find a better decomposition using random search / optimization
    # This is a hard problem, so let's try a few strategies
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Strategy 1: Try to merge some terms
    # For example, if we have terms (i,j,k) and (i',j',k') where they overlap in some way
    
    # Let's try a systematic approach: assume a certain structure and solve
    # for the entries.
    
    # For rank r, we need to find U, V, W such that:
    # sum_{l=0}^{r-1} U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 if (i,j,k) valid, 0 otherwise
    
    # This is a system of equations. Let's try small ranks.
    
    # Try rank 28
    # The system is: for each (i,j,k), sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1
    # And for invalid triples, the sum must be 0.
    
    # This is a non-linear system. Let's try to solve it numerically.
    
    import numpy as np
    
    # Define the tensor structure
    # Valid triples: (i, j, k) for i in 0..1, j in 0..3, k in 0..3
    # U index: i*m + j
    # V index: j*p + k  
    # W index: k*n + i
    
    # Let's set up the problem for a given rank
    # Variables: U[8, r], V[16, r], W[8, r]
    # Equations: for each valid (i,j,k): sum_l U[i*4+j, l] * V[j*4+k, l] * W[k*2+i, l] = 1
    # That's 32 equations.
    # Plus many more for invalid triples.
    
    # Actually, the invalid triples are most of the (8, 16, 8) = 1024 possible entries.
    # For each (a, b, c) in {0..7} x {0..15} x {0..7}, the equation is:
    # sum_l U[a, l] * V[b, l] * W[c, l] = T[a,b,c]
    # where T[a,b,c] = 1 if a=i*4+j, b=j*4+k, c=k*2+i for consistent i,j,k, else 0.
    
    # Let's count: for each (a, b, c), we can check if there exist i, j, k such that:
    # i = a // 4, j = a % 4
    # j' = b // 4, k = b % 4
    # k' = c // 2, i' = c % 2
    # And we need j == j', k == k', i == i'
    
    # Let's build the expected tensor
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(2):
        for j in range(4):
            for k in range(4):
                T[i * 4 + j, j * 4 + k, k * 2 + i] = 1
    
    # Now, for a given rank r, we want to find U, V, W such that:
    # np.einsum('al,bl,cl->abc', U, V, W) = T
    
    # This is a hard non-linear problem. Let's try some heuristics.
    
    # Heuristic 1: Start from schoolbook and try to merge terms
    # If two terms have the same U or V or W pattern, we might be able to combine them.
    
    # Heuristic 2: Use a randomized approach - initialize U, V, W randomly and 
    # use gradient descent or other optimization.
    
    # Let's try a simple approach: fix V and W to have a sparse structure and solve for U.
    # Or vice versa.
    
    # Actually, let's try a different approach. Since the problem is small, let's try
    # to find a rank-28 decomposition by solving a system of equations.
    
    # For rank 28, we have 8*28 + 16*28 + 8*28 = 224 + 448 + 224 = 896 variables.
    # And 1024 equations (one for each (a,b,c) entry of T).
    # The equations are cubic (products of three variables).
    
    # This is still hard. Let's try a more structured approach.
    
    # Key insight: The tensor has a special structure. It's a "matrix multiplication" tensor
    # with asymmetric dimensions. Let's exploit this.
    
    # A is 2x4, B is 4x4, C is 2x4.
    # We can write:
    # C[i, r] = sum_j A[i, j] B[j, r]
    
    # Let's define new variables to reduce the number of scalar multiplications.
    # 
    # Consider the 4 columns of B. Let B_j be the j-th column of B (a 4x1 vector... no wait,
    # B is 4x4, so B[j, r] is the element at row j, column r).
    
    # Let me re-think. B is 4x4. B[j, r] for j, r in 0..3.
    # A[i, j] for i in 0..1, j in 0..3.
    # C[i, r] = sum_{j=0}^{3} A[i, j] * B[j, r]
    
    # This is like computing 2 linear combinations of the rows of B, weighted by the entries of A.
    
    # Let's try to use Strassen-like ideas. Suppose we group the 4 columns of A into pairs.
    # A = [a0, a1, a2, a3] for row 0, [b0, b1, b2, b3] for row 1.
    
    # C[0, r] = a0*B[0,r] + a1*B[1,r] + a2*B[2,r] + a3*B[3,r]
    # C[1, r] = b0*B[0,r] + b1*B[1,r] + b2*B[2,r] + b3*B[3,r]
    
    # Let's define:
    # M0 = a0 + b0, M1 = a0 - b0, M2 = a1 + b1, M3 = a1 - b1
    # N0 = B[0,r] + B[1,r], N1 = B[0,r] - B[1,r], N2 = B[2,r] + B[3,r], N3 = B[2,r] - B[3,r]
    
    # Then:
    # C[0,r] = (M0+N0)/4 * (a0+b0+...) hmm this is getting complicated.
    
    # Let me try a cleaner approach. I'll use a known result: the rank of the 2x4 * 4x4 
    # matrix multiplication tensor.
    
    # Actually, let's just try to do a numerical search for a low-rank decomposition.
    
    # Define the objective: minimize ||T - T_approx||^2 where T_approx is the rank-r approximation.
    
    # For a fixed rank r, the problem is:
    # min_{U,V,W} sum_{a,b,c} (sum_l U[a,l]*V[b,l]*W[c,l] - T[a,b,c])^2
    
    # This is a non-convex problem. Let's try a simple gradient descent.
    
    def try_rank(r, max_iters=1000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        np.random.seed(seed)
        U = np.random.randn(8, r) * 0.1
        V = np.random.randn(16, r) * 0.1
        W = np.random.randn(8, r) * 0.1
        
        # Compute the tensor product
        def compute_tensor(U, V, W):
            return np.einsum('al,bl,cl->abc', U, V, W)
        
        def loss(U, V, W):
            T_approx = compute_tensor(U, V, W)
            return np.sum((T_approx - T)**2)
        
        # Gradient descent
        for it in range(max_iters):
            if it % 100 == 0 and time.time() - start_time > deadline:
                return None
            
            T_approx = compute_tensor(U, V, W)
            diff = T_approx - T
            
            # Gradient with respect to U
            # dL/dU[a,l] = 2 * sum_{b,c} diff[a,b,c] * V[b,l] * W[c,l]
            grad_U = 2 * np.einsum('abc,bl,cl->al', diff, V, W)
            grad_V = 2 * np.einsum('abc,al,cl->bl', diff, U, W)
            grad_W = 2 * np.einsum('abc,al,bl->cl', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we found a good solution
        T_approx = compute_tensor(U, V, W)
        err = np.max(np.abs(T_approx - T))
        if err < 1e-6:
            return U, V, W
        return None
    
    # Try a few ranks
    for r in range(24, 33):
        if time.time() - start_time > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with rational-looking numbers
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            # Round to nearest 0.5 if close
            for lst in [U_list, V_list, W_list]:
                for row in lst:
                    for i in range(len(row)):
                        # Try to find a nice fraction
                        val = row[i]
                        # Check if close to an integer or half-integer
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < 1e-6:
                            row[i] = float(nearest_int)
                        elif abs(val - nearest_half) < 1e-6:
                            row[i] = float(nearest_half)
            return U_list, V_list, W_list
    
    # If all else fails, return schoolbook
    return best_U, best_V, best_W

# EVOLVE-BLOCK-END
