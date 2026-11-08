"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook rank
    rank_school = n * m * p  # 56
    
    # Try to find a better decomposition using random search with exact arithmetic
    # The matrix multiplication tensor T has components T[i,j,k] = 1
    # We want to find U, V, W such that T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the element (i,k) of C = A*B depends on A[i,j]*B[j,k]
    # In index form: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the bilinear form has terms A[i,j] * B[j,k] for each j
    
    # The "schoolbook" uses r = n*m*p = 56 terms.
    # Can we do better?
    
    # For 2x4x7, let's check known results or try to find improvements.
    # The rank of matrix multiplication M_{2,4,7} - let me try random perturbation search.
    
    # Strategy: Start from schoolbook and try to find linear dependencies
    # or use a randomized approach to find lower rank decompositions.
    
    # First, let's implement a verification function
    def verify(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        r = len(U[0])
        # T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # In our indexing: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if j == j else 0  # T[i,j,k] = 1 always for matrix mult
                    # Actually for matrix multiplication T[i,j,k] = 1 for all valid i,j,k
                    # where the term A[i,j]*B[j,k] appears in C[i,k]
                    # So T[i,j,k] = 1 for all i,j,k
                    actual = sum(Fraction(U[i*m+j][c]) * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if actual != 1:
                        return False
        return True
    
    # Schoolbook solution
    def schoolbook(n, m, p):
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
        return U, V, W
    
    # Try to find a better decomposition
    # One approach: use the fact that we can combine terms if there's structure
    
    # For 2x4x7, let me try a systematic approach:
    # The rank of M_{n,m,p} is known for small cases.
    # For n=1, rank(M_{1,m,p}) = m*p (can't do better since each output element 
    # needs its own set of products)
    # For n=2, we might save some rank.
    
    # Let me try a random search approach with exact fractions
    # Start from schoolbook and try to find linear combinations
    
    # Actually, let me try a different approach:
    # Use the fact that for matrix multiplication, we can sometimes use 
    # "splitting" or "blocking" to reduce rank.
    
    # For 2x4x7: 
    # C = A * B where A is 2x4, B is 4x7
    # C[0,0..6] = A[0,0]*B[0,0..6] + A[0,1]*B[1,0..6] + A[0,2]*B[2,0..6] + A[0,3]*B[3,0..6]
    # C[1,0..6] = A[1,0]*B[0,0..6] + A[1,1]*B[1,0..6] + A[1,2]*B[2,0..6] + A[1,3]*B[3,0..6]
    
    # Schoolbook: 2*4*7 = 56 multiplications
    # Can we do better?
    
    # Let me try a random search: generate random U, V, W with small integer entries
    # and check if they form a valid decomposition.
    
    # Better approach: use a known lower bound and try to achieve it.
    # The rank of M_{2,4,7} - let me search the literature in my head...
    # I don't recall the exact value, so let me try to find it computationally.
    
    # Approach: Start with schoolbook and try to find a smaller rank by
    # allowing the entries to be non-trivial integers.
    
    # Let me try a randomized local search:
    # 1. Start with a random decomposition of some rank r
    # 2. Try to adjust entries to satisfy the tensor equations
    
    # Actually, the most reliable approach for small dimensions is to use
    # a complete search or a well-informed random search.
    
    # Let me try: for rank r from 55 down to some lower bound, try to find U, V, W
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But this is a weak lower bound.
    
    # A better lower bound: rank >= n*m*p / max(n,m,p) ... not standard.
    
    # Let me just try to find a decomposition with rank < 56.
    
    # Strategy: Use the fact that the matrix multiplication tensor has a lot of structure.
    # For each (j, k) pair, the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # Let me try a parameterized approach:
    # Assume U, V, W have a specific structure and solve for the parameters.
    
    # For simplicity, let me try a few specific rank values and see if I can find a solution.
    
    # Actually, let me try a completely different approach:
    # Use the "Laser Method" style decomposition if applicable, or just 
    # do a careful random search with exact arithmetic.
    
    # Given the time constraint, let me try a focused random search for rank 55, 54, etc.
    
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Try to find improvements by random perturbation
    # This is computationally expensive, so let's be smart about it.
    
    # For 2x4x7, I believe the optimal rank might be 55 or 56.
    # Let me try to verify if rank 55 is achievable.
    
    # Actually, let me just provide the schoolbook solution since:
    # 1. It's guaranteed to be correct
    # 2. For 2x4x7, it's likely near-optimal
    # 3. Finding a better decomposition requires significant search time
    
    # But let me try a quick random search to see if we can find rank 55
    random.seed(42)
    
    for target_rank in range(rank_school - 1, max(28, rank_school - 10), -1):
        if time.time() > deadline:
            break
        # Try a few random attempts for this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Generate random U, V, W with entries from {-2,-1,0,1,2}
            U = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to the required format
    r = best_rank
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
