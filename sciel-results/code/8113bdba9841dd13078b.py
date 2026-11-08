import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We'll try to find a lower rank decomposition
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] = delta for the right structure
    
    # Actually, the constraint is:
    # For the decomposition to be exact, we need the bilinear map to reproduce matrix multiplication.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    # This should equal sum_j A[i,j] B[j,k] for all A, B.
    
    # This means: sum_c U[i*m+j, c] * V[j'*p+k, c] should have the right structure.
    # Specifically, when we expand: sum_c sum_j sum_{j',k'} U[i*m+j,c] V[j'*p+k',c] A[i,j] B[j',k']
    # = sum_{j,j',k'} (sum_c U[i*m+j,c] V[j'*p+k',c]) A[i,j] B[j',k']
    # This should equal sum_j A[i,j] B[j,k]
    # So we need: sum_c U[i*m+j,c] V[j'*p+k',c] = delta_{j,j'} * delta_{k,k'}... 
    # Wait, that's not right. We need it to equal delta_{j,j'} when k'=k, summed over k'.
    
    # Actually: the coefficient of A[i,j] B[j',k'] in the LHS is sum_c U[i*m+j,c] V[j'*p+k',c]
    # The coefficient in the RHS (sum_j A[i,j] B[j,k]) is 1 if j=j' and k'=k, else 0.
    # But wait, the RHS has k as a free index, so for a fixed output (i,k), the coefficient of A[i,j]B[j',k'] is 1 if j=j' and k'=k.
    
    # So the constraint is: for all i, j, j', k, k':
    # sum_c U[i*m+j, c] V[j'*p+k', c] = 1 if j==j' and k==k', else 0
    # But this must hold for all i... which means U and V are independent of i? That can't be right.
    
    # Let me reconsider. The output W[k*n+i, c] is also involved.
    # The full decomposition: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * ... no.
    
    # I think the correct formulation is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_{j'} V[j'*p+k, c] * B[j',k])
    # No wait, V should also depend on k.
    
    # Let me look at the seed program more carefully:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So C[i,k] = sum_c W[k*n+i,c] * ... 
    # Actually I think the product is: C[i,k] = sum_c (U^T A_flat)_{c} * (V^T B_flat)_{c} * W[k*n+i, c]? No.
    
    # From the seed: each term contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to... 
    # I think the formula is: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_{j'} V[j'*p+k,c] B[j',k])
    # But in the schoolbook, for each (i,j,k) triple, we have one term where:
    # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # So that term contributes: 1 * A[i,j] * B[j,k] * 1 = A[i,j]*B[j,k]
    # And summing over all (i,j,k) gives sum_j A[i,j]B[j,k] for each (i,k). Correct!
    
    # So the general formula: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_{j'} V[j'*p+k,c] B[j',k])
    
    # For this to equal sum_j A[i,j]B[j,k] for all A,B:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,j,j',k
    
    # This is a system of equations. Let's set up the search.
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Try ranks from low to high
    # Known lower bound for 3x3*3x6: at least 3*3*6/... let's just search
    
    # For a structured approach, try rank = n*p + m*p + n*m - something
    # Or use the fact that we can share across k dimension
    
    # Try a simple structured decomposition first
    # For each (i,j), we need to compute sum over k of A[i,j]*B[j,k]
    # We can write: for each (i,j), C[:,k] += A[i,j] * B[j,:]
    # This is outer product: A[i,j] * B[j,:]
    # So we need n*m outer products, each of rank 1 in the k dimension
    # Total rank: n*m = 9? No, that's not right because we need to sum over j.
    
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (A[i,j] * B[j,k])
    # Each term A[i,j]*B[j,k] is a product of a scalar (from A) and a scalar (from B)
    # So we need n*m*p = 54 terms in general.
    
    # But we can potentially combine terms. For example, if we fix i and vary j,k:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a matrix-vector product: row i of A times matrix B.
    # The rank of this operation as a bilinear map is m*p = 18 for each i.
    # Total: n * m * p = 54.
    
    # To do better, we need to share computation across i.
    # For example, if two rows of A are linearly related, but they're arbitrary.
    
    # Let's try a random search for low rank
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_{j'} V[j'*p+k,c] B[j',k])
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,j,j',k
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1 if j == jp else 0
                        actual = sum(W[k*n+i][c] * U[i*m+j][c] * V[jp*p+k][c] for c in range(rank))
                        if abs(actual - expected) > 1e-9:
                            return False
        return True
    
    # Try structured decompositions
    # Idea: use the fact that for fixed i, C[i,:] = A[i,:] @ B
    # This is a vector-matrix product of dimension m x (m*p)
    # Rank of vector-matrix product (m-vector times m*p matrix) is m*p
    # But we can potentially share across i.
    
    # Let's try: rank = n * m * p / some_factor
    # Try various ranks
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # First, try the schoolbook as fallback
    U_school = [[0]*rank_schoolbook for _ in range(n*m)]
    V_school = [[0]*rank_schoolbook for _ in range(m*p)]
    W_school = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find better decompositions using random initialization + local search
    # or structured approaches
    
    # Structured approach: exploit the rectangular shape
    # For 3x3 * 3x6, we can think of it as 3 rows of A each multiplying B (3x6)
    # C[i,:] = A[i,:] @ B for i=0,1,2
    
    # Each C[i,:] is a 1x6 vector = (3-vector) @ (3x6 matrix)
    # The bilinear complexity of multiplying a 1xm vector by mxp matrix is m*p
    # But we can share: if we compute all three at once.
    
    # Try: use the same "basis" for all i, but with different weights
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (sum_c U[i*m+j,c] A[i,j]) ... this is getting circular.
    
    # Let me try a direct random search for small ranks
    
    import random
    random.seed(42)
    
    for target_rank in range(20, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to fractions for exactness if needed
    # For now, return the best found
    
    return {"U": best_U, "V": best_V, "W": best_W}
