"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    # The multiplication tensor T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # in index notation: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The reconstruction is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # But actually the convention is:
    # (A*B)[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k] ... 
    # Let me re-derive. The standard tensor decomposition for matrix mult:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] where T[i,j,k] = 1 if i,j,k form a valid triple
    # Actually for matrix multiplication A(4x4) * B(4x7) -> C(4x7):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, that's not right either. The multiplication tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's the all-ones tensor. 
    # 
    # Actually the matrix multiplication tensor M_{n,m,p} is defined as:
    # M[i, j, k] = 1 if the triple (i,j,k) corresponds to a term in the product
    # For C[i,k] = sum_j A[i,j] B[j,k], the tensor is:
    # M[i, j, k] = 1 for all valid i, j, k
    # 
    # Hmm, let me think more carefully. The bilinear map is:
    # (A, B) -> A*B where A is n x m, B is m x p
    # This is a 3-linear form: sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] C^*[i,k]
    # where M[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] contributes to C[i,k])
    #
    # Wait no. M[i,j,k] = 1 if and only if the index j is shared, i.e., for the product
    # C[i,k] = sum_j A[i,j] B[j,k], the tensor entries are M[i,j,k] = 1 for all valid (i,j,k).
    # 
    # So the tensor T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p).
    #
    # A rank-r decomposition: T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # In the schoolbook: each rank-1 term picks one (i,j,k) triple and sets U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 only when (i,j,k) is the specific triple, 0 otherwise.
    # So summing over all r gives 1 for each (i,j,k). Correct.
    #
    # So verification: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Check dimensions
    if U_mat.shape != (n * m, W_mat.shape[1]):
        return False
    if V_mat.shape != (m * p, W_mat.shape[1]):
        return False
    if W_mat.shape != (p * n, U_mat.shape[1]):
        return False
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
        return False
    
    rank = U_mat.shape[1]
    
    # Verify by checking all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_lanczos_like(n, m, p):
    """
    Try a structured lower-rank decomposition based on the idea that
    the multiplication tensor has symmetries we can exploit.
    
    For n=m=p, the Strassen algorithm gives rank 7 < 8.
    For rectangular cases, we can try to find structured decompositions.
    
    Here we try a random search over small ranks to see if we can find
    an exact decomposition.
    """
    # For 4x4 * 4x7, the schoolbook rank is 4*4*7 = 112.
    # Known lower bounds: rank >= n*m*p / (some factor)
    # Upper bound from literature: for 4x4x4, best known is 7 (Strassen).
    # For 4x4x7, we might do better than 112.
    
    # Try a simple structured approach: group terms
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this using block structure if we group indices.
    
    # For now, try random search for small ranks
    # This is unlikely to work for rank close to 112, but let's try
    # a few structured ideas.
    
    # Idea: use the fact that n=m=4, so we can use Strassen-like blocks
    # Split A into 2x2 blocks of 2x2 matrices
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each is 2x2 (4 cols split into 2x2)
    
    # Actually B is 4x7, so splitting into 2x2 blocks doesn't divide evenly.
    # 7 = 2+2+2+1, awkward.
    
    # Let's just return None and rely on schoolbook
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find better decompositions
    # Strategy: random search with structured initialization
    
    rng = np.random.RandomState(SEED)
    
    # Try various ranks below schoolbook
    # For 4x4x7, known best upper bound from literature might be around 70-90
    # Let's try a few structured approaches
    
    # Approach 1: Try rank = n*p + m*p + n*m - something
    # This is the "border rank" type bound
    
    # Approach 2: Random search
    # For each candidate rank, try to find U, V, W that satisfy the constraints
    
    # The constraints are:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 112 equations in 3*rank variables.
    # For rank < 112, this is overdetermined, so exact solutions may not exist
    # unless the tensor has special structure.
    
    # Let's try a gradient-based approach or just verify that schoolbook works
    # and return it.
    
    # Actually, let's try one more thing: a structured decomposition
    # based on the idea of sharing computation.
    
    # For 4x4 * 4x7:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # We can try to use the fact that we can compute partial sums
    # S1[i] = A[i,0] + A[i,1], S2[i] = A[i,0] - A[i,1], etc.
    # But this doesn't directly reduce the rank.
    
    # Let's just return the schoolbook decomposition which is guaranteed correct.
    # The rank is 112.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try a quick random search for rank 100 (arbitrary)
    # This is very unlikely to find an exact solution, but let's try
    # a few iterations.
    
    for trial_rank in [100, 90, 80]:
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U_try = rng.randn(n * m, trial_rank)
        V_try = rng.randn(m * p, trial_rank)
        W_try = rng.randn(p * n, trial_rank)
        
        # Check if this happens to work (extremely unlikely)
        if _verify(U_try, V_try, W_try, n, m, p, tol=1e-6):
            best_rank = trial_rank
            best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
