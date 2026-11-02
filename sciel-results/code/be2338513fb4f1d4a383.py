"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Condition: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    And for the "wrong" combinations (which correspond to the tensor being exactly the
    multiplication tensor), we need the full tensor equality.
    
    The matrix multiplication tensor M_{n,m,p} has entries:
    M[i, j, k] = 1 for all valid (i, j, k) in the sense that the trilinear form is
    <C, A, B> = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    
    A rank-r decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    But wait - that's only n*m*p equations. The full tensor has n*m*p entries, each of which
    should be 1. Let me reconsider.
    
    Actually, the standard definition: the matrix multiplication tensor T of format (n,m,p)
    is defined by T[i,j,k] = 1 for all valid indices, meaning:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    where we need T[i,j,k] = 1 for all (i,j,k).
    
    But that would mean the tensor is all-ones, which is not the multiplication tensor.
    
    Let me re-read the problem. The output factor follows W[k*n+i, r].
    
    The correct condition is:
    For the multiplication C = AB where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor form: we want coefficients such that
    C[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c]) * W[k*n+i, c]
    
    Expanding: C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c])
    
    For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    sum_c W[k*n+i, c] * U[i*m+j', c] * V[j'*p+k, c] = delta_{j', j} ... 
    
    No wait. Let's be more careful.
    
    sum_c W[k*n+i, c] * (sum_{j1} A[i,j1] U[i*m+j1, c]) * (sum_{j2} B[j2,k] V[j2*p+k, c])
    = sum_c W[k*n+i, c] * sum_{j1,j2} A[i,j1] B[j2,k] U[i*m+j1, c] V[j2*p+k, c]
    = sum_{j1,j2} A[i,j1] B[j2,k] * sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c]
    
    We need this to equal sum_j A[i,j] B[j,k], so:
    sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c] = delta_{j1, j2}
    
    So the condition is: for all i, k, j1, j2:
    sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1, j2}
    
    This is n*p*m*m = n*m^2*p equations.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = np.dot(U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try to find decompositions with rank < 54
    # Use a structured approach: try Kronecker-like decompositions
    
    # Strategy: The 2x3x9 multiplication can be seen as 2x3x3 repeated 3 times (since 9=3*3)
    # If we have a rank-r decomposition for 2x3x3, we can get rank 3r for 2x3x9
    # via block diagonal structure.
    # Known: rank(M_{2,3,3}) <= 9 (schoolbook), but can it be lower?
    # Actually, for small matrices, let's try to find good decompositions.
    
    # Let's try a different approach: use the fact that 9 = 3*3 and try
    # to find a rank-r decomposition where r < 54.
    
    # Try structured decompositions based on Kronecker products
    # M_{2,3,9} can be decomposed using M_{2,1,3} and M_{1,3,3}
    # M_{2,1,3}: rank 6 (schoolbook), but maybe lower?
    # M_{1,3,3}: rank 9 (schoolbook)
    
    # Actually, let's try a direct approach with random initialization and verification.
    # For each rank r from 27 to 53, try to find a valid decomposition.
    
    # Use a smarter initialization: based on the schoolbook structure but with
    # some columns merged.
    
    # First, let's try a simple structured approach:
    # For 2x3x9, we can write 9 = 3*3 and use:
    # C[i,k] where k = k1*3 + k2
    # C[i, k1*3+k2] = sum_j A[i,j] B[j, k1*3+k2]
    
    # This doesn't directly help with tensor rank.
    
    # Let's try random search with a few restarts for lower ranks.
    # Given the time constraint, we'll try a limited number of attempts.
    
    np.random.seed(42)
    
    for target_rank in range(27, min(54, rank_sb)):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations
        found = False
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with a structured pattern
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Try a Kronecker-like structure
            # For each column c, assign it to a "block"
            for c in range(target_rank):
                # Randomly assign non-zero entries
                # Use a sparse pattern
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    # Convert to lists for the return interface
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
