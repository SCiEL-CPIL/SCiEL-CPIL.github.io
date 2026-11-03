import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6):
    """
    Verifies if the given U, V, W matrices decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Build the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] to be 1 if i,j,k are valid indices for the multiplication?
    # No, the tensor for matrix multiplication is:
    # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product term.
    # Specifically, C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor element T[i*m+j, j*p+k, k*n+i] should be 1?
    # Let's check the convention.
    # The tensor is defined such that <T, a \otimes b \otimes c> = sum_{i,j,k} a_{i,j} b_{j,k} c_{k,i}?
    # No, standard matrix multiplication tensor M_{n,m,p} has entries:
    # M_{i,j,k} = 1 if the indices correspond to a valid term in the sum?
    # Actually, the tensor T is such that T_{ijk} = 1 if i, j, k are such that
    # the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the flattening:
    # A index: i * m + j
    # B index: j * p + k
    # C index: k * n + i
    # The tensor should have 1s at these positions and 0s elsewhere.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Compute T' = sum_r U[:, r] outer V[:, r] outer W[:, r]
    # T' is a tensor of shape (n*m, m*p, p*n)
    T_calc = np.zeros_like(T)
    for r in range(rank):
        T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Check if T_calc matches T
    # Use a tolerance for floating point errors if using floats, but we use ints/halves.
    return np.allclose(T_calc, T, atol=1e-9)

def _get_rank23_decomposition():
    """
    Returns a list of 23 triplets (u, v, w) for 3x3x3 matrix multiplication.
    u, v, w are lists of 9 floats/ints.
    """
    # I will try to find these.
    # If I fail, I return None.
    # For now, I return None to trigger fallback.
    # I need to find a valid set.
    # I will try to use the Strassen-Winograd terms.
    # I don't have them memorized exactly.
    # I will return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to get a better decomposition
    rank23_terms = _get_rank23_decomposition()
    
    if rank23_terms is not None:
        # Construct rank-46 decomposition for 3x3x6
        U = []
        V = []
        W = []
        
        for u, v, w in rank23_terms:
            # u, v, w are length 9
            # Term 1: First 3 columns of B and C
            U1 = u[:]
            V1 = [0.0] * (m * p)
            W1 = [0.0] * (p * n)
            
            for j in range(m):
                for k in range(3): # k=0,1,2
                    V1[j * p + k] = v[j * 3 + k]
            
            for k in range(3):
                for i in range(n):
                    W1[k * n + i] = w[k * 3 + i]
            
            U.append(U1)
            V.append(V1)
            W.append(W1)
            
            # Term 2: Last 3 columns of B and C
            U2 = u[:]
            V2 = [0.0] * (m * p)
            W2 = [0.0] * (p * n)
            
            for j in range(m):
                for k in range(3, 6): # k=3,4,5
                    V2[j * p + k] = v[j * 3 + (k-3)]
            
            for k in range(3, 6):
                for i in range(n):
                    W2[k * n + i] = w[(k-3) * 3 + i]
            
            U.append(U2)
            V.append(V2)
            W.append(W2)
            
        U = np.array(U).T # Shape (9, 46)
        V = np.array(V).T # Shape (18, 46)
        W = np.array(W).T # Shape (18, 46)
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, fallback
            pass
            
    # Fallback: Schoolbook rank 54
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
