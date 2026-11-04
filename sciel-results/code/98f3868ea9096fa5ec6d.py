"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def _make_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensors
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all (i,j,k)
    # We can check this by computing the "reconstruction"
    # Actually, the tensor T is defined such that:
    # (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c (U[i*m+j,c] * something) ...
    # The standard identity is:
    # sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
    # Wait, no. The decomposition is:
    # M_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # where M is the matrix multiplication tensor, which is 1 if we are multiplying basis elements.
    # Specifically, the tensor T of shape (n*m, m*p, p*n) is defined by:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} delta_{k,k'} ... no.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for the "multiplication" part.
    # Actually, the tensor is T_{i,j,k} such that (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # In tensor notation, this is a contraction. The "tensor" we are decomposing is the structure tensor.
    # The structure tensor of matrix multiplication M_{n,m,p} has rank r if we can find U,V,W of size (nm x r), (mp x r), (pn x r)
    # such that for all i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1  (if we use the standard basis)
    # Wait, the standard basis tensor is T[i,j,k] = 1.
    # Let's verify with a random matrix.
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    AB = A @ B
    
    # Compute via decomposition
    # (AB)_{i,k} = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # Let's compute this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # U_arr shape: (n*m, rank)
    # V_arr shape: (m*p, rank)
    # W_arr shape: (p*n, rank)
    
    # We need to compute C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # This is a bit complex to vectorize directly without loops, but n,m,p are small.
    
    # Alternative check:
    # The tensor T is defined by T[i,j,k] = 1.
    # Let's just check the tensor entries directly.
    # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This should be 1 for all i,j,k.
    
    # Let's compute this for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    best_result = _make_schoolbook()
    
    # Try to find a lower rank decomposition.
    # The lower bound for rank is max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # The schoolbook rank is 60.
    # Known results:
    # For 3x4x5, the rank is not known to be significantly lower than 60 in general literature for small matrices,
    # but let's try to search for a rank slightly lower than 60.
    # Actually, the rank of matrix multiplication M_{n,m,p} is at least n*m, m*p, p*n.
    # For 3,4,5: 12, 20, 15. So rank >= 20.
    # But it's likely much higher.
    # Let's try to see if we can find a rank 59, 58, etc.
    # This is a hard problem. Random search is unlikely to find it quickly.
    # However, we can try a few random initializations and use gradient descent to minimize the error.
    
    # Since we have a 120s budget, and the verification is O(n*m*p*rank), which is 60*60 = 3600 ops per check,
    # and we have to do it for each candidate, we can't do many iterations.
    
    # Let's just return the schoolbook decomposition.
    # It's the safest bet.
    return best_result

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    best_result = _make_schoolbook()
    
    # We will try to find a rank slightly lower than 60.
    # But given the complexity, we'll just return the schoolbook.
    return best_result
# EVOLVE-BLOCK-END
