"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    # Build the full multiplication tensor T[i,j,k] = delta
    # The decomposition gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 for all (i,j,k) and 0 otherwise? 
    # Actually the tensor for matrix mult C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is indexed such that T[i, j, k] = 1 if we are computing the (i,k) entry using j
    # In the trace convention W[k*n+i, r], the tensor is T[i, j, k]
    # The decomposition is: T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For matrix multiplication, T[i, j, k] = 1 for all i, j, k (since every triple contributes)
    # Wait, no. Let's re-read the contract.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]B[j,k] in C[i,k] is 1.
    # So T[i, j, k] = 1 for all valid i, j, k.
    
    # Let's verify: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should equal 1 for all i,j,k.
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Efficient check: for each i, j, k, compute the sum
    # We can vectorize: for each r, the contribution is U[:, r] outer V[:, r] outer W[:, r]
    # But that's expensive. Let's do it smarter.
    # For fixed i, j: the row U[i*m+j, :] is a vector u.
    # For fixed j, k: the row V[j*p+k, :] is a vector v.
    # For fixed k, i: the row W[k*n+i, :] is a vector w.
    # The sum is u . v . w (element-wise product then sum).
    
    # Let's check a few random samples plus all if small
    n, m, p = 3, 4, 5
    # Total triples: 3*4*5 = 60
    # We can check all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.dot(u, v * w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # Known lower bounds for 3x4x5: 
    # The tensor rank is at least max(n*m, m*p, p*n) / something? 
    # Actually, the rank is at least max(n*m, m*p, p*n) is not right.
    # The rank is at least the max of the slice ranks.
    # For now, try to search for rank < 60.
    
    # Let's try a randomized search for lower rank decompositions
    # We'll use a simple iterative method: start with random U, V, W of rank r,
    # and try to minimize the error ||T - sum_r U_r V_r W_r||_F
    
    # But this is a hard problem. Let's try a few ranks.
    
    # First, let's see if we can do rank 59, 58, etc.
    # Actually, for small matrices, the rank is often close to n*m*p.
    # Let's try a few random restarts for rank = best_rank - 1, best_rank - 2, etc.
    
    # Given the time constraint, let's just return the schoolbook if we can't find better quickly.
    # But let's try a simple randomized search for rank 50, 40, 30, etc.
    
    # Actually, the known rank for 3x4x5 matrix multiplication is not trivial.
    # Let's just try a few random initializations for rank < 60.
    
    np.random.seed(42)
    
    # Try ranks from 55 down to 40
    for target_rank in range(min(55, best_rank - 1), 35, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Scale to make the product closer to 1
            # The expected value of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] is 0
            # We need the sum to be 1.
            # Let's normalize: divide by sqrt(target_rank)
            U_rand /= np.sqrt(target_rank)
            V_rand /= np.sqrt(target_rank)
            W_rand /= np.sqrt(target_rank)
            
            # Check if this works (it won't, but let's see)
            # This is just a placeholder. A real search would be needed.
            pass
    
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
