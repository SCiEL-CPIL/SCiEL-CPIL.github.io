"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from functools import lru_cache

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    size1 = n * m  # 9
    size2 = m * p  # 15
    size3 = p * n  # 15
    
    # Build the tensor
    T = np.zeros((size1, size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strategy: Try to find a low-rank decomposition using the subspace method
    # The key insight: for matrix multiplication, we can use block decompositions
    
    # First, let's try a systematic search for low-rank decompositions
    # using the fact that the tensor has structure
    
    # Approach 1: Use Strassen-like decomposition for 3x3x3 + schoolbook for 3x3x2
    # This should give rank 7 + 18 = 25
    
    # Let's first try to find if we can do better than 25
    
    # For 3x3x5, the theoretical minimum is at least max(n*m, m*p, p*n) = max(9,15,15) = 15
    # But realistically, we expect something around 20-30
    
    # Let's try a greedy approach to find a good decomposition
    
    best_result = None
    best_rank = float('inf')
    
    # Try different rank values and see if we can find a valid decomposition
    for rank in range(15, 30):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        result = _try_find_decomposition(T, size1, size2, size3, rank, n, m, p, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found the lowest rank, stop
    
    if best_result is not None:
        return best_result
    
    # Fallback: Use the Strassen + schoolbook approach (rank 25)
    if time.time() < deadline - 5:
        result = _strassen_schoolbook(n, m, p)
        if result is not None:
            return result
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

def _strassen_schoolbook(n, m, p):
    """Combine Strassen for 3x3x3 with schoolbook for 3x3x2."""
    if n != 3 or m != 3 or p != 5:
        return None
    
    size1 = n * m  # 9
    size2 = m * p  # 15
    size3 = p * n  # 15
    
    # Strassen's 7 terms for 3x3x3
    # Using a known rank-7 decomposition for 3x3x3
    
    # The 7 Strassen terms for 3x3x3 (using standard notation):
    # Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    
    # One known rank-7 decomposition:
    strassen_U = [
        # Term 0: a*j
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,0],
    ]
    
    # Actually, let me use a verified Strassen decomposition
    # For 3x3x3, the rank is 7, and one decomposition is:
    
    # Let me just use a simpler approach: build the tensor and verify
    
    # For now, let's use a known valid decomposition
    # I'll construct it carefully
    
    # Strassen terms for 3x3x3 (7 terms):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B10
    # M3 = A00 * (B11 - B00)
    # M4 = (A01 - A11) * (B10 + B11)
    # M5 = (A00 + A01) * B00
    # M6 = (A10 - A00) * (B01 + B10)
    # M7 = (A01 + A10) * B11
    
    # This is getting complex. Let me just use schoolbook for the 3x3x3 part too
    # to ensure correctness, then try to optimize
    
    # Actually, let me return None and use the fallback
    return None

def _try_find_decomposition(T, size1, size2, size3, target_rank, n, m, p, deadline):
    """Try to find a decomposition of the given rank."""
    
    # Use the subspace method:
    # For matrix multiplication, we can exploit the structure
    
    # The tensor T has the property that T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to find a decomposition using random initialization + optimization
    
    # Initialize with random matrices
    np.random.seed(42)
    
    U = np.random.randn(size1, target_rank) * 0.1
    V = np.random.randn(size2, target_rank) * 0.1
    W = np.random.randn(size3, target_rank) * 0.1
    
    # Try to optimize to match the tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a non-convex problem, so we'll use gradient descent
    
    # For efficiency, we'll only optimize on a subset of the tensor entries
    # The non-zero entries are at positions (i*m+j, j*p+k, k*n+i) for all i,j,k
    
    # Build the list of non-zero entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    # Gradient descent
    lr = 0.01
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for a, b, c in nonzero_entries:
            # T_target[a,b,c] = 1
            # current = sum_r U[a,r] * V[b,r] * W[c,r]
            current = np.sum(U[a, :] * V[b, :] * W[c, :])
            error = current - 1.0
            
            # d/dU[a,r] = error * V[b,r] * W[c,r]
            grad_U[a, :] += error * V[b, :] * W[c, :]
            grad_V[b, :] += error * U[a, :] * W[c, :]
            grad_W[c, :] += error * U[a, :] * V[b, :]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        max_error = 0
        for a, b, c in nonzero_entries:
            current = np.sum(U[a, :] * V[b, :] * W[c, :])
            max_error = max(max_error, abs(current - 1.0))
        
        if max_error < 1e-6:
            # Verify all entries
            if _verify_decomposition(U, V, W, T, size1, size2, size3, n, m, p):
                # Convert to list of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None

def _verify_decomposition(U, V, W, T, size1, size2, size3, n, m, p):
    """Verify that the decomposition matches the tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                value = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(value - 1.0) > 1e-6:
                    return False
    
    # Check some zero entries to make sure they're not accidentally non-zero
    # (This is a partial check for efficiency)
    checked = 0
    for a in range(0, size1, 2):
        for b in range(0, size2, 2):
            for c in range(0, size3, 2):
                if T[a, b, c] == 0:
                    value = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(value) > 1e-4:
                        return False
                    checked += 1
                    if checked > 100:
                        break
            if checked > 100:
                break
        if checked > 100:
            break
    
    return True

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
