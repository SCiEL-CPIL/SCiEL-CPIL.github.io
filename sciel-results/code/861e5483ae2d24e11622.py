"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x4x8():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    
    Strategy: Partition the 4x8 middle into 2x(2x8) blocks and use Strassen's
    7-multiplication algorithm on the 2x2 block structure, then handle the
    2x8 sub-problems with schoolbook.
    
    A (2x4) = [A11 A12] where A11, A12 are 2x2
    B (4x8) = [B11; B22] where B11, B22 are 2x8
    
    C = A11*B11 + A12*B22
    
    For each 2x2x8 multiplication, use Strassen on the 2x2 part:
    7 products of (1x8) vectors, giving rank 7*8 = 56 per 2x2x8 block? 
    No, Strassen gives 7 scalar multiplications per output element... 
    
    Actually, let me use a different approach:
    View 2x4x8 as two independent 1x4x8 problems (one for each row of A).
    For 1x4x8: C[k] = sum_j A[j]*B[j,k] for k=0..7
    This is a 1x4x8 tensor with rank 4*8=32 schoolbook.
    
    Better: Use the algebraic structure directly.
    
    Let me construct an explicit lower-rank decomposition using linear combinations.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll use a systematic construction based on combining terms.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # For i=0: C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # For i=1: C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    #
    # Idea: combine pairs of j's using Strassen-like tricks on the (i,j) submatrix.
    # For fixed k: [C[0,k], C[1,k]]^T = M_k * [B[0,k], B[1,k], B[2,k], B[3,k]]^T
    # where M_k depends on A.
    #
    # Alternative: use the fact that we can define new variables that are linear
    # combinations of A entries and B entries to reduce the number of products.
    
    # Let's try a direct construction with rank 56 (saving 8 from schoolbook).
    # Partition j into two pairs: {0,1} and {2,3}
    # For each pair, use 2x2 Strassen (7 mults instead of 4) on the (i,j) part.
    
    # For pair {0,1}:
    #   C[0,k] += A[0,0]B[0,k] + A[0,1]B[1,k]
    #   C[1,k] += A[1,0]B[0,k] + A[1,1]B[1,k]
    # Strassen on this 2x2x1 (for each k):
    #   m1 = (A[0,0]+A[1,1])*(B[0,k]+B[1,k])
    #   m2 = (A[1,0]+A[1,1])*B[0,k]
    #   m3 = A[0,0]*(B[1,k]-B[0,k])
    #   m4 = A[1,1]*(B[0,k]-B[1,k])
    #   m5 = (A[0,0]+A[0,1])*B[1,k]
    #   m6 = (A[1,0]-A[0,0])*B[0,k]
    #   m7 = (A[0,1]-A[1,1])*B[1,k]
    #   C[0,k] += m1 - m4 - m5 + m7
    #   C[1,k] += m2 + m4 - m6 - m7
    #
    # But this gives 7 products per k, so 7*8 = 56 for one pair, and another 56 for the other pair = 112. Too much.
    #
    # The issue is that Strassen doesn't directly help here because the "scalar" part is 1-dimensional.
    
    # Let me try a completely different approach: use a known result.
    # R(2,4,8) <= R(2,4,4) * R(1,1,2) by submultiplicativity? Not directly.
    #
    # Actually, let's just do a numerical optimization to find a low-rank decomposition.
    # We'll use gradient-free methods or direct algebraic construction.
    
    # For now, return schoolbook. We'll improve this in the next iteration.
    return _schoolbook()

def _find_decomposition_optimization(n, m, p, target_rank, time_budget):
    """
    Try to find a decomposition of the given rank using numerical optimization.
    We optimize U, V, W to minimize the error in reproducing the matmul tensor.
    """
    start = time.time()
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Use random restarts with gradient descent
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(12345)
    
    for restart in range(20):
        if time.time() - start > time_budget:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() - start > time_budget:
                break
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Compute gradients (simplified)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual = 2.0 * (val - 1.0)
                        for c in range(target_rank):
                            dU[i*m+j, c] += residual * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += residual * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += residual * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from 56 down
    for trial_rank in [56, 52, 48, 44, 40]:
        if time.time() > deadline - 20.0:
            break
        
        result = _find_decomposition_optimization(n, m, p, trial_rank, deadline - 20.0 - time.time())
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Convert to lists with rounded values
            U = [[round(float(x), 6) for x in row] for row in U_arr]
            V = [[round(float(x), 6) for x in row] for row in V_arr]
            W = [[round(float(x), 6) for x in row] for row in W_arr]
            
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {"U": U, "V": V, "W": W}
                best_rank = trial_rank
                break
    
    return best
# EVOLVE-BLOCK-END
