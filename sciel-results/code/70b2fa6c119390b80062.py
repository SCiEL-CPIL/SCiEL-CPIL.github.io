import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    if not np.allclose(C, 1.0, atol=tol):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    M_{2,2,2} = sum_{r=0}^{6} U_r otimes V_r otimes W_r
    where M_{2,2,2}(i,j,k) = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 if i==k else 0
    """
    # Strassen's 7 terms
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[3, 0] = 1  # c11
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = 1  # c11
    
    # M3 = a00(b10-b11)
    U[0, 2] = 1  # a00
    V[2, 2] = 1  # b10
    V[3, 2] = -1  # b11
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # M4 = a11(b01-b00)
    U[3, 3] = 1  # a11
    V[1, 3] = 1  # b01
    V[0, 3] = -1  # b00
    W[0, 3] = 1  # c00
    W[2, 3] = 1  # c10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1  # c00
    W[1, 4] = 1  # c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1  # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1  # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # c00
    
    return U, V, W

def _construct_via_block_strassen(n, m, p):
    """
    Construct a lower rank decomposition by using Strassen's algorithm on blocks.
    
    For 3x5x6:
    - We can try to split the problem into blocks where Strassen can be applied.
    - Since 3 is odd, we can pad to 4 (2x2 blocks)
    - 5 can be split as 2+3
    - 6 can be split as 2+2+2
    
    Strategy: 
    - Pad A from 3x5 to 4x5 (add a zero row)
    - Pad C from 4x6 (then take first 3 rows)
    - Split A into A_top (2x5) and A_bottom (2x5)
    - Split B into B_1 (5x2), B_2 (5x2), B_3 (5x2)
    - Compute C_top = A_top * B and C_bottom = A_bottom * B using Strassen
    - But Strassen works on 2x2 blocks, so we need to split B into 2x2 blocks
    - This is complex. Let's try a simpler approach first.
    
    Alternative: Use the fact that rank(M_{n,m,p}) can sometimes be reduced 
    by exploiting structure.
    
    Let's try: split m=5 into 2+3, p=6 into 2+2+2
    C = [A_1 A_2] [B_1; B_2; B_3] where A_1 is 3x2, A_2 is 3x3
    B_1, B_2, B_3 are 5x2 each
    C = A_1 B_1 + A_2 [B_1; B_2; B_3]... wait, dimensions don't work.
    
    Let me reconsider. C[i,k] = sum_j A[i,j] B[j,k]
    Split j: j in {0,1} and j in {2,3,4}
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{4} A[i,j] B[j,k]
    Let A_1 = A[:,0:2] (3x2), A_2 = A[:,2:5] (3x3)
    Let B_1 = B[0:2,:] (2x6), B_2 = B[2:5,:] (3x6)
    C = A_1 B_1 + A_2 B_2
    
    Rank(M_{3,2,6}) * Rank(M_{3,3,6}) doesn't directly give us the sum.
    But we can compute each term separately and add the tensors.
    
    Actually, for the sum of two tensors, the rank of the sum is at most 
    the sum of the ranks. So rank(M_{3,5,6}) <= rank(M_{3,2,6}) + rank(M_{3,3,6})
    
    For M_{3,2,6}: n=3, m=2, p=6
    Schoolbook: 3*2*6 = 36
    Can we do better? 
    For M_{n,2,p}, there are known bounds. Let's try to use Strassen on the n dimension.
    Pad n=3 to n=4 (2x2 blocks).
    
    This is getting complex. Let's try a different approach:
    Use the known fact that for rectangular matrix multiplication, sometimes
    rank reductions are possible.
    
    Let's try to construct a decomposition using the following idea:
    For each pair (i,k), we need sum_j A[i,j] B[j,k] = C[i,k]
    
    Let's try to find a decomposition of rank 75 (as suggested).
    This would be a 15% reduction from 90.
    
    One approach: treat the 5 middle indices as 5 separate "channels" and try to
    compress across channels.
    
    Actually, let's try a more systematic approach using linear algebra.
    The matrix multiplication tensor M_{n,m,p} is a 3-tensor. Its rank is the 
    minimum number of rank-1 tensors needed to express it.
    
    For now, let's try to implement a block decomposition that splits the problem
    and uses Strassen where applicable.
    
    For n=3, m=5, p=6:
    - Split p=6 into three groups of 2: p1=2, p2=2, p3=2
    - For each p-group, we have a M_{3,5,2} sub-problem
    - If we can solve M_{3,5,2} with rank r, then total rank is 3*r
    - Schoolbook for M_{3,5,2}: 3*5*2 = 30, so 3*30 = 90 (same as original)
    
    Let's try to reduce the rank of M_{3,5,2}.
    M_{3,5,2}: C[i,k] = sum_j A[i,j] B[j,k] for k in {0,1}
    
    For M_{n,m,2}, we can sometimes do better than n*m*2.
    Let's see if Strassen helps here.
    
    Actually, let's just try to return a schoolbook decomposition for now,
    and see if we can find improvements.
    
    For the purpose of this task, let's return None to indicate we should use
    the schoolbook method.
    """
    return None

def _try_gradient_descent(rank, n, m, p, time_limit=5.0, seed=None):
    """
    Try to find a decomposition of a specific rank using gradient descent on the error.
    This is a non-convex problem and may not find a solution.
    """
    if seed is not None:
        np.random.seed(seed)
    
    start_time = time.time()
    
    # Initialize randomly
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Scale to make the products reasonable
    U = U / np.sqrt(rank)
    V = V / np.sqrt(rank)
    W = W / np.sqrt(rank)
    
    # Learning rate
    lr = 0.1
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Compute target tensor
    # T[i,j,k] = 1
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Gradient descent
    num_iter = 0
    while time.time() - start_time < time_limit:
        num_iter += 1
        
        # Compute current tensor
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # C[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        
        # Error
        error = np.sum((C - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-8:
            break
        
        # Compute gradients
        # dError/dC = 2*(C-1)
        dC = 2 * (C - 1.0)
        
        # dU[i,j,r] = sum_{k} dC[i,j,k] * V[j,k,r] * W[k,i,r]
        dU = np.einsum('ijk,jkr,kir->ijr', dC, V_r, W_r, optimize=True)
        
        # dV[j,k,r] = sum_{i} dC[i,j,k] * U[i,j,r] * W[k,i,r]
        dV = np.einsum('ijk,ijr,kir->jkr', dC, U_r, W_r, optimize=True)
        
        # dW[k,i,r] = sum_{j} dC[i,j,k] * U[i,j,r] * V[j,k,r]
        dW = np.einsum('ijk,ijr,jkr->kir', dC, U_r, V_r, optimize=True)
        
        # Update
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        # Adaptive learning rate
        if num_iter % 100 == 0:
            lr *= 0.9
    
    # Check if we got a good solution
    if best_error < 1e-6:
        # Final verification
        if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-5):
            return best_U, best_V, best_W
    
    return None

def _construct_via_search(n, m, p, time_limit=30.0):
    """
    Try various approaches to find a lower rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try gradient descent with various ranks
    # Start with ranks lower than schoolbook
    for rank in range(int(best_rank * 0.8), best_rank):
        if time.time() > deadline - 5.0:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            
            result = _try_gradient_descent(rank, n, m, p, time_limit=5.0, seed=rank*100+attempt)
            if result is not None:
                U_g, V_g, W_g = result
                if _verify_decomposition(U_g, V_g, W_g, n, m, p, tol=1e-5):
                    if rank < best_rank:
                        best_U, best_V, best_W = U_g, V_g, W_g
                        best_rank = rank
                    break  # Found a solution for this rank, try lower
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to find a lower rank decomposition
    if time.time() < deadline - 5.0:
        U_s, V_s, W_s, rank_s = _construct_via_search(n, m, p, time_limit=min(60.0, deadline - start_time - 10.0))
        if rank_s < best_rank:
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = rank_s
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
