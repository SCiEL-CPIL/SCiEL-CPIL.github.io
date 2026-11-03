import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing factor matrices
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank is 54
    # We'll try to find a better decomposition
    
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # First, provide the schoolbook solution as fallback
    def schoolbook_solution():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # The tensor T[i,j,k] = 1 for the multiplication C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, the convention is W[k*n+i, r], so W index is k*n+i
        
        # Let me re-read: W[k*n+i, r]
        # So for each r, we have U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta_{i,j,k}
        # where delta is 1 if this is a valid multiplication term, 0 otherwise
        
        # Actually, for matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # The tensor T[i, j, k] = 1 for all valid (i, j, k) combinations
        # But wait, that's not right. The tensor decomposition represents:
        # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * (coefficient)
        # The coefficient must be 1 for the correct j and 0 otherwise
        
        # Actually, the standard tensor decomposition for matrix multiplication:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
        
        # For matrix multiplication, T[i,j,k] = 1 for ALL (i,j,k) in the valid range
        # because every combination (i,j,k) contributes to C[i,k]
        
        # Wait, no. Let me think again.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid i, j, k
        # And we need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        
        # Actually, I think the tensor entries should be:
        # T[i,j,k] = 1 for all (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k]
        
        # Let me verify with the schoolbook:
        # For each (i,j,k), there's exactly one r where U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
        # So sum_r = 1 for that specific r, 0 for others. Correct.
        
        # Now verify:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def try_strassen_like():
        """Try to find a Strassen-like decomposition with lower rank."""
        # For 3x3 * 3x6, we can try various approaches
        
        # Approach 1: Use the known result that 3x3 matrix multiplication has rank 19
        # (or possibly lower). For 3x3 * 3x6, we can split the 6 columns into two 3x3 blocks.
        
        # If we can multiply 3x3 * 3x3 with rank R, then 3x3 * 3x6 with rank 2R
        # But we can do better by sharing some factors.
        
        # Let's try a structured approach:
        # Split B into B1 (3x3) and B2 (3x3)
        # C1 = A * B1, C2 = A * B2
        # Use the same U factors for both, but different V and W
        
        # Actually, let's try a random search for lower rank decompositions
        
        best_rank = n * m * p  # Start with schoolbook
        best_solution = schoolbook_solution()
        
        # Try ranks from 30 to 53
        for rank in range(30, min(54, best_rank)):
            if time.time() > deadline - 20:
                break
            
            # Try a few random initializations
            for attempt in range(10):
                if time.time() > deadline - 20:
                    break
                
                # Random initialization
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
                
                # Use gradient descent to minimize the error
                lr = 0.01
                for step in range(50):
                    if time.time() > deadline - 20:
                        break
                    
                    # Compute gradients
                    U_grad = np.zeros_like(U)
                    V_grad = np.zeros_like(V)
                    W_grad = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # Error term
                                err = 0
                                for r in range(rank):
                                    err += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                err -= 1  # Target is 1
                                
                                # Gradient contributions
                                for r in range(rank):
                                    U_grad[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                    V_grad[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                    W_grad[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                    
                    # Update
                    U -= lr * U_grad
                    V -= lr * V_grad
                    W -= lr * W_grad
                
                # Check if we found a valid solution
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify_decomposition(U_list, V_list, W_list):
                    best_rank = rank
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        return best_solution
    
    # Try to find a better solution
    try:
        solution = try_strassen_like()
    except Exception:
        solution = schoolbook_solution()
    
    return solution
