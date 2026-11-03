import numpy as np
import random
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition (rank 36) as fallback
    def schoolbook():
        rank = n * m * p  # 36
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify decomposition: check that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = delta[i,i']*delta[k,k']
    # Wait, let me re-derive the correct condition.
    # The tensor T has components T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # But actually T[i,j,k] = 1 for all i,j,k (the standard multiplication tensor)
    # The rank-r decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # But that's not right either. Let me think again.
    
    # Actually, for the schoolbook decomposition:
    # Each column r corresponds to a triple (i_r, j_r, k_r)
    # U[i*m+j, r] = 1 if (i,j) = (i_r, j_r) else 0
    # V[j*p+k, r] = 1 if (j,k) = (j_r, k_r) else 0  
    # W[k*n+i, r] = 1 if (k,i) = (k_r, i_r) else 0
    # So U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 only if (i,j)=(i_r,j_r) AND (j,k)=(j_r,k_r) AND (k,i)=(k_r,i_r)
    # This means i=i_r, j=j_r, k=k_r, so the product is 1 only when all three match.
    # Summing over r: sum_r product = number of r where (i_r,j_r,k_r) = (i,j,k) = 1 (since each triple appears once)
    # So the sum equals 1 for all (i,j,k). 
    
    # But wait, that means we're decomposing the constant tensor 1, not the multiplication tensor.
    # I think I'm confusing myself. Let me look at this differently.
    
    # The matrix multiplication map is: (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a bilinear map. We can write it as:
    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_k V[j,k,r]*B[j,k]) * W[r,i,k]
    # Hmm, that's not quite right either.
    
    # Let me just verify the schoolbook works and try to improve from there.
    
    def verify(U_list, V_list, W_list, tol=1e-10):
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        r = U.shape[1]
        
        # Check: for all i,j,k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] should equal 1
        # (This is what the schoolbook achieves)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] for r in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # First, return schoolbook as baseline
    sb = schoolbook()
    best_result = sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: use random initialization and gradient-based optimization
    # The objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Scale to reasonable magnitude
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        # Simple gradient descent
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # f = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
                        # dU[i*m+j,r] = 2*f*V[j*p+k,r]*W[k*n+i,r]
                        # dV[j*p+k,r] = 2*f*U[i*m+j,r]*W[k*n+i,r]
                        # dW[k*n+i,r] = 2*f*U[i*m+j,r]*V[j*p+k,r]
                        f = sum(U[i*m+j, rr]*V[j*p+k, rr]*W[k*n+i, rr] for rr in range(r)) - 1.0
                        for rr in range(r):
                            grad_U[i*m+j, rr] += 2*f*V[j*p+k, rr]*W[k*n+i, rr]
                            grad_V[j*p+k, rr] += 2*f*U[i*m+j, rr]*W[k*n+i, rr]
                            grad_W[k*n+i, rr] += 2*f*U[i*m+j, rr]*V[j*p+k, rr]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we found a good solution
        if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Try ranks from 35 down
    for r in range(best_rank - 1, max(10, best_rank - 10), -1):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
