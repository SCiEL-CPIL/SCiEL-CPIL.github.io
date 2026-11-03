import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W):
    """Verify that U,V,W form a valid decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the multiplication tensor from decomposition
    # T[i,j,k,r] should equal sum over decomposition
    # The tensor T for matrix multiplication has T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # In the factorization: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The tensor element T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (it's the identity mapping in tensor form)
    # Actually no. The matrix multiplication tensor T has T[i,j,k] = delta (the product term).
    # Let me think again. The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T satisfies: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k]
    # So T[i,j,k] = 1 for all i,j,k in the schoolbook case.
    # With factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _rank_score(U, V, W):
    """Return rank if valid, None otherwise."""
    if not _verify(U, V, W):
        return None
    return U.shape[1]

def _try_rank(rank):
    """Try to find a decomposition of given rank using random search with local refinement."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We need U (nm x rank), V (mp x rank), W (pn x rank)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is equivalent to: the tensor T (which is all-ones in the appropriate sense) 
    # can be written as sum_r U[:, r] ⊗ V[:, r] ⊗ W[:, r]
    
    # Random initialization
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Normalize to avoid scale issues
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    # Gradient descent on the loss: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(500):
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    loss += (val - 1.0)**2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            return best_U, best_V, best_W
        
        # Gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = 2 * (val - 1.0)
                    dU[i*m+j] += err * (V[j*p+k] * W[k*n+i])
                    dV[j*p+k] += err * (U[i*m+j] * W[k*n+i])
                    dW[k*n+i] += err * (U[i*m+j] * V[j*p+k])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_loss < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with schoolbook rank 50
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    best_U = np.array(U, dtype=float)
    best_V = np.array(V, dtype=float)
    best_W = np.array(W, dtype=float)
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Try ranks from 46 down to 40
    for target_rank in range(46, 35, -1):
        if time.time() > deadline - 10:
            break
        
        found = None
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            result = _try_rank(target_rank)
            if result is not None:
                U_cand, V_cand, W_cand = result
                if _verify(U_cand, V_cand, W_cand):
                    found = (U_cand, V_cand, W_cand)
                    break
        
        if found is not None:
            U_cand, V_cand, W_cand = found
            # Round to nearest rational for exactness
            best_U = U_cand
            best_V = V_cand
            best_W = W_cand
            best_rank = target_rank
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Try to round to simple fractions
    def round_matrix(M):
        M_rounded = np.round(M * 4) / 4
        return M_rounded.tolist()
    
    U_list = round_matrix(best_U)
    V_list = round_matrix(best_V)
    W_list = round_matrix(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
