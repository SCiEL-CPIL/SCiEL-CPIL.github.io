"""Program for matrix multiplication tensor decomposition (4,5,5)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-8):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        # Build tensor T[i,j,k] and check it equals 1 everywhere
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            T += Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :] * Wr.T[np.newaxis, np.newaxis, :]
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=tol)
    
    # Try to find a lower-rank decomposition using structured random search
    # Focus on ranks that are plausible: lower bound is max(n*m, m*p, p*n) = 25
    # Try ranks from 25 upward, but limit search to keep within time budget
    
    # Strategy: For each candidate rank, try random {0, 1, -1} matrices and verify
    # If found, return immediately
    
    # To speed up verification, use a faster check
    def fast_verify(U, V, W, tol=1e-8):
        """Faster verification using matrix operations"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        # For each (i,k) pair, check that sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is equivalent to checking that the bilinear form matches
        # C[i,k] = sum_j A[i,j]*B[j,k] should equal sum_r (A@U_r[i,:])*(V_r@B[:,k])*W[k*n+i,r]
        # Let's just check the tensor directly but with vectorized ops
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # Ur[i,j] * Vr[j,k] * Wr[k,i] summed over j for each (i,k)
            # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            T += Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :] * Wr.T[np.newaxis, np.newaxis, :]
        return np.allclose(T, 1.0, atol=tol, rtol=tol)
    
    # Try a few ranks with random search
    # Use deterministic seeds for reproducibility
    np.random.seed(42)
    
    # Try ranks from a reasonable range
    # The theoretical lower bound is 25, but finding such a decomposition is very hard
    # Let's try a few ranks and see if we get lucky
    for target_rank in range(25, 60):
        if time.time() > deadline - 15:
            break
        
        # Number of attempts based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        attempts = max(1, min(10, int(time_remaining / 5)))
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Random matrices with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            if fast_verify(U, V, W):
                # Found a valid decomposition!
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify entries are integers (they are, since we chose from {-1,0,1})
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                return best_result
        
        if time.time() > deadline - 30:
            break
    
    # If random search failed, try a structured approach
    # Use the fact that we can combine schoolbook terms
    # For now, return the schoolbook decomposition
    
    return best_result
# EVOLVE-BLOCK-END

