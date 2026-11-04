"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U,V,W correctly decompose the MM tensor."""
        rank = U.shape[1]
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_approx, T_target, atol=1e-10)
    
    # Try to find a lower rank decomposition using structured search
    # Strategy: Use the block structure 3x3x5 = 3x3x3 + 3x3x2
    # Try to find a 3x3x3 decomposition of rank r1 and 3x3x2 of rank r2
    # Total rank = r1 + r2
    
    # First, try to find a good 3x3x3 decomposition
    # The Bini decomposition has rank 23, but let's try to find one
    
    def find_3x3x3_decomp(max_rank=27, max_attempts=30):
        """Find a decomposition for 3x3x3 MM tensor."""
        n3, m3, p3 = 3, 3, 3
        T_target = np.zeros((n3*m3, m3*p3, p3*n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T_target[i*m3+j, j*p3+k, k*n3+i] = 1.0
        
        best_U, best_V, best_W = None, None, None
        best_r = max_rank + 1
        
        for target_rank in range(20, max_rank + 1):
            if time.time() > deadline - 20:
                break
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Initialize
                U = np.random.randn(n3*m3, target_rank) * 0.5
                V = np.random.randn(m3*p3, target_rank) * 0.5
                W = np.random.randn(p3*n3, target_rank) * 0.5
                
                # Gradient descent
                lr = 0.02
                for step in range(300):
                    if time.time() > deadline - 5:
                        break
                    
                    T_approx = np.zeros((n3*m3, m3*p3, p3*n3))
                    for r in range(target_rank):
                        T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                    
                    error = T_approx - T_target
                    if np.linalg.norm(error) < 1e-12:
                        break
                    
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    for r in range(target_rank):
                        dU[:,r] = np.einsum('abc,bc->a', error, np.outer(V[:,r], W[:,r]))
                        dV[:,r] = np.einsum('abc,ac->b', error, np.outer(U[:,r], W[:,r]))
                        dW[:,r] = np.einsum('abc,ab->c', error, np.outer(U[:,r], V[:,r]))
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                if verify_decomposition(U, V, W, n3, m3, p3):
                    best_U, best_V, best_W = U, V, W
                    best_r = target_rank
                    break
            if best_r == target_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_r
        return None, None, None, max_rank
    
    # Find 3x3x3 decomposition
    U3, V3, W3, r3 = find_3x3x3_decomp(max_rank=27, max_attempts=20)
    
    if U3 is not None and r3 < 27:
        # Now handle the 3x3x2 part
        # This is simpler: rank 18 schoolbook
        n2, m2, p2 = 3, 3, 2
        r2 = n2 * m2 * p2  # 18
        
        U2 = np.zeros((n2*m2, r2))
        V2 = np.zeros((m2*p2, r2))
        W2 = np.zeros((p2*n2, r2))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i*m2+j, col] = 1
                    V2[j*p2+k, col] = 1
                    W2[k*n2+i, col] = 1
                    col += 1
        
        # Combine: the 3x3x5 tensor is the sum of 3x3x3 (for k=0,1,2) and 3x3x2 (for k=3,4)
        # U has shape (9, r3 + r2), V has shape (15, r3 + r2), W has shape (15, r3 + r2)
        total_rank = r3 + r2
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # First part: 3x3x3 (k=0,1,2)
        U_full[:, :r3] = U3
        V_full[:9, :r3] = V3  # V3 has shape (9, r3), first 9 rows of V_full
        W_full[:9, :r3] = W3  # W3 has shape (9, r3), first 9 rows of W_full
        
        # Second part: 3x3x2 (k=3,4)
        # V2 has shape (9, 18), but we need to map to V_full[9:, r3:]
        # V_full[9:, :] corresponds to j*p+k where j=0,1,2 and k=3,4
        # So V_full[9:12, r3:] = V2[0:3, :]  (j=0, k=3,4)
        # V_full[12:15, r3:] = V2[3:6, :]  (j=1, k=3,4)
        # Wait, V2 has shape (m2*p2, r2) = (6, 18)
        # V_full has shape (m*p, total_rank) = (15, total_rank)
        # V_full[9:, r3:] should be V2
        
        V_full[9:, r3:] = V2  # This should work: V2 is (6, 18), V_full[9:15, r3:] is (6, 18)
        
        # W2 has shape (p2*n2, r2) = (6, 18)
        # W_full[9:, r3:] should be W2
        W_full[9:, r3:] = W2
        
        # Verify the full decomposition
        if verify_decomposition(U_full, V_full, W_full, n, m, p):
            best_result = (U_full.tolist(), V_full.tolist(), W_full.tolist())
            best_rank = total_rank
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
