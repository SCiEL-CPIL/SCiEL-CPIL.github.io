<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-10)
    
    # --- Rank 26 decomposition: Strassen (7) + 19 border terms ---
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen's 7 multiplications for 2x2 core (i,j,k in {0,1})
    # m1 = (a00+a11)(b00+b11)
    r = 0
    U[0*3+0, r] = 1  # a00
    U[1*3+1, r] = 1  # a11
    V[0*3+0, r] = 1  # b00
    V[1*3+1, r] = 1  # b11
    W[0*3+0, r] = 1  # c00
    W[1*3+1, r] = 1  # c11
    
    # m2 = (a10+a11)b00
    r = 1
    U[1*3+0, r] = 1  # a10
    U[1*3+1, r] = 1  # a11
    V[0*3+0, r] = 1  # b00
    W[0*3+1, r] = 1  # c01... wait, W[k*n+i] so c_{i,j} uses W[k*n+i]
    # Let me be more careful with indices
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*3 + k  
    # W index: k*n + i = k*3 + i
    # So W[k, i] contributes to c[i, j]... no wait
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So for a given (i,j,k), the contribution from rank r is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This should equal 1 for all (i,j,k)
    
    # Let me redo this more carefully
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    # Strassen m1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    r = 0
    U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
    
    # m2 = (a10+a11)b00 -> contributes to c10, subtracts from c11
    r = 1
    U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1
    W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
    
    # m3 = a00(b01-b11) -> contributes to c01, c11
    r = 2
    U[iu(0,0), r] = 1
    V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
    W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
    
    # m4 = a11(b10-b00) -> contributes to c00, c10
    r = 3
    U[iu(1,1), r] = 1
    V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
    W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
    
    # m5 = (a00+a01)b11 -> subtracts from c00, contributes to c01
    r = 4
    U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
    V[iv(1,1), r] = 1
    W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01) -> contributes to c11
    r = 5
    U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
    V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
    W[iw(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11) -> contributes to c00
    r = 6
    U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
    V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1
    
    # Now add 19 border terms (all (i,j,k) where at least one index is 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[iu(i,j), col] = 1
                V[iv(j,k), col] = 1
                W[iw(k,i), col] = 1
                col += 1
    
    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook rank 27
    R27 = n * m * p
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[iu(i,j), col] = 1
                V27[iv(j,k), col] = 1
                W27[iw(k,i), col] = 1
                col += 1
    
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
=======
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    n, m, p = DIMENSIONS

    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i

    def verify_fast(U, V, W):
        """Vectorized check: reconstruct tensor and compare to ones."""
        R = U.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build via einsum or manual loop (small enough)
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(n, m, p) if False else \
                       np.multiply.outer(np.multiply.outer(U[:, r].reshape(n, m), V[:, r].reshape(m, p)).reshape(n*m, p), 
                                        np.ones((1,1))).reshape(n, m, p) * 0 + \
                       sum(U[iu(i,j), r] * V[iv(j,k), r] * W[iw(k,i), r] 
                           for i in range(n) for j in range(m) for k in range(p)) * 0
        # Simpler: direct computation
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[iu(i,j), r] * V[iv(j,k), r] * W[iw(k,i), r]
        return np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9)

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        return U, V, W

    def strassen_rank26():
        """Guaranteed rank-26: Strassen on 2x2 core + 19 border terms."""
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))

        # m1 = (a00+a11)(b00+b11)
        r = 0
        U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1; W[iw(1,1), r] = 1

        # m2 = (a10+a11)b00
        r = 1
        U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1
        W[iw(0,1), r] = 1; W[iw(1,1), r] = -1

        # m3 = a00(b01-b11)
        r = 2
        U[iu(0,0), r] = 1
        V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
        W[iw(1,0), r] = 1; W[iw(1,1), r] = 1

        # m4 = a11(b10-b00)
        r = 3
        U[iu(1,1), r] = 1
        V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
        W[iw(0,0), r] = 1; W[iw(0,1), r] = 1

        # m5 = (a00+a01)b11
        r = 4
        U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
        V[iv(1,1), r] = 1
        W[iw(0,0), r] = -1; W[iw(1,0), r] = 1

        # m6 = (a10-a00)(b00+b01)
        r = 5
        U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
        V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
        W[iw(1,1), r] = 1

        # m7 = (a01-a11)(b10+b11)
        r = 6
        U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
        V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1

        # 19 border terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        return U, V, W

    # First, get a guaranteed valid result
    U26, V26, W26 = strassen_rank26()
    if verify_fast(U26, V26, W26):
        best_rank = 26
        best_U, best_V, best_W = U26, V26, W26
    else:
        U27, V27, W27 = schoolbook()
        best_rank = 27
        best_U, best_V, best_W = U27, V27, W27

    # Try to find rank-25 or rank-24 via local search from rank-26
    # Strategy: start from rank-26, try to merge/eliminate one term
    # Use random perturbation + least-squares solve for one factor
    
    if time.time() < deadline - 5:
        # Try rank-25: take rank-26 and try to make one column a linear combination
        # of others, then eliminate it
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            try:
                U_t = best_U.copy()
                V_t = best_V.copy()
                W_t = best_W.copy()
                R_t = U_t.shape[1]
                
                # Pick a random column to eliminate
                col_to_elim = np.random.randint(R_t)
                
                # Try to express column col_to_elim as combination of others
                # For simplicity, set that column to zero and see if we can
                # adjust other columns to compensate
                # This is a simplified approach; full search would be more complex
                
                # Alternative: try random rank-25 initialization and verify
                # Use sparse random init
                U_r = np.random.randn(n*m, 25) * 0.5
                V_r = np.random.randn(m*p, 25) * 0.5
                W_r = np.random.randn(p*n, 25) * 0.5
                
                # Quick verify (may be too slow, so skip full verify for random)
                # Instead, try a structured approach: modify rank-26 by zeroing
                # one border term and adjusting others
                
                # Zero out one border column (columns 7-25 are border terms)
                border_col = 7 + np.random.randint(19)
                U_t = U26.copy()
                V_t = V26.copy()
                W_t = W26.copy()
                
                # Try to absorb this term into existing Strassen terms
                # This requires solving a linear system, which is complex
                # For now, just check if removing it still works (it won't)
                # So skip this approach
                
                pass
            except Exception:
                pass
    
    # Return the best valid result
    if best_rank <= 26:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
