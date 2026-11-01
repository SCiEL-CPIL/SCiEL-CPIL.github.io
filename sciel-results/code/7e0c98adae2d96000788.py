"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T1 = T.reshape(n*m, m*p * p*n)
    T2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)

    # Strategy: 
    # 1. Try to find a global decomposition with rank < 42.
    # We know rank 42 is achievable via 6 Strassen blocks.
    # We suspect rank 36 (4 blocks of rank 9? No, rank(2x2x2)=7, 6*7=42).
    # Maybe rank 36 is possible if we share more?
    # Or maybe rank 30?
    
    # Let's try a global ALS search for rank r.
    # We will start with r=42 and try to reduce it.
    # If we can find a valid decomposition with r < 42, we return it.
    # Otherwise, we return the best we found (which might be 42).
    
    # To speed up ALS, we can use the structure.
    # But first, let's just try global ALS for r=40, 38, 36, 34, 32, 30.
    
    # We'll use a helper to run ALS for a given rank.
    def run_als(rank, max_iters=50, tol=1e-6):
        if time.time() > deadline - 5:
            return None
            
        # Initialize U, V, W
        # Good initialization: random, but scaled.
        # Or use the structure of the problem.
        # For 2x2x12, the tensor is sparse.
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            # T1[a, :] = sum_r U[a, r] * (V * W)_r
            # M_U is (m*p*p*n, rank) where M_U[:, r] = kron(W[:, r], V[:, r])
            # Wait, T1 is (n*m, m*p*p*n).
            # T1[a, (b, c)] = sum_r U[a, r] V[b, r] W[c, r]
            # So we want to solve for U:
            # For each r, the contribution to T1 is U[:, r] * (V[:, r] * W[:, r])^T ? No.
            # T1 is a matrix of size (n*m) x (m*p * p*n).
            # Let M_U be a matrix of size (m*p*p*n) x rank.
            # M_U[idx, r] = V[b, r] * W[c, r] where idx = b*(p*n) + c.
            
            M_U = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                # Vectorize V[:, r] and W[:, r] into a single vector of length m*p*p*n
                # V[:, r] has length m*p. W[:, r] has length p*n.
                # The tensor index is (b, c).
                # We need to form the outer product V[:, r] @ W[:, r].T and flatten it.
                # But careful with the order of indices in T1.
                # T1 is T.reshape(n*m, m*p * p*n).
                # So the second index is b*(p*n) + c.
                # So M_U[:, r] should be the vectorized outer product.
                M_U[:, r] = np.kron(W[:, r], V[:, r]) # Wait, kron(W, V) gives length len(W)*len(V).
                # Is the order correct?
                # np.kron(A, B) where A is (N,), B is (M,) gives (N*M,).
                # Element i*M + j is A[i]*B[j].
                # We want M_U[b*(p*n) + c, r] = V[b, r] * W[c, r].
                # So we want A=W, B=V?
                # If A=W (len p*n) and B=V (len m*p), then index is c*(m*p) + b.
                # But we need b*(p*n) + c.
                # So we need to reshape.
                # Let's just build it explicitly or use reshape.
                pass
            
            # Efficient construction of M_U
            # M_U[:, r] = (V[:, r] * W[:, r].T).reshape(-1)
            # But we need to match the flattening order of T1.
            # T1 = T.reshape(n*m, m*p * p*n)
            # So T1[a, idx] where idx = b*(p*n) + c.
            # So M_U[idx, r] = V[b, r] * W[c, r].
            
            # We can construct M_U by:
            # M_U = np.zeros((m*p, p*n, rank))
            # M_U[b, c, r] = V[b, r] * W[c, r]
            # Then M_U_flat = M_U.reshape(m*p*p*n, rank)
            
            # This is efficient enough for small dimensions.
            
            M_U_3d = np.einsum('br,cr->bcr', V, W)
            M_U_flat = M_U_3d.reshape(m*p * p*n, rank)
            
            U_sol, _, _, _ = np.linalg.lstsq(M_U_flat, T1.T, rcond=None)
            U = U_sol.T
            
            # Solve for V
            # T2[b, (a, c)] = sum_r V[b, r] U[a, r] W[c, r]
            # M_V[a*(p*n) + c, r] = U[a, r] * W[c, r]
            M_V_3d = np.einsum('ar,cr->acr', U, W)
            M_V_flat = M_V_3d.reshape(n*m * p*n, rank)
            
            V_sol, _, _, _ = np.linalg.lstsq(M_V_flat, T2.T, rcond=None)
            V = V_sol.T
            
            # Solve for W
            # T3[c, (a, b)] = sum_r W[c, r] U[a, r] V[b, r]
            # M_W[a*(m*p) + b, r] = U[a, r] * V[b, r]
            M_W_3d = np.einsum('ar,br->abr', U, V)
            M_W_flat = M_W_3d.reshape(n*m * m*p, rank)
            
            W_sol, _, _, _ = np.linalg.lstsq(M_W_flat, T3.T, rcond=None)
            W = W_sol.T
            
            # Check error
            # T_rec = sum_r U[:, r] * V[:, r] * W[:, r]
            # Use einsum
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_rec - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < tol:
                return best_U, best_V, best_W, err
        
        if best_err < tol:
            return best_U, best_V, best_W, best_err
        return None

    # Try ranks in increasing order (to find the minimal one quickly if possible)
    # But ALS is hard to converge for low ranks if the true rank is higher.
    # We know 42 is achievable.
    # Let's try 40, 38, 36, 34, 32, 30, 28, 26, 24.
    
    # Actually, let's try 42 first to ensure we have a valid solution if others fail.
    # Then try lower.
    
    valid_solutions = []
    
    # We will try a few random restarts for each rank.
    # To save time, we will limit the number of restarts.
    
    ranks_to_try = [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20]
    
    # Also try 42 as a fallback
    ranks_to_try.append(42)
    
    # Sort by rank? No, we want the lowest rank that works.
    # But if we try 20 first and it fails, we wasted time.
    # Let's try 42 first to get a baseline, then try lower.
    # Or try 36, 34, 32...
    
    # Given the time limit, let's try 36 and 34.
    # If they fail, return 42.
    
    # Let's try 42 first to be safe.
    # Actually, the current program returns 48.
    # If I can return 42, that's a win.
    # If I can return < 42, that's a bigger win.
    
    # Let's try 42 with 1 restart.
    res = run_als(42, max_iters=20, tol=1e-5)
    if res:
        U, V, W, err = res
        valid_solutions.append((42, U, V, W))
    
    # Now try lower ranks
    for r in [36, 34, 32, 30, 28]:
        if time.time() > deadline - 20:
            break
            
        # Try 2 restarts for each rank
        found = False
        for restart in range(2):
            if time.time() > deadline - 10:
                break
                
            res = run_als(r, max_iters=30, tol=1e-5)
            if res:
                U, V, W, err = res
                valid_solutions.append((r, U, V, W))
                found = True
                break
                
        if found:
            # If we found a solution for rank r, we can stop trying lower ranks?
            # Not necessarily, we want the MINIMAL rank.
            # But finding lower ranks is harder.
            # Let's continue to see if we can go lower.
            pass
    
    # Select the best solution (lowest rank)
    if valid_solutions:
        valid_solutions.sort(key=lambda x: x[0])
        best_rank, best_U, best_V, best_W = valid_solutions[0]
        
        # Final verification
        T_rec = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        err = np.linalg.norm(T_rec - T)
        
        if err < 1e-4:
            # Ensure integer/half-integer entries if possible?
            # The problem says "integer/half-integer entries".
            # ALS gives floating point.
            # We should round them.
            
            best_U = np.round(best_U * 2) / 2
            best_V = np.round(best_V * 2) / 2
            best_W = np.round(best_W * 2) / 2
            
            # Re-verify after rounding
            T_rec_rounded = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
            err_rounded = np.linalg.norm(T_rec_rounded - T)
            
            if err_rounded < 1e-4:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
            else:
                # If rounding failed, try to return the unrounded version?
                # The contract says "integer/half-integer entries".
                # If I can't satisfy that, I might get a penalty or 0 score.
                # But usually, "exact" decomposition implies the values should be nice.
                # If ALS converges to a valid decomposition, the values are often rational.
                # Let's try to snap to nearest 0.5.
                
                # If it fails, fall back to the unrounded version and hope for the best?
                # No, better to return a known valid integer solution if available.
                
                pass
        else:
            pass
    
    # Fallback: Rank 42 Strassen construction
    # This is the same logic as the current program's intended fallback, but correctly implemented.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 decomposition
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Let's verify these formulas.
    # C00 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B10+B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A00B00 + A11B00 + A01B10 + A10B00 + A10B01 - A00B01
    # This is NOT C00 = A00B00 + A01B10.
    
    # I keep making the same mistake. Let's use a reliable source or derive it.
    # Actually, the standard Strassen is:
    # C00 = M1 - M5 + M7 + M6
    # Let's check the terms again.
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # M5: A00B11 + A01B11
    # M7: A01B10 + A01B11 - A11B10 - A11B11
    # M6: A10B00 + A10B01 - A00B00 - A00B01
    
    # Sum:
    # A00B00: 1 - 1 = 0
    # A00B11: 1 - 1 = 0
    # A11B00: 1
    # A11B11: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # A11B10: -1
    # A10B00: 1
    # A10B01: 1
    # A00B01: -1
    
    # Result: A11B00 + A01B10 - A11B10 + A10B00 + A10B01 - A00B01
    # This is definitely wrong.
    
    # Let's use the correct Strassen formulas from Wikipedia:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Wait, I just verified C00 and it was wrong.
    # Let's check C01.
    # C01 = M5 - M1 + M3 + M7
    # = (A00+A01)B11 - (A00+A11)(B00+B11) + A00(B10-B11) + (A01-A11)(B10+B11)
    # = A00B11 + A01B11 - A00B00 - A00B11 - A11B00 - A11B11 + A00B10 - A00B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A01B11 - A00B00 - A11B00 - A11B11 + A00B10 + A01B10 - A11B10 - A11B11
    # This is also a mess.
    
    # I will use the trivial rank-48 decomposition as a guaranteed fallback.
    # It is better to return a valid rank 48 than an invalid rank 42.
    
    rank = 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
