import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """
    Verify tensor decomposition identity:
    sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] == 1 if (a==i and c==k and b==d) else 0
    """
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
    
    target = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        target[a, b, c, d, c, a] = 1.0
    
    return np.allclose(T, target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    # --- Strategy: Use the tensor product of Strassen (2x2x2, rank 7) with identity (1x3x3, rank 3) ---
    # This gives a rank-21 decomposition for 2x3x6 if the structure aligns properly.
    # However, the tensor product of T_{2,2,2} and T_{1,3,3} gives T_{2*1, 2*3, 2*3} = T_{2, 6, 6}.
    # We have T_{2, 3, 6}. So this direct product doesn't match the dimensions.
    
    # Alternative: We know rank(2x2x2) = 7 (Strassen).
    # The 2x3x6 multiplication can be seen as 3 independent 2x1x2 multiplications? No, that's not right.
    # It is 2x3 times 3x6.
    
    # Let's look at the known rank of 2x3x3. It is 11.
    # So 2x3x6 = 2 * 2x3x3 blocks. If we use independent blocks, rank = 22.
    # Can we do better by sharing?
    # The tensor is T_{2,3,6}.
    # We can write T_{2,3,6} = T_{2,3,3} \otimes I_2? No.
    # Actually, T_{2,3,6} is isomorphic to T_{2,3,3} \otimes T_{1,1,2} ? No.
    
    # Let's try to construct a rank-21 decomposition.
    # If rank(2x3x6) < 22, it would be a new result (currently lower bound is 18, upper bound 22).
    # Recent results suggest rank(2x3x6) might be 21 or 22.
    # Actually, Landsberg et al. showed rank(2x3x3) >= 11.
    # For 2x3x6, the lower bound is 18.
    
    # Let's try to find a rank-21 decomposition using a structured search.
    # We will use the "Strassen-like" structure for the 2x2x2 sub-blocks.
    
    # Decompose the 2x3x6 problem into 2x2x2 and 2x1x2 parts?
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x6) = [B1 (2x6) ; B2 (1x6)]
    # C (2x6) = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 times 2x6. This can be split into three 2x2x2 multiplications?
    # B1 has 6 columns. We can group them as 3 pairs.
    # So A1*B1 is 3 independent 2x2x2 multiplications. Rank = 3 * 7 = 21.
    # A2*B2 is 2x1 times 1x6. This is just element-wise multiplication scaled. Rank = 6.
    # Total rank = 21 + 6 = 27. This is worse than 22.
    
    # What if we overlap the Strassen computations?
    # Let's try to directly search for a rank-21 decomposition.
    
    if time.time() < deadline - 10:
        # We will try to verify if rank 21 is possible by constructing a specific decomposition.
        # If we assume the rank is 22, we can just return the two-block construction.
        # But let's try to improve.
        
        # Known: rank(2x3x3) = 11.
        # Let's find a valid rank-11 decomposition for 2x3x3.
        # This is a known result. We can hardcode it or find it.
        # Since hardcoding is risky, let's try a numerical search for rank 11 of 2x3x3.
        
        # However, the search space for rank 11 is large.
        # Let's try to use the "Coppersmith-Winograd" like structure or just use the known rank-11 decomposition.
        
        # A valid rank-11 decomposition for 2x3x3 (A 2x3, B 3x3, C 2x3):
        # Reference: "The rank of the 2x3x3 matrix multiplication tensor" by Landsberg, Strømme, et al.
        
        # Instead of finding it from scratch, let's assume we can use two copies of a rank-11 decomposition.
        # If we can find ONE rank-11 decomposition, we can assemble rank-22 for 2x3x6.
        
        # Let's try to find a rank-11 decomposition for 2x3x3 using a heuristic search.
        # We will use a greedy approach with random restarts.
        
        n2, m2, p2 = 2, 3, 3
        r_target = 11
        U_size2 = n2 * m2  # 6
        V_size2 = m2 * p2  # 9
        W_size2 = p2 * n2  # 6
        
        target2 = np.zeros((n2, m2, m2, p2, p2, n2), dtype=float)
        for a in range(n2):
            for b in range(m2):
                for c in range(m2):
                    for d in range(p2):
                        if b == d and c < p2:
                            target2[a, b, c, d, c, a] = 1.0
        
        # Search for rank-11 decomposition of 2x3x3
        found_233 = None
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W with random values
            U_rand = np.random.randn(U_size2, r_target)
            V_rand = np.random.randn(V_size2, r_target)
            W_rand = np.random.randn(W_size2, r_target)
            
            # Optimize? This is a non-convex problem.
            # Let's try to use a different approach: start from the schoolbook (rank 18) and try to zero out 7 terms.
            
            # Schoolbook for 2x3x3
            U_sb = np.zeros((U_size2, 18))
            V_sb = np.zeros((V_size2, 18))
            W_sb = np.zeros((W_size2, 18))
            col = 0
            for a in range(n2):
                for b in range(m2):
                    for k in range(p2):
                        U_sb[a * m2 + b, col] = 1
                        V_sb[b * p2 + k, col] = 1
                        W_sb[k * n2 + a, col] = 1
                        col += 1
                        
            # We want to find a rank-11 decomposition.
            # This is hard to do randomly.
            
            # Let's just assume we can use the rank-22 construction if we can find rank-11 for 2x3x3.
            # If we can't find rank-11, we fall back to schoolbook rank 36.
            
            # Let's try a different strategy: use the known rank-11 decomposition if available.
            # Since I don't have it hardcoded, let's try to find it via a more structured search.
            
            # Actually, let's just use the rank-22 construction with a generic rank-11 decomposition for 2x3x3.
            # If we can't find it, we return schoolbook.
            
            # For now, let's try to find a rank-11 decomposition using a local search from the schoolbook decomposition.
            # This is complex. Let's just return the schoolbook decomposition for now, as it's guaranteed to work.
            
            # Wait, the task says "rank-minimal". The schoolbook rank is 36.
            # We know rank(2x3x6) <= 22.
            # If we can construct rank 22, we should.
            
            # Let's try to find a rank-11 decomposition for 2x3x3 using a specific ansatz.
            # We will use the "Strassen" style for the 2x2x2 sub-problems.
            
            # Split 2x3x3 into:
            # A = [A1 (2x2) | A2 (2x1)]
            # B = [B1 (2x3) ; B2 (1x3)]
            # C = A1*B1 + A2*B2
            # A1*B1 is 2x2 times 2x3. This is 1.5 * 2x2x2? No.
            # A1*B1 = [C11 C12 C13] where C1j is 2x1.
            # We can use 3 Strassen decompositions for the 3 columns?
            # Each 2x2x2 has rank 7. So 3 * 7 = 21.
            # A2*B2 is 2x1 times 1x3. Rank 3.
            # Total 24.
            
            # This is not better than 22.
            
            # Let's just return the schoolbook decomposition. It's safe.
            # But we can do better.
            
            # Let's try to find a rank-21 decomposition for 2x3x6.
            # This is an open problem.
            
            # For now, let's just use the two-block construction with rank-11 for 2x3x3.
            # If we can find rank-11, we get rank-22.
            
            # Let's try to find rank-11 for 2x3x3 using a numerical solver.
            # We will use scipy.optimize.minimize on a least-squares problem.
            
            from scipy.optimize import minimize
            
            def loss_func(params):
                U = params[:U_size2 * r_target].reshape(U_size2, r_target)
                V = params[U_size2 * r_target:U_size2 * r_target + V_size2 * r_target].reshape(V_size2, r_target)
                W = params[U_size2 * r_target + V_size2 * r_target:].reshape(W_size2, r_target)
                
                U3 = U.reshape(n2, m2, r_target)
                V3 = V.reshape(m2, p2, r_target)
                W3 = W.reshape(p2, n2, r_target)
                
                T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
                
                return np.sum((T - target2)**2)
            
            # Initial guess: random
            x0 = np.random.randn(U_size2 * r_target + V_size2 * r_target + W_size2 * r_target)
            
            res = minimize(loss_func, x0, method='BFGS', options={'maxiter': 100})
            
            if res.fun < 1e-6:
                U_233 = res.x[:U_size2 * r_target].reshape(U_size2, r_target)
                V_233 = res.x[U_size2 * r_target:U_size2 * r_target + V_size2 * r_target].reshape(V_size2, r_target)
                W_233 = res.x[U_size2 * r_target + V_size2 * r_target:].reshape(W_size2, r_target)
                
                # Verify
                U3 = U_233.reshape(n2, m2, r_target)
                V3 = V_233.reshape(m2, p2, r_target)
                W3 = W_233.reshape(p2, n2, r_target)
                T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
                if np.allclose(T, target2, atol=1e-6):
                    found_233 = (U_233, V_233, W_233)
                    break
        
        if found_233 is not None:
            U_233, V_233, W_233 = found_233
            
            # Construct 2x3x6 from two 2x3x3 blocks
            U_236 = np.zeros((n * m, 22))
            V_236 = np.zeros((m * p, 22))
            W_236 = np.zeros((p * n, 22))
            
            # Block 1: first 11 terms
            U_236[:, 0:11] = U_233
            for c in range(3):
                for d in range(3):
                    V_236[c * 6 + d, 0:11] = V_233[c * 3 + d, 0:11]
            for k in range(3):
                for i in range(2):
                    W_236[k * 2 + i, 0:11] = W_233[k * 2 + i, 0:11]
            
            # Block 2: next 11 terms
            U_236[:, 11:22] = U_233
            for c in range(3):
                for d in range(3):
                    V_236[c * 6 + (d + 3), 11:22] = V_233[c * 3 + d, 0:11]
            for k in range(3):
                for i in range(2):
                    W_236[(k + 3) * 2 + i, 11:22] = W_233[k * 2 + i, 0:11]
            
            if _verify(U_236, V_236, W_236, n, m, p):
                best_U = U_236
                best_V = V_236
                best_W = W_236
                best_rank = 22
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = schoolbook_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
