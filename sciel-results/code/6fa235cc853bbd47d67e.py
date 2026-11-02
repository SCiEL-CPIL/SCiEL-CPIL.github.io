import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        rank = len(W[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check dimensions
        if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
            return False
            
        # Verify tensor decomposition
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        lA = np.dot(U_np[i*m:(i+1)*m, r], A[i, :])
                        lB = np.dot(V_np[k::p, r], B[:, k])
                        C_computed[i, k] += W_np[k*n+i, r] * lA * lB
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def get_rank_11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def search_rank_10():
        """
        Search for a rank-10 decomposition using parameterized structure.
        
        The 2x2x3 multiplication can be viewed as:
        - 2x2x2 part (rank 7 via Strassen)
        - Plus contributions for the third output column
        
        We parameterize the rank-10 solution by allowing the Strassen factors
        to also contribute to the third column, and adjusting the trivial terms.
        
        Key idea: In the rank-11 solution, terms 8-11 are "trivial" (single
        products). We try to absorb their contributions into the Strassen
        structure by modifying the U, V, W coefficients.
        """
        if time.time() > deadline - 5:
            return None
        
        # Parameterized rank-10 search
        # We use the Strassen structure as a base and add free parameters
        # that allow the "trivial" terms to be absorbed.
        
        # The tensor entries for 2x2x3 are:
        # C[0,0] = A00*B00 + A01*B10
        # C[0,1] = A00*B01 + A01*B11
        # C[0,2] = A00*B02 + A01*B12
        # C[1,0] = A10*B00 + A11*B10
        # C[1,1] = A10*B01 + A11*B11
        # C[1,2] = A10*B02 + A11*B12
        
        # We'll try a systematic search over a reduced parameter space.
        # Start from the rank-11 solution and try to merge two terms.
        
        # Approach: Try all pairs of the 4 "trivial" terms (8,9,10,11)
        # and see if they can be merged with one of the Strassen terms.
        
        # For a more effective search, let's use a gradient-based approach
        # with a smooth loss function.
        
        rank = 10
        rng = np.random.default_rng(123)
        
        # Initialize from rank-11 solution, dropping one term
        base = get_rank_11()
        U11 = np.array(base["U"], dtype=float)
        V11 = np.array(base["V"], dtype=float)
        W11 = np.array(base["W"], dtype=float)
        
        # Try dropping each of the 4 trivial terms and see if we can adjust
        # the remaining 10 terms to compensate.
        
        for drop_idx in [7, 8, 9, 10]:  # 0-indexed: terms 8,9,10,11
            if time.time() > deadline - 5:
                return None
            
            # Start with rank-11 minus the dropped term
            U = U11[:, [r for r in range(11) if r != drop_idx]].copy()
            V = V11[:, [r for r in range(11) if r != drop_idx]].copy()
            W = W11[:, [r for r in range(11) if r != drop_idx]].copy()
            
            # Now try to optimize the remaining 10 terms
            # Use a simple iterative refinement
            
            def compute_loss(U, V, W):
                loss = 0.0
                for _ in range(5):
                    A = rng.standard_normal((n, m))
                    B = rng.standard_normal((m, p))
                    C_expected = A @ B
                    C_computed = np.zeros((n, p))
                    
                    for i in range(n):
                        for k in range(p):
                            for r in range(rank):
                                lA = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                                lB = np.dot(V[k::p, r], B[:, k])
                                C_computed[i, k] += W[k*n+i, r] * lA * lB
                    
                    loss += np.sum((C_computed - C_expected)**2)
                return loss / 5.0
            
            # Simple gradient-free search: try small perturbations
            best_loss = compute_loss(U, V, W)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try random perturbations
            for _ in range(50):
                if time.time() > deadline - 5:
                    return None
                
                U_pert = U.copy()
                V_pert = V.copy()
                W_pert = W.copy()
                
                # Perturb randomly
                for r in range(rank):
                    if np.random.random() < 0.5:
                        U_pert[:, r] += rng.standard_normal(n*m) * 0.1
                    if np.random.random() < 0.5:
                        V_pert[:, r] += rng.standard_normal(m*p) * 0.1
                    if np.random.random() < 0.5:
                        W_pert[:, r] += rng.standard_normal(p*n) * 0.1
                
                loss = compute_loss(U_pert, V_pert, W_pert)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U_pert.copy(), V_pert.copy(), W_pert.copy()
            
            # Check if we found a valid solution
            if best_loss < 1e-8:
                # Round to nearest half-integer
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
                if verify(result["U"], result["V"], result["W"]):
                    return result
        
        return None

    # First try to find a rank-10 solution
    rank10 = search_rank_10()
    if rank10 is not None:
        return rank10
    
    # Fall back to rank-11
    return get_rank_11()
# EVOLVE-BLOCK-END
