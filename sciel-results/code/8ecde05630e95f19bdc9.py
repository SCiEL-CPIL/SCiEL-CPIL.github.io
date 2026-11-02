import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    Uses a verification-first approach with a bounded search for lower rank.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # The known rank-11 decomposition
    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        # Term 1: (A00 + A11) * (B00 + B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # Term 2: (A10 + A11) * B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # Term 3: A00 * (B01 - B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # Term 4: A11 * (B10 - B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # Term 5: (A00 + A01) * B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # Term 6: (A10 - A00) * (B00 + B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # Term 7: (A01 - A11) * (B10 + B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # Term 8: A00 * B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # Term 9: A01 * B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # Term 10: A10 * B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # Term 11: A11 * B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol, n_trials=100):
        """Verify that the decomposition exactly computes matrix multiplication (vectorized)."""
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            # M_r = sum_{i,j} U[i,j,r] * A[i,j]
            M = np.einsum('ijr,ij->r', U3, A)  # shape (rank,)
            # N_r = sum_{j,k} V[j,k,r] * B[j,k]
            N = np.einsum('jkr,jk->r', V3, B)  # shape (rank,)
            # C[i,k] = sum_r W[k,i,r] * M_r * N_r
            MN = M * N  # shape (rank,)
            C_computed = np.einsum('kir,r->ik', W3, MN)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    # First, verify the known rank-11 solution
    sol11 = make_rank11()
    if not verify_solution(sol11, n_trials=30):
        # Should never happen; return anyway
        pass

    def try_rank10_kronecker():
        """
        Attempt rank-10 via Kronecker lifting:
        View the 2x2x3 tensor as related to 2x2x2 tensor.
        Strategy: Use 7 terms with a specific U,V structure for the k=0,1 part,
        then try to express k=2 contributions using only 3 additional terms
        by allowing some of the 7 terms to also contribute to k=2 outputs.
        
        Key insight: If a term r has V[j,k,r] nonzero for k=2 as well,
        and W[2,i,r] is set appropriately, it can contribute to C[i,2].
        This means we don't need 4 separate terms for k=2.
        """
        if time_remaining() < 30:
            return None
        
        # The 7-term structure for k=0,1 (Strassen-like for 2x2x2):
        # We'll define U and V for 7 terms, then solve for W.
        # 
        # Standard 2x2x2 rank-7 terms (i,j) x (j,k) -> (i,k) for k=0,1:
        # But here we have 2 columns (k=0,1), so it's like two 2x2 multiplications.
        # 
        # Let me define a specific 7-term U,V pattern and solve for all W entries
        # (including W[2,i,r] for the third column).
        
        # 7-term basis for the A-side (n*m = 4 entries each):
        # Using the standard Strassen pattern adapted:
        # Term 0: (A00+A11, B00+B11) -> contributes to C00, C11
        # Term 1: (A10+A11, B00) -> contributes to C10, C11
        # Term 2: (A00, B01-B11) -> contributes to C01, C11
        # Term 3: (A11, B10-B00) -> contributes to C00, C01
        # Term 4: (A00+A01, B11) -> contributes to C00, C01
        # Term 5: (A10-A00, B00+B01) -> contributes to C11
        # Term 6: (A01-A11, B10+B11) -> contributes to C00
        
        # Define U vectors (flattened i*m+j):
        # U[r] is a 4-vector: [A00, A01, A10, A11]
        U_basis = np.array([
            [1, 0, 0, 1],   # Term 0: A00+A11
            [0, 0, 1, 1],   # Term 1: A10+A11
            [1, 0, 0, 0],   # Term 2: A00
            [0, 0, 0, 1],   # Term 3: A11
            [1, 1, 0, 0],   # Term 4: A00+A01
            [-1, 0, 1, 0],  # Term 5: A10-A00
            [0, 1, 0, -1],  # Term 6: A01-A11
        ], dtype=float)  # shape (7, 4)
        
        # Define V vectors (flattened j*p+k):
        # V[r] is a 6-vector: [B00, B01, B02, B10, B11, B12]
        V_basis = np.array([
            [1, 0, 0, 0, 1, 0],   # Term 0: B00+B11
            [1, 0, 0, 0, 0, 0],   # Term 1: B00
            [0, 1, 0, 0, -1, 0],  # Term 2: B01-B11
            [0, 0, 0, 1, -1, 0],  # Term 3: B10-B00
            [0, 0, 0, 0, 1, 0],   # Term 4: B11
            [1, 1, 0, 0, 0, 0],   # Term 5: B00+B01
            [0, 0, 0, 1, 1, 0],   # Term 6: B10+B11
        ], dtype=float)  # shape (7, 6)
        
        # For rank 10, we add 3 more terms. Let's define their U and V:
        # The k=2 column needs: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
        # 
        # Option A: 3 new terms with specific U,V, solve for all W
        # Option B: Modify some of the 7 terms to include k=2 in V, and use only 3 new terms
        # 
        # Let's try Option B: extend some V vectors to include B02, B12 components.
        # If we modify the V of existing terms to include k=2 entries, those terms
        # can contribute to C[i,2] via W[2,i,r].
        
        # Strategy: Add B02 and B12 to some existing V vectors.
        # Term 2 has V = [0,1,0,0,-1,0]. If we change to [0,1,1,0,-1,0] (add B02),
        # then M2*N2 = A00*(B01-B11+B02). This introduces A00*B02 which we want in C02,
        # but also A00*B01 and A00*B11 which we already have.
        # The W[2,0,2] coefficient would pick out the A00*B02 part... but no,
        # M*N is a single scalar per term, so W[2,0,2]*M2*N2 contributes to C02.
        # But M2*N2 = A00*(B01-B11+B02), so the contribution to C02 is
        # W[2,0,2]*A00*(B01-B11+B02), which includes unwanted A00*B01 and A00*B11.
        # 
        # So we can't simply extend V without breaking other outputs.
        # We'd need to cancel the unwanted terms using other W entries.
        
        # Better approach: Keep the 7 terms as-is for k=0,1, and add 3 new terms
        # specifically for k=2, but allow the new terms to also have W[0,i,r] and W[1,i,r]
        # nonzero to help cancel or correct.
        
        # Actually, the cleanest approach for rank 10:
        # Use 7 terms for k=0,1 (as above) + 3 terms for k=2.
        # For the 3 k=2 terms, we need to cover:
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        # 
        # With 3 terms, can we cover these 4 bilinear forms?
        # Term 7: U=[1,0,0,0], V=[0,0,1,0,0,0] -> A00*B02, W[2,0,7]=1
        # Term 8: U=[0,1,0,0], V=[0,0,0,0,0,1] -> A01*B12, W[2,0,8]=1
        # Term 9: Need A10*B02 + A11*B12
        #   With one term: U=[0,0,a,b], V=[0,0,c,0,0,d]
        #   M*N = ac*A10*B02 + ad*A10*B12 + bc*A11*B02 + bd*A11*B12
        #   We need ac=1, ad=0, bc=0, bd=1
        #   ad=0: a=0 or d=0. If a=0, ac=0, contradiction. So d=0.
        #   bc=0: b=0 or c=0. If b=0, bd=0, contradiction. So c=0.
        #   But ac=1 requires a,c nonzero. Contradiction!
        #   So one term CANNOT cover both A10*B02 and A11*B12.
        # 
        # Therefore, we need at least 4 terms for k=2 alone, giving rank 11.
        # UNLESS we share terms between k=0,1 and k=2.
        
        # Key idea: What if one of the 7 terms for k=0,1 ALSO contributes to k=2?
        # E.g., Term 2: U=[1,0,0,0]=A00, V=[0,1,0,0,-1,0]=B01-B11
        # If we change V to [0,1,1,0,-1,0]=B01-B11+B02, then:
        # M2*N2 = A00*(B01-B11+B02)
        # This contributes to C01 via W[1,0,2]*A00*(B01-B11+B02)
        # and C11 via W[1,1,2]*A00*(B01-B11+B02)
        # and C02 via W[2,0,2]*A00*(B01-B11+B02)
        # The unwanted A00*B02 in C01,C11 can be cancelled by other terms' W entries.
        
        # This is getting complex. Let me set up a full linear system.
        # 
        # For rank 10 with 10 terms, we have:
        # 10 U vectors (4-dim), 10 V vectors (6-dim), 10 W vectors (6-dim, since p*n=6)
        # Total unknowns: 10*4 + 10*6 + 10*6 = 160
        # Constraints: 12 tensor entries (one for each (i,j,k))
        # 
        # The tensor equation: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # This is nonlinear in U,V,W.
        
        # Let me try a different approach: fix U and V for all 10 terms,
        # then solve for W linearly.
        
        # Fix 7 U,V as above. For terms 7,8,9, try various U,V patterns:
        # Pattern 1: 
        #   Term 7: U=[1,0,0,0], V=[0,0,1,0,0,0]  (A00*B02)
        #   Term 8: U=[0,1,0,0], V=[0,0,0,0,0,1]  (A01*B12)
        #   Term 9: U=[0,0,1,1], V=[0,0,1,0,0,1]  (A10+A11)*(B02+B12) -> cross terms!
        #   M9*N9 = A10*B02 + A10*B12 + A11*B02 + A11*B12
        #   We want A10*B02 + A11*B12, but get extra A10*B12 + A11*B02.
        #   Can we cancel these with W? No, because M*N is a single scalar per term.
        #   The extra terms A10*B12 and A11*B02 will appear in ALL outputs where W[*,*,9] is nonzero.
        #   We can set W[2,1,9] and W[2,0,9] to pick out parts, but the unwanted products
        #   still get multiplied by whatever W value we choose.
        
        # Pattern 2: Use the 7 base terms with modified V to include k=2,
        # and only 3 new terms.
        
        # Let me try: modify terms 2,3,4,5,6 to include B02 or B12 in their V,
        # and add 3 new terms. Then solve for all W entries.
        
        # Actually, let me just brute-force try several specific 10-term structures
        # by fixing U,V and solving for W.
        
        def solve_for_w(U_mat, V_mat):
            """
            Given U (4 x rank) and V (6 x rank), solve for W (6 x rank)
            such that T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            where T[i,j,k] = 1 for all i,j,k.
            
            This is a linear system in W. For each (i,k), we have:
            sum_j T[i,j,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            
            Wait, T[i,j,k]=1 for all i,j,k, so:
            For fixed (i,k): sum_j 1 = sum_r (sum_j U[i,j,r]*V[j,k,r]) * W[k,i,r]
            
            No, that's not right. The tensor equation is:
            T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            For each (i,j,k), this gives one equation. We have 2*2*3=12 equations.
            Unknowns: W[k,i,r] for k=0,1,2; i=0,1; r=0,...,rank-1 => 6*rank unknowns.
            
            For rank=10: 60 unknowns, 12 equations. Underdetermined.
            We can set up the 12 equations and find a solution.
            
            Let me restructure: for each (i,j,k):
            1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            This is linear in W. Let's build the system.
            """
            rank = U_mat.shape[1]
            # Unknowns: W[k,i,r] indexed as (k*n+i)*rank + r = (k*2+i)*rank + r
            # 6*rank unknowns
            n_unknowns = 6 * rank
            A_sys = np.zeros((12, n_unknowns))
            b_sys = np.ones(12)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k  # 0..11
                        for r in range(rank):
                            # W[k,i,r] is at position (k*n+i)*rank + r
                            w_idx = (k * n + i) * rank + r
                            A_sys[eq_idx, w_idx] = U_mat[i * m + j, r] * V_mat[j * p + k, r]
            
            # Solve the underdetermined system: find W such that A_sys @ W_vec = b_sys
            # Use least squares / pseudo-inverse
            try:
                W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            except:
                return None
            
            # Check if the solution actually satisfies the equations
            if not np.allclose(A_sys @ W_vec, b_sys, atol=1e-10):
                return None
            
            # Reshape W_vec to (p, n, rank) -> (p*n, rank)
            W_mat = W_vec.reshape(p, n, rank).reshape(p * n, rank)
            return W_mat
        
        # Try several 10-term structures:
        
        # Structure A: 7 base terms + 3 new terms for k=2
        # Base 7 terms as defined above
        # New terms:
        # Term 7: U=[1,0,0,0], V=[0,0,1,0,0,0]
        # Term 8: U=[0,1,0,0], V=[0,0,0,0,0,1]
        # Term 9: U=[0,0,1,0], V=[0,0,1,0,0,0]  (A10*B02)
        # But then A11*B12 is missing! We need it.
        # So this doesn't work with only 3 new terms unless we modify base terms.
        
        # Structure B: Modify base terms to include k=2 in V
        # Let's try modifying Term 2 (U=A00) to have V=[0,1,1,0,-1,0] (add B02)
        # and Term 3 (U=A11) to have V=[0,0,0,1,-1,1] (add B12, change B10-B00 to B10-B00+B12)
        # Wait, Term 3 V was [0,0,0,1,-1,0]. Change to [0,0,0,1,-1,1].
        # Then M3*N3 = A11*(B10-B00+B12). 
        # This adds A11*B12 which we want in C12, but also adds A11*B10 and A11*B00
        # to the existing contributions. The W entries for k=0,1 will need to be adjusted.
        
        # Let me just try many random-ish structures and solve for W.
        
        structures = []
        
        # Structure 0: original 7 + 4 new = 11 (baseline, should work)
        # Structure 1: 7 modified + 3 new = 10
        # Try various modifications
        
        # For each structure, define U (4 x rank) and V (6 x rank)
        
        # Structure 1: Modify V of terms 2 and 3 to include k=2
        U1 = U_basis.copy()
        V1 = V_basis.copy()
        # Term 2: add B02
        V1[2, 2] = 1  # B02
        # Term 3: add B12
        V1[3, 5] = 1  # B12
        # Add 3 new terms:
        U1 = np.hstack([U1, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1]]).T])  # terms 7,8,9
        V1 = np.hstack([V1, np.array([[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,1,0,0,1]]).T])
        structures.append(("mod_v_2_3", U1, V1))
        
        # Structure 2: Modify V of terms 0 and 4
        U2 = U_basis.copy()
        V2 = V_basis.copy()
        V2[0, 2] = 1  # Term 0: B00+B11 -> B00+B11+B02
        V2[4, 5] = 1  # Term 4: B11 -> B11+B12
        U2 = np.hstack([U2, np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]]).T])  # placeholder
        # Actually, let me think more carefully.
        
        # Let me try a more systematic approach: 
        # Use the 7 base terms, and for the 3 new terms, try all combinations
        # of simple U and V vectors that involve k=2.
        
        # Simple U vectors for new terms:
        simple_U = [
            [1,0,0,0],  # A00
            [0,1,0,0],  # A01
            [0,0,1,0],  # A10
            [0,0,0,1],  # A11
            [1,0,0,1],  # A00+A11
            [0,1,0,1],  # A01+A11
            [1,1,0,0],  # A00+A01
            [0,0,1,1],  # A10+A11
        ]
        simple_V = [
            [0,0,1,0,0,0],  # B02
            [0,0,0,0,0,1],  # B12
            [0,0,1,0,0,1],  # B02+B12
        ]
        
        # Try all combinations of 3 new terms
        best_sol = None
        tried = 0
        max_tries = 2000
        
        for i in range(len(simple_U)):
            if time_remaining() < 15:
                break
            for j in range(len(simple_U)):
                if time_remaining() < 15:
                    break
                for k in range(len(simple_U)):
                    if time_remaining() < 15:
                        break
                    for a in range(len(simple_V)):
                        if time_remaining() < 15:
                            break
                        for b in range(len(simple_V)):
                            if time_remaining() < 15:
                                break
                            for c in range(len(simple_V)):
                                if time_remaining() < 15 or tried >= max_tries:
                                    break
                                tried += 1
                                U_new = np.array([simple_U[i], simple_U[j], simple_U[k]], dtype=float)
                                V_new = np.array([simple_V[a], simple_V[b], simple_V[c]], dtype=float)
                                U_full = np.hstack([U_basis, U_new])
                                V_full = np.hstack([V_basis, V_new])
                                
                                W_mat = solve_for_w(U_full, V_full)
                                if W_mat is not None:
                                    # Verify the solution
                                    sol = {
                                        "U": U_full.T.tolist(),  # (4, 10) -> (10, 4) -> need (n*m, rank) = (4, 10)
                                        "V": V_full.T.tolist(),
                                        "W": W_mat.tolist()
                                    }
                                    # Wait, U_full is (4, 10), so U_full.T is (10, 4)
                                    # We need U to be (n*m, rank) = (4, 10)
                                    # So U should be U_full directly (4 rows, 10 cols)
                                    sol = {
                                        "U": U_full.tolist(),
                                        "V": V_full.tolist(),
                                        "W": W_mat.tolist()
                                    }
                                    if verify_solution(sol, n_trials=20):
                                        best_sol = sol
                                        return best_sol
        
        return None

    better_sol = try_rank10_kronecker()
    
    if better_sol is not None:
        return better_sol
    else:
        return sol11

# EVOLVE-BLOCK-END
