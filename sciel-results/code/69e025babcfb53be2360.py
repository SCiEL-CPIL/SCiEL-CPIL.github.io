"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Rank-84 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Strassen's 7-multiplication algorithm on 2x2 blocks.
    Then attempts to reduce rank by finding linear dependencies.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def strassen_2x2_2x2(oi, oj, ok):
        """
        Add 7 Strassen terms for 2x2*2x2 product.
        oi: offset row in A/C (0 or 2)
        oj: offset col in A / row in B (0 or 2)  
        ok: offset col in B/C (0, 2, or 4)
        """
        terms = [
            # m1
            ([(oi, oj, 1), (oi+1, oj+1, 1)],
             [(oj, ok, 1), (oj+1, ok+1, 1)],
             [(oi, ok, 1), (oi+1, ok+1, 1)]),
            # m2
            ([(oi+1, oj, 1), (oi+1, oj+1, 1)],
             [(oj, ok, 1)],
             [(oi+1, ok, 1), (oi+1, ok+1, -1)]),
            # m3
            ([(oi, oj, 1)],
             [(oj, ok+1, 1), (oj+1, ok+1, -1)],
             [(oi, ok+1, 1), (oi+1, ok+1, 1)]),
            # m4
            ([(oi+1, oj+1, 1)],
             [(oj+1, ok, 1), (oj, ok, -1)],
             [(oi, ok, 1), (oi+1, ok, 1)]),
            # m5
            ([(oi, oj, 1), (oi, oj+1, 1)],
             [(oj+1, ok+1, 1)],
             [(oi, ok, -1), (oi, ok+1, 1)]),
            # m6
            ([(oi+1, oj, 1), (oi, oj, -1)],
             [(oj, ok, 1), (oj, ok+1, 1)],
             [(oi+1, ok+1, 1)]),
            # m7
            ([(oi, oj+1, 1), (oi+1, oj+1, -1)],
             [(oj+1, ok, 1), (oj+1, ok+1, 1)],
             [(oi, ok, 1)]),
        ]
        
        for u_c, v_c, w_c in terms:
            uc = [0.0] * (n * m)
            vc = [0.0] * (m * p)
            wc = [0.0] * (p * n)
            
            for (i, j, c) in u_c:
                uc[i * m + j] = c
            for (j, k, c) in v_c:
                vc[j * p + k] = c
            for (i, k, c) in w_c:
                wc[k * n + i] = c
            
            U_cols.append(uc)
            V_cols.append(vc)
            W_cols.append(wc)
    
    # For each output block position (rb, cb):
    # C[rb][cb] = A[rb][0]*B[0][cb] + A[rb][1]*B[1][cb]
    for rb in range(2):
        for cb in range(3):
            if _time_remaining() < 5.0:
                break
            strassen_2x2_2x2(rb * 2, 0, cb * 2)
            strassen_2x2_2x2(rb * 2, 2, cb * 2)
    
    rank = len(U_cols)
    
    # Build initial matrices
    U_mat = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_mat = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_mat = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Attempt rank reduction by finding linear dependencies
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if some columns are linearly dependent in a way that
    # allows us to eliminate them while preserving the tensor.
    
    if _time_remaining() > 20.0:
        try:
            import numpy as np
            
            # Convert to numpy for efficient computation
            U_np = np.array(U_mat).T  # shape (rank, n*m)
            V_np = np.array(V_mat).T  # shape (rank, m*p)
            W_np = np.array(W_mat).T  # shape (rank, p*n)
            
            # Verify the decomposition is correct
            # T[i,j,k] should equal 1 for all valid (i,j,k)
            # Let's build the full tensor and verify
            n, m, p = 4, 4, 6
            T_computed = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] = sum(
                            U_np[r, i*m+j] * V_np[r, j*p+k] * W_np[r, k*n+i]
                            for r in range(rank)
                        )
            
            # T[i,j,k] should be 1 for all i,j,k
            if np.allclose(T_computed, 1.0, atol=1e-9):
                # Now try to find linear dependencies
                # If sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0, we can eliminate one term
                # This is equivalent to checking if the rank-1 tensors are linearly independent
                
                # Try a simpler approach: check if any two rank-1 tensors are proportional
                # or if there's a linear combination that gives zero
                # For now, just verify correctness and return rank-84
                
                pass
            
            # Try to reduce rank by checking for duplicate or proportional columns
            # This is a heuristic - in general, finding exact rank is NP-hard
            # But for structured decompositions, we might find reductions
            
            # Check for exactly duplicate (U,V,W) triples
            seen = set()
            unique_indices = []
            for r in range(rank):
                key = (tuple(U_np[r]), tuple(V_np[r]), tuple(W_np[r]))
                if key not in seen:
                    seen.add(key)
                    unique_indices.append(r)
            
            if len(unique_indices) < rank:
                # Found duplicates, remove them
                U_mat = [[U_cols[r][i] for r in unique_indices] for i in range(n * m)]
                V_mat = [[V_cols[r][i] for r in unique_indices] for i in range(m * p)]
                W_mat = [[W_cols[r][i] for r in unique_indices] for i in range(p * n)]
                rank = len(unique_indices)
        
        except Exception:
            # If anything goes wrong, return the original rank-84 decomposition
            pass
    
    # Final safety check: verify the decomposition
    if _time_remaining() > 5.0:
        try:
            import numpy as np
            U_np = np.array(U_mat).T
            V_np = np.array(V_mat).T
            W_np = np.array(W_mat).T
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i, j, k] = sum(
                            U_np[r, i*m+j] * V_np[r, j*p+k] * W_np[r, k*n+i]
                            for r in range(rank)
                        )
            if not np.allclose(T_check, 1.0, atol=1e-9):
                # Fallback to original if verification fails
                U_mat = [[U_cols[r][i] for r in range(84)] for i in range(n * m)]
                V_mat = [[V_cols[r][i] for r in range(84)] for i in range(m * p)]
                W_mat = [[W_cols[r][i] for r in range(84)] for i in range(p * n)]
        except Exception:
            pass
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
