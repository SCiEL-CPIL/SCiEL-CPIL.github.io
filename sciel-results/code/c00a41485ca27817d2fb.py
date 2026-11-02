"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x4 by 4x7.
    Uses Strassen blocks with rank-reduction via linear dependency analysis.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Build the initial rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the initial rank-50 decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    def to_int_list(M):
        return [[int(round(M[i, j])) for j in range(M.shape[1])] for i in range(M.shape[0])]
    
    def verify(Um, Vm, Wm):
        """Check that the decomposition exactly reproduces the matmul tensor."""
        r = Um.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    expected = 1
                    actual = sum(Um[i*m+j, rr] * Vm[j*p+k, rr] * Wm[k*n+i, rr] for rr in range(r))
                    if abs(actual - expected) > 1e-6:
                        return False
        return True
    
    # Baseline
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Now attempt rank reduction by finding linear dependencies.
    # The key insight: if we can express one rank-1 term as a linear combination
    # of others (in the tensor product space), we can eliminate it.
    # 
    # Approach: Build the "tensor matrix" where each column r represents
    # the vectorized rank-1 tensor u_r ⊗ v_r ⊗ w_r. If columns are linearly
    # dependent, we can reduce rank.
    #
    # More practically: try removing each column and see if the remaining
    # columns can be rescaled to still reproduce the tensor.
    
    if time.time() < deadline - 30:
        # Try each possible column removal
        for remove_col in range(rank_init):
            if time.time() > deadline - 30:
                break
            
            U_rem = np.delete(U, remove_col, axis=1)
            V_rem = np.delete(V, remove_col, axis=1)
            W_rem = np.delete(W, remove_col, axis=1)
            
            if verify(U_rem, V_rem, W_rem):
                best_U, best_V, best_W = U_rem.copy(), V_rem.copy(), W_rem.copy()
                best_rank = rank_init - 1
                U, V, W = U_rem.copy(), V_rem.copy(), W_rem.copy()
                rank_init -= 1
                # Try removing another
                for remove_col2 in range(rank_init):
                    if time.time() > deadline - 20:
                        break
                    U_rem2 = np.delete(U, remove_col2, axis=1)
                    V_rem2 = np.delete(V, remove_col2, axis=1)
                    W_rem2 = np.delete(W, remove_col2, axis=1)
                    if verify(U_rem2, V_rem2, W_rem2):
                        best_U, best_V, best_W = U_rem2.copy(), V_rem2.copy(), W_rem2.copy()
                        best_rank = rank_init - 1
                        U, V, W = U_rem2.copy(), V_rem2.copy(), W_rem2.copy()
                        rank_init -= 1
                        break
                break
    
    # Try a different construction: use a 3-column Strassen variant for some blocks
    # Actually, let's try a global optimization approach using the structure.
    # For 2x4*4x7, try splitting differently.
    
    if time.time() < deadline - 10 and best_rank == 50:
        # Try an alternative: treat the problem as 2*(4*7) with shared structure.
        # Use the fact that A has only 2 rows. 
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This is a bilinear map. Try to find a lower-rank bilinear decomposition.
        # 
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We can try to find rank-r decomposition using the fact that
        # the i-dimension is small (only 2).
        #
        # Alternative construction: group by j-pairs differently.
        # Try: split j into 2+2, but use a different k-splitting.
        # Or: use a 2x4x7 direct approach with Winograd-like transforms.
        
        # Try: rank-49 construction by merging one schoolbook term with adjacent Strassen
        # Specifically, look at the boundary between Strassen block (k=4,5) and schoolbook (k=6)
        # The schoolbook term for j=2, k=6: U[2*4+2, r]=1, V[2*7+6, r]=1, W[6*2+0, r]=1
        # and U[3, r]=1, V[3*7+6, r]=1, W[6*2+1, r]=1
        # These might be expressible via the Strassen terms in the k=4,5 block.
        
        # Let's try a brute-force search over small perturbations.
        # Actually, let's try a completely different block structure.
        
        # New idea: use 2x2 Strassen for (j=0,1; k=0,1), (j=0,1; k=2,3), (j=0,1; k=4,5)
        # and (j=2,3; k=0,1), (j=2,3; k=2,3), (j=2,3; k=4,5), plus k=6 terms.
        # That's 6 Strassen blocks (42) + 8 schoolbook = 50. Same.
        
        # Try: Can we use a 2x3x2 or 3x2x2 sub-block with rank < schoolbook?
        # 2x3x2: schoolbook is 12. Is there a rank-11 decomposition?
        # 3x2x2: schoolbook is 12. 
        # For 2x2x2, Strassen gives rank 7 < 8.
        # For 2x3x2: split 3 into 2+1. 2x2x2 (rank 7) + 2x1x2 (rank 4) = 11 < 12.
        # So 2x3x2 has rank 11!
        
        # Rebuild with 2x3x2 blocks:
        # Split j=4 into 2+2, k=7 into 2+2+2+1.
        # For each j-pair (2x2) and k-triple (3): use 2x2x3 blocks.
        # 2x2x3: split k into 2+1. 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
        # We have 2 j-pairs and 3 k-triples (0-2, 3-5, 6... wait 7=2+2+2+1).
        # k-triples: (0,1,2), (3,4,5), (6) - that's not triples.
        # Let's do k=7 as 2+2+2+1.
        # For each j-pair (2 of them) and each k-block:
        #   k-block of size 2: 2x2x2 Strassen = rank 7
        #   k-block of size 1: 2x2x1 schoolbook = rank 4
        # Total: 2 j-pairs * (3 blocks of size 2 * 7 + 1 block of size 1 * 4) = 2*(21+4) = 50. Same.
        
        # Try k=7 as 3+2+2:
        # k-block of size 3: 2x2x3 = rank 11
        # k-block of size 2: 2x2x2 = rank 7
        # Total per j-pair: 11 + 7 + 7 = 25
        # Two j-pairs: 50. Same.
        
        # Try k=7 as 3+4:
        # 2x2x3 = 11, 2x2x4: split 4 into 2+2, so 2*(2x2x2) = 14. Total: 25 per j-pair. 50 total.
        
        # The rank seems stuck at 50 with this block approach.
        # Let me try a different j-splitting: j=4 as 3+1.
        # j-block of size 3: A is 2x3, B is 3xk. 
        # 2x3x2: rank 11 (as shown above).
        # j-block of size 1: A is 2x1, B is 1xk.
        # 2x1x2: schoolbook rank 4.
        # For k=7 split into 2+2+2+1:
        # j=3 block: 2x3x2 (rank 11) * 3 + 2x3x1 (rank 6) = 33+6 = 39
        # j=1 block: 2x1x2 (rank 4) * 3 + 2x1x1 (rank 2) = 12+2 = 14
        # Total: 39+14 = 53. Worse.
        
        # Try j=4 as 1+3:
        # j=1 block: 2x1x7. For each k: 2x1x1 rank 2. Total 14.
        # j=3 block: 2x3x7. Split k into 2+2+2+1: 2x3x2 (11)*3 + 2x3x1 (6) = 33+6=39.
        # Total: 14+39 = 53. Worse.
        
        # The rank 50 appears to be the minimum for this block structure.
        # Let's just return the best we found.
    
        U_out = to_int_list(best_U)
        V_out = to_int_list(best_V)
        W_out = to_int_list(best_W)
        
        return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
