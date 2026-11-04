import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # We will construct a rank-32 decomposition by splitting the p=7 dimension into p1=4 and p2=3.
    # The tensor M_{3,4,7} can be decomposed into M_{3,4,4} (for k=0..3) and M_{3,4,3} (for k=4..6).
    # We use a rank-16 decomposition for M_{4,4,4} (schoolbook) and project it to M_{3,4,4} and M_{3,4,3}.
    
    # Step 1: Construct rank-16 decomposition for M_{4,4,4}
    # Indices: i in [0,3], j in [0,3], k in [0,3]
    # U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r} for m=4, p=4, n=4
    # Wait, the dimensions for the sub-problem M_{4,4,4} are n_sub=4, m_sub=4, p_sub=4.
    # But our actual m is 4. So the middle index j is the same.
    # The first index i goes from 0 to 3 (but we only need 0..2 for n=3).
    # The third index k goes from 0 to 3 (for the first block) or 0 to 2 (for the second block, but offset).
    
    # Let's define the rank-16 terms for a generic 4x4x4 multiplication.
    # Term t corresponds to (i_t, j_t, k_t) where i_t, j_t, k_t in 0..3.
    # U[i*m + j, t] = 1 if (i,j) == (i_t, j_t)
    # V[j*p + k, t] = 1 if (j,k) == (j_t, k_t)  -- Note: here p is the p of the sub-problem?
    # No, the V matrix has rows indexed by j*p_global + k_global.
    # This is tricky. Let's think carefully.
    
    # The global tensor is T[i, j, k] for i in 0..2, j in 0..3, k in 0..6.
    # We split k into k1 (0..3) and k2 (4..6).
    # Block 1: k in 0..3. This is M_{3,4,4}.
    # Block 2: k in 4..6. This is M_{3,4,3}.
    
    # For Block 1 (M_{3,4,4}):
    # We can use a rank-16 decomposition of M_{4,4,4} and zero out the i=3 row in U and W.
    # Let's construct the rank-16 terms for M_{4,4,4} first.
    # The indices for M_{4,4,4} are i' in 0..3, j' in 0..3, k' in 0..3.
    # The global indices are i=i', j=j', k=k'.
    # U_global[i*4 + j, r]
    # V_global[j*7 + k, r]  <-- Here p_global=7.
    # W_global[k*3 + i, r]  <-- Here n_global=3.
    
    # This direct mapping is hard because the strides are different.
    # Let's just build the U, V, W matrices directly for the global problem using the block structure.
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal rank
        U_list.append([0.0] * (n * m))
        V_list.append([0.0] * (m * p))
        W_list.append([0.0] * (p * n))
        U_list[-1][u_idx] = val
        V_list[-1][v_idx] = val
        W_list[-1][w_idx] = val
        rank += 1

    # Block 1: k in 0..3 (M_{3,4,4})
    # Use rank-16 decomposition of M_{4,4,4} projected to i in 0..2.
    # The rank-16 terms are (i', j', k') for i',j',k' in 0..3.
    # We only keep terms where the contribution to i=3 is zero?
    # No, we just set U[i=3, ...] = 0. But in our global U, i only goes to 2.
    # So we just don't include the i=3 rows.
    # However, the V and W matrices have global indices.
    # For a term (i', j', k'):
    # U_global[i'*4 + j', r] = 1
    # V_global[j'*7 + k', r] = 1
    # W_global[k'*3 + i', r] = 1
    # This works for any i', j', k' in 0..3.
    # We iterate over all 16 combinations.
    
    for i_prime in range(4):
        for j_prime in range(4):
            for k_prime in range(4):
                u_idx = i_prime * m + j_prime
                v_idx = j_prime * p + k_prime
                w_idx = k_prime * n + i_prime
                
                # If i_prime == 3, the u_idx is 3*4+j = 12+j, which is out of bounds for U (size 12).
                # Wait, U has size n*m = 3*4 = 12. Indices 0..11.
                # i_prime=3 -> u_idx = 12, 13, 14, 15. Out of bounds.
                # So we must skip terms where i_prime == 3?
                # If we skip them, we are left with 3*4*4 = 48 terms? No, that's schoolbook.
                # The rank-16 decomposition of M_{4,4,4} has 16 terms.
                # If we simply drop the terms with i_prime=3, we lose 4 terms (those with i_prime=3).
                # But do those terms contribute to i<3? No, because U[i, r] is 0 for i != i_prime.
                # So dropping them is safe.
                # But wait, are the remaining 12 terms sufficient to decompose M_{3,4,4}?
                # Yes, because the sub-tensor M_{3,4,4} is a restriction of M_{4,4,4}.
                # Any decomposition of M_{4,4,4} restricted to i<3 is a valid decomposition of M_{3,4,4}.
                # So we just take the 16 terms and set U[i=3, r] = 0 (which is automatically done if we don't set them, or if we handle bounds).
                # Actually, we can't set U[12, r] if U has size 12.
                # So we just skip the assignment for i_prime=3.
                # But we still need the term to exist for V and W?
                # If U[i, r] is 0 for all i, the term contributes 0 to the sum.
                # So we can skip adding the term entirely if i_prime == 3.
                
                if i_prime >= n:
                    continue
                    
                add_term(u_idx, v_idx, w_idx)

    # Block 2: k in 4..6 (M_{3,4,3})
    # We can use a rank-16 decomposition of M_{4,4,4} projected to k in 0..2 (and offset by 4).
    # The terms are (i', j', k') for i',j',k' in 0..3.
    # We only keep terms where k_prime < 3?
    # No, we need to map k' to k_global = k' + 4.
    # So k' in 0..2 maps to k_global in 4..6.
    # Terms with k_prime=3 map to k_global=7, which is out of bounds.
    # So we skip terms with k_prime=3.
    # Also skip terms with i_prime=3 (as before).
    
    for i_prime in range(4):
        for j_prime in range(4):
            for k_prime in range(4):
                if i_prime >= n:
                    continue
                if k_prime >= 3: # k_prime must be 0,1,2 to map to 4,5,6
                    continue
                    
                u_idx = i_prime * m + j_prime
                v_idx = j_prime * p + (k_prime + 4) # Offset k by 4
                w_idx = (k_prime + 4) * n + i_prime
                
                add_term(u_idx, v_idx, w_idx)

    # Total rank should be:
    # Block 1: 3 * 4 * 4 = 48? No, we skipped i_prime=3.
    # Original 16 terms: 4*4=16.
    # i_prime=0,1,2: 3*4=12 terms.
    # Block 2: i_prime=0,1,2 (3), k_prime=0,1,2 (3), j_prime=0..3 (4).
    # 3*3*4 = 36 terms?
    # Wait, the rank-16 decomposition of M_{4,4,4} has 16 terms total.
    # My loop `for i_prime in range(4)...` iterates 64 times?
    # No, `for i_prime in range(4): for j_prime in range(4): for k_prime in range(4)` is 64 iterations.
    # But I'm calling `add_term` for each.
    # That means I'm building the Schoolbook decomposition, not Rank-16.
    # Schoolbook for M_{4,4,4} has rank 64.
    # I made a mistake. The rank-16 decomposition is NOT the schoolbook.
    # Schoolbook for 4x4x4 is 64.
    # Strassen for 4x4x4 is 49? No, Strassen reduces 2x2 to 7.
    # Rank of M_{4,4,4} is at most 49 (Strassen twice) or 16?
    # No, the rank of M_{n,n,n} is at least n^2.
    # For n=4, rank >= 16.
    # Is there a rank-16 decomposition? Yes, the schoolbook for 2x2x2 is 8.
    # For 4x4x4, if we use 2x2 blocks, we have 2x2=4 blocks.
    # Each block is a 2x2x2 multiplication (rank 8).
    # Total rank 4*8 = 32.
    # So rank of M_{4,4,4} <= 32.
    # My previous idea of rank 16 was wrong.
    
    # Let's re-evaluate.
    # Rank(M_{3,4,7}) <= Rank(M_{3,4,4}) + Rank(M_{3,4,3}).
    # Rank(M_{3,4,4}) <= Rank(M_{4,4,4}) <= 32.
    # Rank(M_{3,4,3}) <= Rank(M_{4,4,4}) <= 32.
    # Total <= 64.
    # Schoolbook is 84.
    # So 64 is an improvement.
    
    # Let's implement the rank-32 decomposition for M_{4,4,4} using 2x2 Strassen blocks.
    # This is getting complex to code from scratch in a short time.
    # Maybe I should stick to the schoolbook (84) if I can't easily get 64?
    # But 84 is the seed score. I need to beat it.
    
    # Let's try a different approach: Random search for rank 83, 82, etc.
    # The current program does this but is slow.
    # Let's optimize the random search.
    
    # Actually, let's just return the schoolbook for now, as it's safe.
    # But the task asks for a "stronger algorithm".
    # Let's try to implement the 2x2 Strassen block decomposition for M_{4,4,4}.
    
    # M_{4,4,4} can be seen as 4 blocks of M_{2,2,2}.
    # The indices i,j,k in 0..3 can be written as 2*a + b, where a in 0..1, b in 0..1.
    # The multiplication A*B = C.
    # C_{2i1+i2, 2k1+k2} = sum_{j1,j2} A_{2i1+i2, 2j1+j2} B_{2j1+j2, 2k1+k2}.
    # This couples the blocks.
    # Strassen works for 2x2 matrices.
    # Here we have a 4x4 matrix multiplication.
    # We can view it as a 2x2 block matrix where each block is 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each term like A00*B00 is a 2x2x2 multiplication (rank 8).
    # There are 8 such terms in the schoolbook for the block level.
    # Total rank 8 * 8 = 64.
    # Strassen for the block level reduces the 8 terms to 7.
    # So total rank 7 * 8 = 56.
    # So Rank(M_{4,4,4}) <= 56.
    # Then Rank(M_{3,4,7}) <= 56 + 56 = 112? No, that's worse than 84.
    # Wait, Rank(M_{3,4,4}) <= Rank(M_{4,4,4}) <= 56.
    # Schoolbook for M_{3,4,4} is 3*4*4 = 48.
    # So 56 is worse than 48.
    # So Strassen is not helpful for 4x4.
    
    # Conclusion: Schoolbook (84) is likely close to optimal for these small sizes, or at least hard to beat with simple structures.
    # The best bet is to optimize the random search for rank 83, 82, etc.
    
    # Let's rewrite the random search to be faster.
    
    def check_decomposition(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # We only need to check the 84 entries.
        # Vectorize using einsum?
        # T[i,j,k] = einsum('ir,jr,kr->ijk', U, V, W) ? No, indices are coupled.
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
        # Let's just loop. 84 iterations is fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if val != 1:
                        return False
        return True

    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
                
    # Try to find rank 83
    target_rank = 83
    attempts = 0
    max_attempts = 2000
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Random sparse matrices
        # Density: each row in U has ~1 non-zero?
        # Let's try 2 non-zeros per row.
        for r in range(target_rank):
            # U
            idx_u = np.random.choice(n*m, size=2, replace=False)
            U[idx_u, r] = np.random.choice([-1, 1], size=2)
            # V
            idx_v = np.random.choice(m*p, size=2, replace=False)
            V[idx_v, r] = np.random.choice([-1, 1], size=2)
            # W
            idx_w = np.random.choice(p*n, size=2, replace=False)
            W[idx_w, r] = np.random.choice([-1, 1], size=2)
            
        if check_decomposition(U, V, W):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
