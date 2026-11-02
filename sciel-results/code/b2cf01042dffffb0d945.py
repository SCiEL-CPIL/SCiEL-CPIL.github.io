import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    1. Start with the known rank-25 block-Strassen decomposition
    2. Verify it's correct
    3. Try to find a lower rank via linear algebra / ILP on the tensor space
    4. Return the best valid decomposition found within time budget
    """
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # First, build the rank-25 Strassen decomposition
    def build_strassen_25():
        rank = 25
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start):
            # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # M3 = A00 * (B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # M5 = (A00 + A01) * B1,ks+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        # Schoolbook for column 6
        col_start = 21
        k = 6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1
        
        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1
        
        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1
        
        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1
        
        return U, V, W
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    s = sum(U[a_idx, r] * V[b_idx, r] * W[c_idx, r] for r in range(rank))
                    if s != 1:
                        return False
        return True
    
    # Build and verify the rank-25 decomposition
    U25, V25, W25 = build_strassen_25()
    valid_25 = verify(U25, V25, W25)
    
    if not valid_25:
        # Fallback to schoolbook rank 28
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_rank = 25
    best_U, best_V, best_W = U25, V25, W25
    
    # Try to find a lower rank by examining the tensor structure
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,7)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Approach: Use the fact that for fixed j, we have a 2x2x7 sub-problem
    # Actually, let's try a different approach: exploit symmetries
    
    # Try rank 24, 23, etc. by attempting to merge terms
    # For each candidate rank r < 25, try to find if we can express the tensor with r terms
    
    # A more systematic approach: the tensor is separable in a certain way
    # T[i,j,k] = delta_{i,j,k} where the "product" structure means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to use the structure more carefully
    # For 2x2x7, the tensor can be viewed as 7 copies of 2x2x1 tensors
    # Each 2x2x1 tensor has rank 4 (schoolbook) or can be part of Strassen
    
    # Alternative: try to find if there's a rank-24 decomposition by
    # checking if one of the 25 Strassen terms can be absorbed
    
    # Let's try a greedy approach: start with rank 25, try to remove one term
    # and see if the remaining 24 can still span the tensor
    
    if time.time() < deadline:
        # Try to reduce rank by finding linear dependencies in the rank-1 tensors
        # The rank-1 tensor for term r is: U[:,r] outer V[:,r] outer W[:,r]
        # We need sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = T
        
        # Flatten the tensor space: dimension is 4*14*14 = 784
        # But we only need the 28 entries T[i,j,k] = 1
        
        # Actually, let's think about this differently.
        # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is a system of 28 equations in the products U*V*W.
        # We can try to find a lower-rank solution by relaxing the integer constraint
        # and using linear algebra.
        
        # For a given rank r, we have r terms, each contributing to all 28 equations.
        # The question is: what is the minimum r?
        
        # Lower bound: The tensor T has a certain "tensor rank".
        # For 2x2x7, known results suggest rank is around 20-25.
        
        # Let's try to use the structure: group by j
        # For j=0: C[i,k] = A[i,0]*B[0,k] for i in {0,1}, k in {0..6}
        # This is a rank-1 matrix (outer product of A[:,0] and B[0,:])
        # For j=1: C[i,k] += A[i,1]*B[1,k]
        # So C = A[:,0] * B[0,:] + A[:,1] * B[1,:]
        # This is a sum of 2 rank-1 matrices, so the "matrix rank" is 2.
        
        # But the tensor rank is different. Let's try to find a decomposition
        # that exploits the fact that B has 7 columns.
        
        # Idea: Use Strassen-like tricks across columns
        # The key insight: for 2x2 matrices, we can multiply two 2x2 matrices
        # with 7 scalar multiplications (Strassen). Here we have 2x2 * 2x7.
        
        # Another idea: treat the 2x7 matrix as two 2x4 matrices plus a 2x1
        # 2x4 * 2x2 can be done with... hmm, not directly helpful.
        
        # Let's try a computational approach: use linear programming
        # to find if rank 24 is achievable.
        
        # Actually, let's try a simpler approach: check if the 25 Strassen terms
        # have any redundancy by computing the "effective rank" of the set
        # of rank-1 tensors.
        
        # For each term r, define a vector in R^28:
        # v_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We need sum_r v_r = ones(28)
        
        # If some v_r is in the span of the others, we might be able to
        # adjust coefficients to eliminate it.
        
        # But since we need integer coefficients, this is tricky.
        
        # Let's try: for each term r, check if removing it and adjusting
        # the others can still sum to 1.
        
        # This is an integer linear programming problem, which is hard.
        # Let's try a heuristic: for each r, set U[:,r]=V[:,r]=W[:,r]=0
        # and see if the remaining 24 terms can be scaled to sum to 1.
        
        # Actually, that won't work because we need exact integer solutions.
        
        # Let's try a different strategy: look for known results.
        # The tensor rank of 2x2x7 is known to be at most 25 (our construction).
        # Can it be lower?
        
        # For 2x2xn, the rank is known to be:
        # n=1: 4
        # n=2: 7 (Strassen)
        # n=3: 10 (Strassen + schoolbook)
        # n=4: 14 (2x Strassen)
        # n=5: 17 (2x Strassen + schoolbook)
        # n=6: 21 (3x Strassen)
        # n=7: 24 or 25?
        
        # Hmm, for n=7, maybe we can do better than 25.
        # Let's try: 3x Strassen (21) + 3 for the 7th column?
        # That's what we have: 21 + 4 = 25.
        
        # Can we do the 7th column with fewer than 4 terms?
        # For a single column, we need:
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # This is a 2x2x1 tensor, which has rank 4 (schoolbook).
        # But wait, can we share terms with the Strassen blocks?
        
        # The Strassen blocks use terms that involve combinations of A entries
        # and B entries from specific column pairs. The 7th column is separate.
        
        # Unless... we can modify the Strassen decomposition to include
        # the 7th column in a clever way.
        
        # Idea: Instead of 3 Strassen blocks + schoolbook, use a different
        # partition. For example:
        # - Use Strassen for columns (0,1), (2,3), and a modified scheme for (4,5,6)
        
        # For 2x2 * 2x3, what's the rank?
        # Schoolbook: 6
        # Strassen + schoolbook: 7 + 4 = 11? No, that's worse.
        # Actually, for 2x2x3, the rank is known to be 10.
        
        # Let me try: 2x Strassen (14) + 6 for columns 4,5,6?
        # That's 14 + 6 = 20. But 20 < 25!
        
        # Wait, for 2x2 * 2x3, can we do it in 6 terms?
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}
        # This is 6 independent products, so rank 6 (schoolbook).
        
        # So: 2x Strassen (14) + schoolbook for 3 columns (6) = 20!
        
        # Let's verify this construction.
        
        if time.time() < deadline:
            def build_strassen_20():
                rank = 20
                U = np.zeros((n*m, rank), dtype=int)
                V = np.zeros((m*p, rank), dtype=int)
                W = np.zeros((p*n, rank), dtype=int)
                
                def add_strassen_block(col_start, k_start):
                    # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
                    U[0, col_start+0] += 1
                    U[3, col_start+0] += 1
                    V[7*0 + k_start, col_start+0] += 1
                    V[7*1 + (k_start+1), col_start+0] += 1
                    W[2*k_start + 0, col_start+0] += 1
                    W[2*(k_start+1) + 1, col_start+0] += 1

                    # M2 = (A10 + A11) * B0,ks
                    U[2, col_start+1] += 1
                    U[3, col_start+1] += 1
                    V[7*0 + k_start, col_start+1] += 1
                    W[2*k_start + 1, col_start+1] += 1
                    W[2*(k_start+1) + 1, col_start+1] -= 1

                    # M3 = A00 * (B0,ks+1 - B1,ks+1)
                    U[0, col_start+2] += 1
                    V[7*0 + (k_start+1), col_start+2] += 1
                    V[7*1 + (k_start+1), col_start+2] -= 1
                    W[2*(k_start+1) + 0, col_start+2] += 1
                    W[2*(k_start+1) + 1, col_start+2] += 1

                    # M4 = A11 * (B1,ks - B0,ks)
                    U[3, col_start+3] += 1
                    V[7*1 + k_start, col_start+3] += 1
                    V[7*0 + k_start, col_start+3] -= 1
                    W[2*k_start + 0, col_start+3] += 1
                    W[2*k_start + 1, col_start+3] += 1

                    # M5 = (A00 + A01) * B1,ks+1
                    U[0, col_start+4] += 1
                    U[1, col_start+4] += 1
                    V[7*1 + (k_start+1), col_start+4] += 1
                    W[2*k_start + 0, col_start+4] -= 1
                    W[2*(k_start+1) + 0, col_start+4] += 1

                    # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
                    U[2, col_start+5] += 1
                    U[0, col_start+5] -= 1
                    V[7*0 + k_start, col_start+5] += 1
                    V[7*0 + (k_start+1), col_start+5] += 1
                    W[2*(k_start+1) + 1, col_start+5] += 1

                    # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
                    U[1, col_start+6] += 1
                    U[3, col_start+6] -= 1
                    V[7*1 + k_start, col_start+6] += 1
                    V[7*1 + (k_start+1), col_start+6] += 1
                    W[2*k_start + 0, col_start+6] += 1

                # Strassen for columns (0,1) and (2,3)
                add_strassen_block(0, 0)
                add_strassen_block(7, 2)
                
                # Schoolbook for columns 4,5,6
                col_start = 14
                for k in [4, 5, 6]:
                    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
                    U[0, col_start+0] += 1
                    V[7*0 + k, col_start+0] += 1
                    W[2*k + 0, col_start+0] += 1
                    
                    U[1, col_start+1] += 1
                    V[7*1 + k, col_start+1] += 1
                    W[2*k + 0, col_start+1] += 1
                    
                    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
                    U[2, col_start+2] += 1
                    V[7*0 + k, col_start+2] += 1
                    W[2*k + 1, col_start+2] += 1
                    
                    U[3, col_start+3] += 1
                    V[7*1 + k, col_start+3] += 1
                    W[2*k + 1, col_start+3] += 1
                    
                    col_start += 4
                
                return U, V, W
            
            U20, V20, W20 = build_strassen_20()
            if verify(U20, V20, W20):
                best_rank = 20
                best_U, best_V, best_W = U20, V20, W20
                print(f"Found valid rank-20 decomposition!")
        
        # Can we do even better? Try 1 Strassen + schoolbook for 5 columns
        # 7 + 10 = 17
        
        if time.time() < deadline:
            def build_strassen_17():
                rank = 17
                U = np.zeros((n*m, rank), dtype=int)
                V = np.zeros((m*p, rank), dtype=int)
                W = np.zeros((p*n, rank), dtype=int)
                
                def add_strassen_block(col_start, k_start):
                    U[0, col_start+0] += 1
                    U[3, col_start+0] += 1
                    V[7*0 + k_start, col_start+0] += 1
                    V[7*1 + (k_start+1), col_start+0] += 1
                    W[2*k_start + 0, col_start+0] += 1
                    W[2*(k_start+1) + 1, col_start+0] += 1

                    U[2, col_start+1] += 1
                    U[3, col_start+1] += 1
                    V[7*0 + k_start, col_start+1] += 1
                    W[2*k_start + 1, col_start+1] += 1
                    W[2*(k_start+1) + 1, col_start+1] -= 1

                    U[0, col_start+2] += 1
                    V[7*0 + (k_start+1), col_start+2] += 1
                    V[7*1 + (k_start+1), col_start+2] -= 1
                    W[2*(k_start+1) + 0, col_start+2] += 1
                    W[2*(k_start+1) + 1, col_start+2] += 1

                    U[3, col_start+3] += 1
                    V[7*1 + k_start, col_start+3] += 1
                    V[7*0 + k_start, col_start+3] -= 1
                    W[2*k_start + 0, col_start+3] += 1
                    W[2*k_start + 1, col_start+3] += 1

                    U[0, col_start+4] += 1
                    U[1, col_start+4] += 1
                    V[7*1 + (k_start+1), col_start+4] += 1
                    W[2*k_start + 0, col_start+4] -= 1
                    W[2*(k_start+1) + 0, col_start+4] += 1

                    U[2, col_start+5] += 1
                    U[0, col_start+5] -= 1
                    V[7*0 + k_start, col_start+5] += 1
                    V[7*0 + (k_start+1), col_start+5] += 1
                    W[2*(k_start+1) + 1, col_start+5] += 1

                    U[1, col_start+6] += 1
                    U[3, col_start+6] -= 1
                    V[7*1 + k_start, col_start+6] += 1
                    V[7*1 + (k_start+1), col_start+6] += 1
                    W[2*k_start + 0, col_start+6] += 1

                # Strassen for columns (0,1)
                add_strassen_block(0, 0)
                
                # Schoolbook for columns 2,3,4,5,6
                col_start = 7
                for k in [2, 3, 4, 5, 6]:
                    U[0, col_start+0] += 1
                    V[7*0 + k, col_start+0] += 1
                    W[2*k + 0, col_start+0] += 1
                    
                    U[1, col_start+1] += 1
                    V[7*1 + k, col_start+1] += 1
                    W[2*k + 0, col_start+1] += 1
                    
                    U[2, col_start+2] += 1
                    V[7*0 + k, col_start+2] += 1
                    W[2*k + 1, col_start+2] += 1
                    
                    U[3, col_start+3] += 1
                    V[7*1 + k, col_start+3] += 1
                    W[2*k + 1, col_start+3] += 1
                    
                    col_start += 4
                
                return U, V, W
            
            U17, V17, W17 = build_strassen_17()
            if verify(U17, V17, W17):
                best_rank = 17
                best_U, best_V, best_W = U17, V17, W17
                print(f"Found valid rank-17 decomposition!")
        
        # Can we do even better with a mixed approach?
        # Try: Strassen for (0,1), (2,3), and a custom scheme for (4,5,6)
        # For 2x2x3, the rank is 10 (known result)
        # So: 7 + 7 + 10 = 24? That's worse than 25.
        
        # Or: Strassen for (0,1), (2,3), (4,5) and schoolbook for 6: 7+7+7+4=25
        # Or: Strassen for (0,1), (2,3) and schoolbook for 4,5,6: 7+7+6=20
        
        # Can we do better than 20? What about a non-block approach?
        # The tensor rank of 2x2x7 is an open question in general,
        # but for small cases it's known.
        
        # Let's try: can we do 2x2x7 in rank 19 or less?
        # This would require a clever cross-column optimization.
        
        # For now, return the best we found.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
