import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    n, m, p = 3, 3, 8
    
    def time_left():
        return deadline - time.time()
    
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for all i,j,k
        # Use einsum for efficiency
        # First compute T[j,k,i] = sum_r V[j,k,r] * W[k,i,r]
        # Then result[i,j,k] = sum_r U[i,j,r] * T[j,k,i]
        
        # Actually, let's do it step by step
        # For each r, we have outer products
        # Let's compute the full tensor
        
        # Method: for each r, compute contribution
        # U_r: (n,m), V_r: (m,p), W_r: (p,n)
        # term_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        total = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # term[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # We can compute this as:
            # temp[j,k] = sum over i? No.
            # Let's use broadcasting
            # u[i,j] * v[j,k] gives (n,m,p) if we broadcast properly
            # u: (n, m, 1), v: (m, p) -> need to align
            # Actually: u[i,j] * v[j,k] * w[k,i]
            # = (u[:, :, None] * v[None, :, :] * w[None, None, :]) but w is (p,n)
            # w[k,i] means w indexed by (k,i) where k is col of B, i is row of C
            # So w_reshaped[k, i] = W_r[k*n + i]
            # w is already (p, n) = (k, i)
            
            # u[i,j] * v[j,k] * w[k,i]
            # u: (n, m, 1)
            # v: (1, m, p) 
            # w: (p, n, 1) -> need to transpose to (n, 1, p)? No.
            # w[k,i] -> we want (i, j, k) indexing
            # w[k,i] = w_reshaped[k, i]
            # So for fixed i,j,k: u[i,j] * v[j,k] * w[k,i]
            
            # Let's compute:
            # temp_1[i,j,k] = u[i,j] * v[j,k]  -> shape (n, m, p)
            # temp_1 = u[:, :, None] * v[None, :, :]  # (n,m,1) * (1,m,p) -> (n,m,p)
            # But v[j,k] means v indexed by (j,k), so v is (m,p)
            # u[:, :, None] is (n, m, 1)
            # v[None, :, :] is (1, m, p)
            # Product: (n, m, p) where [i,j,k] = u[i,j] * v[j,k] ✓
            
            # Then multiply by w[k,i]:
            # w is (p, n), we need w[k,i] for each (i,j,k)
            # w[k,i] = w_reshaped[k, i]
            # So we need to index w with k (axis 2 of result) and i (axis 0)
            # w_transposed[i, k] = w[k, i] -> w.T is (n, p), so w.T[i, k] = w[k, i]
            # So: result[i,j,k] = temp_1[i,j,k] * w.T[i,k]
            # w.T: (n, p), so w.T[None, :, :] is (1, n, p)
            # temp_1 * w.T[None, :, :] -> (n, m, p) where [i,j,k] = temp_1[i,j,k] * w.T[i,k] = u[i,j]*v[j,k]*w[k,i] ✓
            
            term = u[:, :, None] * v[None, :, :] * w.T[None, :, :]
            total += term
        
        return np.allclose(total, 1.0, atol=1e-9)
    
    # Known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the construction by Bini, Capovani, Lotti, Romani (1979)
    # or similar algebraic constructions
    
    def get_3x3_decomp():
        """
        Returns U, V, W for 3x3 matrix multiplication with rank 23.
        U: (9, 23), V: (9, 23), W: (9, 23)
        where U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        # Use a known rank-23 construction
        # Based on the "Bini-Capovani-Lotti-Romani" type construction
        
        # One approach: use the fact that 3x3 can be done with 23 multiplications
        # Let me use a concrete construction
        
        # Actually, let me use a simpler approach: 
        # Start with Strassen-like for 2x2 and extend, or use known tables
        
        # For reliability, let me use a construction that I can verify
        # The minimal rank for 3x3 is 19 (proven lower bound is 19, upper bound is 23)
        # But 23 is a well-known achievable rank
        
        # Let me use a different approach: exploit the structure
        # 3x3 = 2x2 + 1x1 + cross terms
        
        # Actually, for the purpose of this problem, let me use a 
        # carefully constructed decomposition.
        
        # Known: rank of 3x3 matrix multiplication is between 19 and 23
        # Let me use a rank-23 construction
        
        # I'll use the following approach:
        # Decompose into blocks and use Strassen for the 2x2 parts
        
        # Let A = [[A11, A12, A13],
        #          [A21, A22, A23],
        #          [A31, A32, A33]]
        # B = [[B11, B12, B13],
        #      [B21, B22, B23],
        #      [B31, B32, B33]]
        
        # This is complex. Let me use a different strategy.
        
        # For the 3x8 problem, I can use:
        # 3x8 = 3x3 + 3x3 + 3x2
        # For each 3x3, use rank 23
        # For 3x2, use rank 16 (or less)
        # Total: 23 + 23 + 16 = 62
        
        # But we can do better by sharing structure.
        
        # Let me try a different approach: use the fact that we can 
        # batch the columns.
        
        # Actually, the simplest reliable approach:
        # Use schoolbook for 3x3 (rank 9) x2 + schoolbook for 3x2 (rank 6) = 24
        # That's rank 24, which is better than 72.
        
        # But we can do even better. Let me use Strassen for the 2x2 subblocks.
        
        # For now, let me use a hybrid:
        # For 3x3: use a rank-15 or rank-16 construction if possible
        # For 3x2: use rank 8 or 9
        
        # Let me construct a rank-16 decomposition for 3x3
        # This is non-trivial. Let me use a known construction.
        
        # Alternative: use the "laser method" or "border rank" constructions
        # But those may not give exact decompositions.
        
        # For reliability, let me use:
        # 3x3: rank 23 (known achievable)
        # 3x2: rank 12 (can be done with Strassen-like)
        # Total: 23 + 23 + 12 = 58
        
        # Or better: 3x3: rank 20, 3x2: rank 10
        # Total: 20 + 20 + 10 = 50
        
        # Let me just use a safe construction:
        # 3x3 via Strassen decomposition of 2x2 blocks
        # This gives rank 15 for 3x3 (not minimal but good)
        # 3x2 via similar: rank 10
        # Total: 15 + 15 + 10 = 40
        
        # Actually, let me use the simplest reliable approach:
        # For 3x3: use 9 scalar multiplications (schoolbook)
        # For 3x2: use 6 scalar multiplications (schoolbook)  
        # Total: 9 + 9 + 6 = 24
        
        # This is rank 24, which is a 3x improvement over 72.
        
        # But let me try to do better with Strassen.
        
        # Strassen for 2x2: rank 7
        # For 3x3, we can partition into 2x2 + 1x1 + cross terms
        # This is complex. Let me use a different strategy.
        
        # I'll use the following:
        # For the 3x8 problem, treat it as multiplying 3x3 by 3x8
        # We can use block Strassen or similar.
        
        # Let me just implement a clean schoolbook with rank 24
        # and verify it works.
        
        # Actually, let me try to find a better construction.
        # 
        # Key insight: For 3x3 matrix multiplication, we can use:
        # - 4 blocks of 2x2 (each rank 7 via Strassen) = 28
        # - Plus correction terms
        # This is not straightforward.
        
        # Let me use a practical approach:
        # 1. For 3x3, use rank 23 (I'll construct it)
        # 2. For 3x2, use rank 12
        # 3. Total: 23 + 23 + 12 = 58
        
        # But constructing rank-23 for 3x3 is non-trivial.
        
        # Alternative: use the fact that we can share the U factor
        # across the two 3x3 blocks since they use the same A matrix.
        
        # Wait, that's the key! The U factor depends only on A (3x3),
        # not on which block of B we're multiplying.
        
        # So for 3x8 = 3x3 + 3x3 + 3x2:
        # U is the same for all blocks (depends on A)
        # V and W differ per block
        # 
        # But in the tensor decomposition, each term r has:
        # U_r[i,j] * V_r[j,k] * W_r[k,i]
        # 
        # For block 1 (k=0,1,2): V_r1[j,k], W_r1[k,i]
        # For block 2 (k=3,4,5): V_r2[j,k-3], W_r2[k-3,i]  
        # For block 3 (k=6,7): V_r3[j,k-6], W_r3[k-6,i]
        #
        # If we use the same U for all, we can't just concatenate
        # because U_r must be consistent.
        
        # Actually, we CAN use different U for different blocks.
        # The decomposition is:
        # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
        #
        # For k in {0,1,2}: sum_{r in R1} U_r1[i,j] * V_r1[j,k] * W_r1[k,i] = 1
        # For k in {3,4,5}: sum_{r in R2} U_r2[i,j] * V_r2[j,k-3] * W_r2[k-3,i] = 1
        # For k in {6,7}: sum_{r in R3} U_r3[i,j] * V_r3[j,k-6] * W_r3[k-6,i] = 1
        #
        # And for k not in the block, the terms are 0.
        #
        # So we can use:
        # U_final = [U_3x3, U_3x3, U_3x2] (concatenated along rank)
        # V_final: for each r, V_r[j,k] is non-zero only for k in the corresponding block
        # W_final: similar
        #
        # This gives total rank = rank(3x3) + rank(3x3) + rank(3x2)
        
        # For 3x3, let me use rank 23
        # For 3x2, let me use rank 12
        # Total: 23 + 23 + 12 = 58
        
        # But I need to actually construct these.
        
        # Let me use a simpler approach for now:
        # 3x3: rank 15 (using a Strassen-like construction)
        # 3x2: rank 9
        # Total: 15 + 15 + 9 = 39
        
        # I'll construct these explicitly.
        
        # For 3x2 matrix multiplication (3x3 times 3x2):
        # C = A * B where A is 3x3, B is 3x2
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This is just 6 independent dot products of length 3
        # Schoolbook: rank 6
        # Can we do better? 
        # Each dot product of length 3 has rank 3 (schoolbook)
        # But we can share structure.
        # Actually, for a single dot product of length 3, rank is 3
        # For 6 independent dot products, rank is 6 (no sharing possible)
        # Unless we use a different decomposition.
        
        # Wait, the tensor for 3x3 times 3x2 is:
        # T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2}, k in {0,1}
        # This is a 3x3x2 tensor.
        # The rank of this tensor...
        
        # Actually, I think for 3x3 times 3x2, the minimal rank is 6
        # (just 6 scalar multiplications, one for each output element)
        
        # And for 3x3 times 3x3, the minimal rank is 19 (lower bound) 
        # but 23 is achievable.
        
        # Let me use:
        # 3x3: rank 23
        # 3x2: rank 6
        # Total: 23 + 23 + 6 = 52
        
        # I need to construct the rank-23 decomposition for 3x3.
        
        # Let me use a known construction. I'll use the following:
        # Based on the "Bini et al." construction or similar.
        
        # Actually, for simplicity and reliability, let me use:
        # 3x3: rank 27 (schoolbook) - but that's not better
        # 
        # Let me try a different approach: use the fact that 
        # we can do 3x3 with rank 23 using a specific construction.
        
        # I'll look up a concrete rank-23 construction for 3x3.
        
        # One known construction uses the following idea:
        # Write A and B in terms of 2x2 blocks and use Strassen.
        
        # Let me just use a practical construction:
        # For 3x3, use 4 Strassen 2x2 multiplications (rank 7 each) = 28
        # Plus 5 correction terms = 33. That's worse than schoolbook.
        
        # Let me use a different strategy:
        # Use the "decomposition by borders" or "laser method"
        # But those give border rank, not exact rank.
        
        # For exact rank, I'll use:
        # 3x3: rank 23 (I'll construct it carefully)
        # 3x2: rank 6
        # Total: 52
        
        # Let me construct the rank-23 decomposition.
        
        # Based on the construction in "Algebraic complexity of matrix multiplication"
        # by Bini, Capovani, Lotti, Romani (1979)
        
        # The construction is complex. Let me use a simpler approach:
        # Use rank 27 for 3x3 (schoolbook) but optimize the 3x2 part.
        
        # Actually, let me reconsider. The schoolbook for 3x8 is rank 72.
        # If I use schoolbook for 3x3 (rank 9) twice and schoolbook for 3x2 (rank 6):
        # Total: 9 + 9 + 6 = 24
        
        # That's rank 24, which is a 3x improvement over 72.
        
        # Let me implement this cleanly and verify it works.
        
        U_3x3 = np.zeros((9, 9))
        V_3x3 = np.zeros((9, 9))
        W_3x3 = np.zeros((9, 9))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_3x3[i*3+j, col] = 1
                    V_3x3[j*3+k, col] = 1
                    W_3x3[k*3+i, col] = 1
                    col += 1
        
        U_3x2 = np.zeros((9, 6))
        V_3x2 = np.zeros((6, 6))
        W_3x2 = np.zeros((6, 6))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_3x2[i*3+j, col] = 1
                    V_3x2[j*2+k, col] = 1
                    W_3x2[k*3+i, col] = 1
                    col += 1
        
        return U_3x3, V_3x3, W_3x3, U_3x2, V_3x2, W_3x2
    
    # Get the decompositions
    U_3x3, V_3x3, W_3x3, U_3x2, V_3x2, W_3x2 = get_3x3_decomp()
    
    rank_3x3 = 9
    rank_3x2 = 6
    total_rank = rank_3x3 + rank_3x3 + rank_3x2  # 9 + 9 + 6 = 24
    
    U_final = np.zeros((9, total_rank))
    V_final = np.zeros((24, total_rank))
    W_final = np.zeros((24, total_rank))
    
    # Block 1: k=0,1,2 (first 3x3)
    U_final[:, 0:rank_3x3] = U_3x3
    for j in range(3):
        for k in range(3):
            V_final[j*8+k, 0:rank_3x3] = V_3x3[j*3+k]
    for k in range(3):
        for i in range(3):
            W_final[k*3+i, 0:rank_3x3] = W_3x3[k*3+i]
    
    # Block 2: k=3,4,5 (second 3x3)
    offset = rank_3x3
    U_final[:, offset:offset+rank_3x3] = U_3x3
    for j in range(3):
        for k in range(3):
            V_final[j*8+(k+3), offset:offset+rank_3x3] = V_3x3[j*3+k]
    for k in range(3):
        for i in range(3):
            W_final[(k+3)*3+i, offset:offset+rank_3x3] = W_3x3[k*3+i]
    
    # Block 3: k=6,7 (3x2)
    offset2 = 2 * rank_3x3
    U_final[:, offset2:offset2+rank_3x2] = U_3x2
    for j in range(3):
        for k in range(2):
            V_final[j*8+(k+6), offset2:offset2+rank_3x2] = V_3x2[j*2+k]
    for k in range(2):
        for i in range(3):
            W_final[(k+6)*3+i, offset2:offset2+rank_3x2] = W_3x2[k*3+i]
    
    # Verify
    if not verify_decomp(U_final, V_final, W_final, n, m, p):
        # Fallback to pure schoolbook
        rank = 72
        U_final = np.zeros((9, rank))
        V_final = np.zeros((24, rank))
        W_final = np.zeros((24, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U_final[i*3+j, col] = 1
                    V_final[j*8+k, col] = 1
                    W_final[k*3+i, col] = 1
                    col += 1
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
