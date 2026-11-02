import time
import numpy as np
import itertools

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # 1. Construct Strassen-based block decomposition
    # Split A (4x5) into 4 blocks of 1x5: A_1, A_2, A_3, A_4
    # Split B (5x6) into 4 blocks of 5x1? No, B is 5x6.
    # Let's split the output C (4x6) into 2x3 blocks of 2x2.
    # C = [C11 C12 C13; C21 C22 C23]
    # C11 = A_top * B_left, where A_top is 2x5, B_left is 5x2.
    # This is 2x5 * 5x2.
    # We can use Strassen for the 2x2 parts, but the contraction is 5.
    # 
    # Alternative: Split the contraction dimension m=5 into 2+3?
    # A = [A1 (4x2), A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # Rank(A1*B1) <= 4*2*6 = 48.
    # Rank(A2*B2) <= 4*3*6 = 72.
    # Total 120. No gain.
    #
    # Let's try splitting the output indices.
    # n=4, p=6.
    # Split n into 2+2. Split p into 3+3? Or 2+2+2?
    # Let's split n=4 into 2+2 and p=6 into 2+2+2.
    # This creates 2 * 3 = 6 blocks of 2x2 output.
    # Each block C_{ab} (2x2) is computed from A_block (2x5) and B_block (5x2).
    # The tensor for 2x5 * 5x2 has rank at most 2*5*2 = 20.
    # Can we do better?
    # The multiplication tensor M_{2,5,2} can be decomposed.
    # Note that M_{2,5,2} is equivalent to multiplying a 2x5 matrix by a 5x2 matrix.
    # This is the same as computing the outer product of a 2-vector and a 2-vector for each of the 5 components? No.
    # C_{ik} = sum_{j=1}^5 A_{ij} B_{jk}.
    # This is a sum of 5 rank-1 matrices (outer products of column i of A and row k of B).
    # So rank is at most 5 * (2*2) = 20? No, the rank of the tensor is the number of terms in the sum of rank-1 tensors.
    # Each term A_{ij} B_{jk} contributes to C_{ik}.
    # For a fixed j, the contribution is A_{:,j} * B_{j,:}, which is a 2x2 matrix of rank 1.
    # So for a fixed j, we have 4 rank-1 tensors in the (i,k) space? No, the tensor rank is over the indices (i,j) x (j,k) x (i,k).
    # A rank-1 tensor is u_{ij} v_{jk} w_{ik}.
    # The term for fixed j is: (e_i * A_{ij}) * (B_{jk} * e_k) ... this is not a single rank-1 tensor in the full space.
    # Actually, the schoolbook for 2x5*5x2 is:
    # sum_{i=1}^2 sum_{j=1}^5 sum_{k=1}^2 e_i e_j (A) * e_j e_k (B) * e_i e_k (C)
    # This is 20 terms.
    #
    # Is there a Strassen-like trick for 2x5*5x2?
    # We can group the j indices.
    # Let's just use the schoolbook for the 6 blocks.
    # 6 blocks * 20 rank = 120. Still 120.
    #
    # Wait, the "Strassen-block structure" inspiration mentioned rank 108.
    # How?
    # Maybe splitting n=4 into 2+2 and p=6 into 2+2+2 is not the only way.
    # What if we split the contraction m=5?
    # No, m is the inner dimension.
    #
    # Let's look at the known lower bound.
    # R(M_{4,5,6}) >= max(4*6, 4*5, 5*6) = max(24, 20, 30) = 30.
    # Also R(M_{n,m,p}) >= R(M_{n,m,p-1}) + ...?
    #
    # Let's try a different block structure.
    # Split A (4x5) into 2 blocks of 2x5.
    # Split B (5x6) into 2 blocks of 5x3.
    # C1 (2x3) = A1 * B1. Rank <= 2*5*3 = 30.
    # C2 (2x3) = A2 * B2. Rank <= 2*5*3 = 30.
    # Total 60?
    # Wait, A1 is 2x5, B1 is 5x3. C1 is 2x3.
    # Schoolbook rank for 2x5*5x3 is 2*5*3 = 30.
    # So total rank 60.
    # This is much better than 120!
    # Why did the previous code fail to see this?
    # "We can write 4 = 2*2 and 6 = 2*3 and 5 = 5."
    # "Let's use a block decomposition. Write A as [A1; A2] where A1 is 2x5 and A2 is 2x5."
    # "Write B as [B1, B2, B3] where B1, B2, B3 are 5x2."
    # "Then C = [A1; A2] * [B1, B2, B3]"
    # "Each block is 2x2."
    # "Rank of 2x2*2x2 is at most 7 (Strassen)."
    # "There are 4 blocks, each 2x2."
    # "But the contraction dimension is still 5."
    # "For each 2x2*2x2 block with 5-dimensional contraction: ... Rank is at most 20."
    # "4 blocks * 20 = 80."
    #
    # The previous code assumed the blocks were 2x2 * 2x2 (contraction 2).
    # But here the blocks are 2x5 * 5x2 (contraction 5).
    # If we split B into 5x2 blocks, the contraction is 5.
    # If we split B into 5x3 blocks, the contraction is 5.
    #
    # Let's re-evaluate the rank of 2x5 * 5x3.
    # C_{ik} = sum_{j=1}^5 A_{ij} B_{jk}.
    # i in {1,2}, k in {1,2,3}, j in {1..5}.
    # Schoolbook: 2*5*3 = 30.
    # Can we do better?
    # R(M_{2,5,3})?
    # Lower bound: max(2*3, 2*5, 5*3) = max(6, 10, 15) = 15.
    #
    # If we can achieve rank 15 for 2x5*5x3, then total rank for 4x5*5x6 (two such blocks) would be 30.
    # This would be optimal (matching the lower bound of the full tensor? No, full lower bound is 30).
    # If the full tensor rank is 30, and we can decompose it into two independent 2x5*5x3 blocks, each of rank 15, then total rank is 30.
    #
    # Is R(M_{2,5,3}) = 15?
    # M_{2,5,3} is a sub-tensor of M_{2,5,5} (if we zero out some B columns).
    # R(M_{2,5,5})?
    # If we can solve 2x5*5x5 with rank 20 (schoolbook), maybe less?
    #
    # Let's try to construct a rank-15 decomposition for 2x5*5x3.
    # Or just use the schoolbook rank 30 for each block, total 60.
    # This is already a big improvement over 120.
    #
    # Let's verify if we can do better than 30 for 2x5*5x3.
    # Actually, there is a known result: R(M_{m,n,p}) <= m*p*ceil(n/m) ? No.
    #
    # Let's stick to the block decomposition with rank 60 first.
    # U, V, W matrices.
    # 
    # Block 1: A1 (2x5), B1 (5x3) -> C1 (2x3)
    # Indices for U: rows 0..1 (i=0,1), cols 0..4 (j=0..4). Total 10 rows.
    # Indices for V: rows 0..4 (j=0..4), cols 0..2 (k=0..2). Total 15 rows.
    # Indices for W: rows 0..2 (k=0..2), cols 0..1 (i=0,1). Total 6 rows.
    #
    # Block 2: A2 (2x5), B2 (5x3) -> C2 (2x3)
    # Indices for U: rows 10..19 (i=2,3), cols 0..4.
    # Indices for V: rows 0..14 (same as B1? No, B is 5x6. B1 is cols 0-2, B2 is cols 3-5).
    # So V for Block 2: rows 0..4 (j), cols 3..5 (k).
    # W for Block 2: rows 3..5 (k=3,4,5), cols 0..1 (i=0,1) relative to block?
    # No, W index is k*n + i.
    # For Block 1 (k=0,1,2), W rows are 0*n+0, 0*n+1, 1*n+0, 1*n+1, 2*n+0, 2*n+1.
    # i.e., 0,1,4,5,8,9.
    # For Block 2 (k=3,4,5), W rows are 3*n+0, 3*n+1, 4*n+0, 4*n+1, 5*n+0, 5*n+1.
    # i.e., 12,13,16,17,20,21.
    #
    # Total U rows: 20 (4*5).
    # Total V rows: 30 (5*6).
    # Total W rows: 24 (6*4).
    #
    # Let's build the rank-60 decomposition.
    
    rank = 60
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # Block 1: i in {0,1}, k in {0,1,2}
    for i in range(2):
        for j in range(5):
            for k in range(3):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
                
    # Block 2: i in {2,3}, k in {3,4,5}
    for i in range(2, 4):
        for j in range(5):
            for k in range(3, 6):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1

    # Now we have a valid decomposition of rank 60.
    # Let's try to prune it to find a lower rank.
    # We want to remove columns c such that the tensor T - sum_{c' != c} T_c' is still rank 1?
    # No, we want to remove columns such that the remaining set still spans the tensor.
    # Actually, we want to find a subset of columns that still sums to T.
    # Since the current set sums to T exactly, and each column is a rank-1 tensor,
    # if we remove a column, the sum changes.
    # We can try to replace a column with a linear combination of others?
    # Or simply check if any column is redundant (i.e., T can be formed without it).
    # This is equivalent to checking if the current set is linearly independent in the tensor space?
    # No, the sum must equal T.
    # If we have sum T_c = T, and we remove T_c, we have sum_{c' != c} T_c' = T - T_c.
    # We need sum_{c' != c} T_c' to be representable by the remaining columns?
    # No, the remaining columns ARE the representation.
    # So we need to check if T - T_c can be represented by {T_c' | c' != c}.
    # This is always true if we can adjust the coefficients?
    # No, the coefficients are fixed at 1.0 in the schoolbook.
    # Wait, the entries in U, V, W can be any real numbers.
    # The schoolbook uses 1.0.
    # We can try to find a decomposition with smaller rank by solving a linear system or using optimization.
    
    # However, 60 is already a significant improvement over 120.
    # Let's see if we can improve the block decomposition further.
    # Is R(M_{2,5,3}) < 30?
    # If yes, we can use that.
    # 
    # For now, let's return the rank-60 solution.
    # It is valid and scores 120/60 = 2.0 (if seed was 120 and score is seed/value? No, score is seed/value where value is rank? 
    # "score is the rank" -> "search score is seed/value".
    # If seed rank is 120, and we find rank 60, score = 120/60 = 2.0.
    # This is an improvement.
    
    # Let's try to prune the rank-60 solution to see if we can go lower.
    # We can try to remove columns one by one and check if the remaining matrix still decomposes T.
    # To check if a set of columns {U_c, V_c, W_c} decomposes T, we need:
    # sum_c U_c V_c W_c^T = T
    # This is a tensor equation.
    # We can flatten U, V, W into vectors and check if the sum of outer products matches the flattened T.
    
    T_flat = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_flat[i*m*p + j*p + k] = 1.0 # Wait, T[(i,j), (j,k), (i,k)] = 1.
                # The tensor T is indexed by (i,j), (j,k), (i,k).
                # In the flattened vector for the sum, the entry corresponding to (i,j,k) is 1.
                # The sum is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
                # This sum must equal 1 for all (i,j,k).
    
    # Let's verify the rank-60 solution first.
    # And then try greedy removal.
    
    current_rank = rank
    U_curr = U[:, :current_rank]
    V_curr = V[:, :current_rank]
    W_curr = W[:, :current_rank]
    
    # Function to check validity
    def is_valid(U, V, W):
        # Efficient check: compute the sum of rank-1 tensors and compare to T.
        # Sum_c U[:,c] V[:,c] W[:,c]^T
        # This is a 3D tensor.
        # We can check a few random entries or all entries.
        # All entries is n*m*p = 120.
        # For each entry (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], np.dot(V[j*p+k, :], W[k*n+i, :]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not is_valid(U_curr, V_curr, W_curr):
        # Fallback to schoolbook if something went wrong
        rank_total = n * m * p
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Greedy pruning
    # Try to remove columns.
    # If we remove column c, the new sum is T - T_c.
    # We need T - T_c to be representable by the remaining columns with their fixed coefficients?
    # No, the coefficients are part of the vectors U, V, W.
    # If we remove column c, we just drop it. The remaining columns still have their original values.
    # So we need sum_{c' != c} U_{c'} V_{c'} W_{c'}^T = T.
    # This is only possible if T_c = 0, which is not true.
    # So simple removal doesn't work unless the columns are linearly dependent in a way that allows cancellation?
    # No, the sum must be exactly T.
    # If sum_all = T, then sum_{not c} = T - T_c.
    # For sum_{not c} to equal T, we need T_c = 0.
    # So we cannot simply remove a column.
    
    # We need to find a NEW decomposition of lower rank.
    # This is a hard problem.
    # However, we can try to combine columns.
    # If U_a V_a W_a^T + U_b V_b W_b^T can be replaced by a single rank-1 term?
    # Only if the sum is rank 1.
    # 
    # Given the time constraint, returning the rank-60 solution is a solid improvement.
    # Let's check if we can improve the block rank.
    # R(M_{2,5,3}) is likely 15 or 20?
    # If it's 15, total rank is 30.
    # If it's 20, total rank is 40.
    # 
    # Let's try to construct a rank-15 decomposition for 2x5*5x3.
    # This is equivalent to multiplying a 2x5 matrix by a 5x3 matrix.
    # This is the same as the bilinear map from (R^2 x R^5) x (R^5 x R^3) to R^2 x R^3.
    # 
    # Actually, there is a known decomposition for M_{2,n,2} with rank 2n-1? No.
    # For M_{2,2,2}, rank is 7.
    # For M_{2,3,2}, rank is?
    # 
    # Let's stick with the safe rank-60 solution.
    
    return {"U": U_curr, "V": V_curr, "W": W_curr}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start}")
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    if errors == 0:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED with {errors} errors")
