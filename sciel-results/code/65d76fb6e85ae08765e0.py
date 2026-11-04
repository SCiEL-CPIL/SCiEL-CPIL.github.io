import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We want to decompose the matrix multiplication tensor for 2x5 * 5x5 -> 2x5.
    # The tensor T has indices (i, j, k) for inputs and (i', k') for outputs.
    # Specifically, the decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In the rank-r decomposition:
    # C[i, k] = sum_{r=1}^R U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, the standard definition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = delta_{i,i'} delta_{k,k'}
    # Let's verify the index conventions from the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the output index is `k*n + i`.
    # The input indices are `i*m + j` (for A) and `j*p + k` (for B).
    # So the tensor equation is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 1 if i==i' and k==k' else 0.
    
    # The naive rank is n*m*p = 2*5*5 = 50.
    # We can use Strassen's algorithm for 2x2 matrices to reduce the rank.
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications.
    # The naive rank for 2x2 is 8.
    # We can partition the 5x5 matrices into blocks.
    # 5 = 2 + 2 + 1.
    # Let's try to use block decomposition.
    # A is 2x5, B is 5x5.
    # We can split the columns of A and rows of B.
    # A = [A1 A2 A3] where A1, A2 are 2x2 and A3 is 2x1.
    # B = [B1; B2; B3] where B1, B2 are 2x5 and B3 is 1x5.
    # C = A1*B1 + A2*B2 + A3*B3.
    
    # The multiplication A1*B1 is 2x2 * 2x2. Rank 7 (Strassen).
    # The multiplication A2*B2 is 2x2 * 2x5. 
    #   This can be seen as two 2x2 * 2x2 multiplications plus a 2x2 * 2x1?
    #   Or simply, we can use the naive method for these smaller blocks if Strassen doesn't apply directly or if the overhead is too high.
    #   Actually, we can use Strassen for any 2x2 * 2xk by splitting B into 2x2 and 2x(k-2).
    #   Let's just use the naive method for the non-2x2x2 blocks to keep it simple and correct, and Strassen for the 2x2x2 blocks.
    #   Wait, the task asks for a rank-minimal *exact* decomposition. The seed is 50.
    #   If I use Strassen for the two 2x2x2 blocks (A1B1 and A2B2), and naive for the rest, what is the rank?
    
    # Block 1: A1 (2x2) * B1 (2x5).
    #   B1 = [B1a B1b] where B1a is 2x2, B1b is 2x3.
    #   A1*B1a uses Strassen (rank 7).
    #   A1*B1b uses naive (rank 2*2*3 = 12)? Or can we do better?
    #   Actually, let's just use naive for everything except where Strassen clearly helps.
    #   Strassen helps for 2x2 * 2x2.
    
    # Let's analyze the blocks:
    # A is 2x5. B is 5x5.
    # Split indices j (0..4) into sets: {0,1}, {2,3}, {4}.
    # Term 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x5)
    # Term 2: A[:, 2:4] * B[2:4, :] (2x2 * 2x5)
    # Term 3: A[:, 4:5] * B[4:5, :] (2x1 * 1x5)
    
    # For Term 1 (2x2 * 2x5):
    #   Split B[0:2, :] into B[0:2, 0:2] (2x2) and B[0:2, 2:5] (2x3).
    #   A1 * B1a (2x2 * 2x2): Use Strassen (rank 7).
    #   A1 * B1b (2x2 * 2x3): Use naive (rank 2*2*3 = 12).
    #   Total rank for Term 1: 7 + 12 = 19.
    
    # For Term 2 (2x2 * 2x5):
    #   Split B[2:4, :] into B[2:4, 0:2] (2x2) and B[2:4, 2:5] (2x3).
    #   A2 * B2a (2x2 * 2x2): Use Strassen (rank 7).
    #   A2 * B2b (2x2 * 2x3): Use naive (rank 2*2*3 = 12).
    #   Total rank for Term 2: 7 + 12 = 19.
    
    # For Term 3 (2x1 * 1x5):
    #   Naive rank: 2*1*5 = 10.
    #   Can we do better? 2x1 * 1x5 is just outer product of a 2-vector and a 5-vector?
    #   No, it's element-wise product of the single column of A and the single row of B, then summed?
    #   A3 is 2x1. B3 is 1x5. A3*B3 is 2x5.
    #   C[i,k] += A3[i,0] * B3[0,k].
    #   This is an outer product of A3 and B3.
    #   The tensor for outer product of 2x1 and 1x5 has rank 1?
    #   Let's check.
    #   T[i, j, k] where j is fixed (j=4).
    #   U[i*m+4, r] V[4*p+k, r] W[k*n+i, r].
    #   We need sum_r U[i, r] V[k, r] W[k, i] = A3[i,0] B3[0,k].
    #   If we set rank 1:
    #   U[i, 0] = A3[i,0] (which is 1 in the basis)
    #   V[k, 0] = B3[0,k] (which is 1 in the basis)
    #   W[k, i, 0] = 1.
    #   Sum = 1*1*1 = 1.
    #   Wait, the coefficients U, V, W are constants for the basis elements.
    #   In the tensor decomposition, U, V, W are the structure constants.
    #   For the outer product of a vector u and v, the rank is 1.
    #   So Term 3 has rank 1?
    #   Let's verify.
    #   The contribution to C[i,k] from Term 3 is A[i,4] * B[4,k].
    #   This is a product of two input variables.
    #   In the tensor T, the component T[i, 4, k] for output (i,k) is 1.
    #   We need to represent this single "1" in the tensor.
    #   Actually, the tensor is sparse.
    #   For a specific (i,j,k), the value is 1 if the output is (i,k).
    #   For Term 3, j=4. The non-zero entries are at (i, 4, k) for all i in {0,1}, k in {0..4}.
    #   There are 10 such entries.
    #   Can they be covered by rank 1?
    #   Rank 1 means U[i,4,r] V[4,k,r] W[k,i,r] sums to 1 for all i,k.
    #   If r=1, U[i,4,1] V[4,k,1] W[k,i,1] = 1.
    #   Let U[i,4,1] = 1, V[4,k,1] = 1, W[k,i,1] = 1.
    #   Then sum is 1.
    #   Does this interfere with other terms?
    #   The decomposition is a sum of rank-1 tensors.
    #   The total tensor is the sum of the tensors for each term.
    #   Since the supports (i,j,k) are disjoint for different j blocks, we can just add the ranks.
    #   So Term 3 contributes rank 1.
    
    # Total Rank = 19 + 19 + 1 = 39.
    # This is significantly better than 50.
    
    # Let's double check if we can do better for 2x2 * 2x3.
    # Naive is 12.
    # Is there a lower rank for 2x2 * 2x3?
    # R(2,2,3) >= ?
    # Lower bound: R(2,2,3) >= R(2,2,2) = 7? No, that's not a lower bound.
    # R(2,2,3) is the rank of the tensor for multiplying 2x2 by 2x3.
    # It is known that R(2,2,3) = 11? Or 12?
    # Actually, R(2,2,3) can be done with 11 multiplications?
    # Let's stick with naive for simplicity and correctness, unless I can easily implement 11.
    # 39 is already a big improvement.
    
    # Let's implement the decomposition.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_rank1(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        # Append to U, V, W
        # U is list of lists. We are building columns.
        # It's easier to build lists of rows and then transpose, or just append to columns.
        # Let's use numpy arrays or lists.
        # Since we need to return lists, let's build them.
        pass

    # Let's construct the matrices U, V, W as lists of lists.
    # Rows of U: n*m = 10.
    # Rows of V: m*p = 25.
    # Rows of W: p*n = 10.
    
    U = [[0] * 1 for _ in range(n * m)]
    V = [[0] * 1 for _ in range(m * p)]
    W = [[0] * 1 for _ in range(p * n)]
    
    # Helper to add a rank-1 component
    def add_component(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        U[u_idx].append(u_val)
        V[v_idx].append(v_val)
        W[w_idx].append(w_val)
        rank += 1

    # Strassen for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # A indices: (i, j) -> i in {0,1}, j in {0,1}
    # B indices: (j, k) -> j in {0,1}, k in {0,1}
    # Output: C (2x2) -> (i, k)
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # Let base_i, base_j, base_k be the offsets for the block.
    
    def strassen_block(base_i, base_j, base_k):
        # base_i: offset in i (0 or 1)
        # base_j: offset in j (0..4)
        # base_k: offset in k (0..4)
        
        # A indices: (base_i+0, base_j+0), (base_i+0, base_j+1), (base_i+1, base_j+0), (base_i+1, base_j+1)
        # B indices: (base_j+0, base_k+0), (base_j+0, base_k+1), (base_j+1, base_k+0), (base_j+1, base_k+1)
        # C indices: (base_i+0, base_k+0), (base_i+0, base_k+1), (base_i+1, base_k+0), (base_i+1, base_k+1)
        
        # M1
        # (A00 + A11)(B00 + B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C01 (1), C10 (1), C11 (1)
        # Wait, this is not a single rank-1 term. It's a sum.
        # We need to decompose M1 into rank-1 terms? No, M1 is one multiplication.
        # In the tensor decomposition, each multiplication corresponds to one rank-1 term.
        # So for M1, we have one rank-1 term.
        # U values: 1 for A00, 1 for A11, 0 else.
        # V values: 1 for B00, 1 for B11, 0 else.
        # W values: 1 for C00, 1 for C01, 1 for C10, 1 for C11?
        # No. The W factor multiplies the product.
        # The product is M1.
        # M1 contributes to C00, C01, C10, C11 with coefficients 1, 1, 1, 1?
        # C00 += M1
        # C01 += M1
        # C10 += M1
        # C11 += M1
        # So W should be 1 for all these C indices.
        
        # Let's verify the index mapping.
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        
        # M1:
        # U: u(base_i, base_j) = 1, u(base_i+1, base_j+1) = 1
        # V: v(base_j, base_k) = 1, v(base_j+1, base_k+1) = 1
        # W: w(base_k, base_i) = 1, w(base_k+1, base_i) = 1, w(base_k, base_i+1) = 1, w(base_k+1, base_i+1) = 1
        
        # We need to add this as ONE rank-1 term.
        # But U, V, W are vectors of length rank.
        # So we append a new column to U, V, W.
        
        # Initialize the new column with 0s.
        new_u = [0] * (n * m)
        new_v = [0] * (m * p)
        new_w = [0] * (p * n)
        
        # Set values
        new_u[(base_i) * m + (base_j)] = 1
        new_u[(base_i + 1) * m + (base_j + 1)] = 1
        
        new_v[(base_j) * p + (base_k)] = 1
        new_v[(base_j + 1) * p + (base_k + 1)] = 1
        
        new_w[(base_k) * n + (base_i)] = 1
        new_w[(base_k + 1) * n + (base_i)] = 1
        new_w[(base_k) * n + (base_i + 1)] = 1
        new_w[(base_k + 1) * n + (base_i + 1)] = 1
        
        U.append(new_u)
        V.append(new_v)
        W.append(new_w)
        rank += 1

    # We will call strassen_block for the two 2x2x2 blocks.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Wait, B is 5x5. The columns of B are k.
    # In Term 1, we used B[0:2, 0:2]. So k is in {0,1}.
    # In Term 2, we used B[2:4, 0:2]. So k is in {0,1}.
    # So yes, base_k is 0 for both.
    strassen_block(0, 2, 0)
    
    # Now for the naive parts.
    # Term 1 remainder: A1 (2x2) * B1b (2x3).
    # j in {0,1}, k in {2,3,4}.
    # Naive: for each i, j, k, add a rank-1 term.
    # i in {0,1}, j in {0,1}, k in {2,3,4}.
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                new_u = [0] * (n * m)
                new_v = [0] * (m * p)
                new_w = [0] * (p * n)
                
                new_u[i * m + j] = 1
                new_v[j * p + k] = 1
                new_w[k * n + i] = 1
                
                U.append(new_u)
                V.append(new_v)
                W.append(new_w)
                rank += 1
                
    # Term 2 remainder: A2 (2x2) * B2b (2x3).
    # j in {2,3}, k in {2,3,4}.
    for i in range(2):
        for j in range(2, 4):
            for k in range(2, 5):
                new_u = [0] * (n * m)
                new_v = [0] * (m * p)
                new_w = [0] * (p * n)
                
                new_u[i * m + j] = 1
                new_v[j * p + k] = 1
                new_w[k * n + i] = 1
                
                U.append(new_u)
                V.append(new_v)
                W.append(new_w)
                rank += 1
                
    # Term 3: A3 (2x1) * B3 (1x5).
    # j = 4.
    # i in {0,1}, k in {0,1,2,3,4}.
    # This is an outer product. Rank 1.
    # We can represent it with a single rank-1 term.
    # U: 1 for i=0, j=4 and i=1, j=4.
    # V: 1 for j=4, k=0..4.
    # W: 1 for k=0..4, i=0..1.
    
    new_u = [0] * (n * m)
    new_v = [0] * (m * p)
    new_w = [0] * (p * n)
    
    for i in range(2):
        new_u[i * m + 4] = 1
        
    for k in range(5):
        new_v[4 * p + k] = 1
        
    for k in range(5):
        for i in range(2):
            new_w[k * n + i] = 1
            
    U.append(new_u)
    V.append(new_v)
    W.append(new_w)
    rank += 1
    
    # Total rank:
    # Strassen 1: 1
    # Strassen 2: 1
    # Naive 1: 2 * 2 * 3 = 12
    # Naive 2: 2 * 2 * 3 = 12
    # Outer: 1
    # Total: 1 + 1 + 12 + 12 + 1 = 27.
    
    # Wait, my previous calculation was 19+19+1 = 39.
    # Why the difference?
    # Ah, I said Strassen rank 7.
    # But in my implementation above, I only added ONE rank-1 term for Strassen!
    # Strassen uses 7 multiplications, so it should add 7 rank-1 terms.
    # My `strassen_block` function only adds 1 term. That is WRONG.
    
    # I need to implement the 7 terms of Strassen.
    
    # Let's restart the construction with the correct Strassen implementation.
    
    U = []
    V = []
    W = []
    rank = 0
    
    def add_rank1(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal rank
        new_u = [0] * (n * m)
        new_v = [0] * (m * p)
        new_w = [0] * (p * n)
        
        for idx, val in u_indices_vals:
            new_u[idx] = val
        for idx, val in v_indices_vals:
            new_v[idx] = val
        for idx, val in w_indices_vals:
            new_w[idx] = val
            
        U.append(new_u)
        V.append(new_v)
        W.append(new_w)
        rank += 1
        
    def strassen_block(base_i, base_j, base_k):
        # base_i: 0
        # base_j: 0 or 2
        # base_k: 0
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = [(base_i * m + base_j, 1), ((base_i + 1) * m + (base_j + 1), 1)]
        v1 = [(base_j * p + base_k, 1), ((base_j + 1) * p + (base_k + 1), 1)]
        w1 = [(base_k * n + base_i, 1), ((base_k + 1) * n + base_i, 1), (base_k * n + (base_i + 1), 1), ((base_k + 1) * n + (base_i + 1), 1)]
        add_rank1(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = [((base_i + 1) * m + base_j, 1), ((base_i + 1) * m + (base_j + 1), 1)]
        v2 = [(base_j * p + base_k, 1)]
        w2 = [(base_k * n + (base_i + 1), 1)] # C10
        # Wait, C10 = M2 + M4. So M2 contributes to C10.
        # C10 index: k=0, i=1 -> 0*n + 1 = 1.
        # base_k=0, base_i=0. So w index is 1. Correct.
        add_rank1(u2, v2, w2)
        
        # M3 = A00(B01 - B11)
        u3 = [(base_i * m + base_j, 1)]
        v3 = [(base_j * p + (base_k + 1), 1), ((base_j + 1) * p + (base_k + 1), -1)]
        w3 = [((base_k + 1) * n + base_i, 1)] # C01
        # C01 index: k=1, i=0 -> 1*n + 0 = 2.
        # base_k=0, base_i=0. So w index is 2. Correct.
        add_rank1(u3, v3, w3)
        
        # M4 = A11(B10 - B00)
        u4 = [((base_i + 1) * m + (base_j + 1), 1)]
        v4 = [((base_j + 1) * p + base_k, 1), (base_j * p + base_k, -1)]
        w4 = [(base_k * n + (base_i + 1), 1), (base_k * n + base_i, 1)] # C10 and C00
        # C10: 1, C00: 0.
        add_rank1(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = [(base_i * m + base_j, 1), (base_i * m + (base_j + 1), 1)]
        v5 = [((base_j + 1) * p + (base_k + 1), 1)]
        w5 = [((base_k + 1) * n + (base_i + 1), 1), (base_k * n + (base_i + 1), -1)] # C11 and C10
        # C11 = M1 - M2 + M3 + M6.
        # C10 = M2 + M4.
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M5 contributes to C01 and C11?
        # C01 = M3 + M5.
        # C11 = M1 - M2 + M3 + M6.
        # So M5 contributes to C01 and ... wait.
        # Let's check standard Strassen.
        # M5 = (A00 + A01)B11.
        # C01 = M3 + M5.
        # C11 = M1 - M2 + M3 + M6.
        # So M5 does NOT contribute to C11.
        # My previous w5 was wrong.
        w5 = [((base_k + 1) * n + base_i, 1)] # C01
        add_rank1(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = [((base_i + 1) * m + base_j, 1), (base_i * m + base_j, -1)]
        v6 = [(base_j * p + base_k, 1), (base_j * p + (base_k + 1), 1)]
        w6 = [((base_k + 1) * n + (base_i + 1), 1)] # C11
        add_rank1(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = [(base_i * m + (base_j + 1), 1), ((base_i + 1) * m + (base_j + 1), -1)]
        v7 = [((base_j + 1) * p + base_k, 1), ((base_j + 1) * p + (base_k + 1), 1)]
        w7 = [(base_k * n + base_i, 1), ((base_k + 1) * n + (base_i + 1), 1)] # C00 and C11
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 + M6.
        # So M7 contributes to C00 and ... wait.
        # C00 has M7.
        # C11 does not have M7.
        # Let's re-verify Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 only contributes to C00.
        w7 = [(base_k * n + base_i, 1)] # C00
        add_rank1(u7, v7, w7)
        
    # Call Strassen for the two blocks
    strassen_block(0, 0, 0)
    strassen_block(0, 2, 0)
    
    # Naive parts
    # Term 1 remainder: j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_rank1([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)])
                
    # Term 2 remainder: j in {2,3}, k in {2,3,4}
    for i in range(2):
        for j in range(2, 4):
            for k in range(2, 5):
                add_rank1([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)])
                
    # Term 3: j=4
    add_rank1([(0 * m + 4, 1), (1 * m + 4, 1)], 
              [(4 * p + 0, 1), (4 * p + 1, 1), (4 * p + 2, 1), (4 * p + 3, 1), (4 * p + 4, 1)],
              [(0 * n + 0, 1), (0 * n + 1, 1), (1 * n + 0, 1), (1 * n + 1, 1), (2 * n + 0, 1), (2 * n + 1, 1), (3 * n + 0, 1), (3 * n + 1, 1), (4 * n + 0, 1), (4 * n + 1, 1)])
              
    # Total rank:
    # Strassen 1: 7
    # Strassen 2: 7
    # Naive 1: 12
    # Naive 2: 12
    # Outer: 1
    # Total: 39.
    
    return {"U": U, "V": V, "W": W}
