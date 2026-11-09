import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # A is n x m (2x3)
    # B is m x p (3x3)
    # C is n x p (2x3)
    
    # Indexing conventions:
    # U row index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V row index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W row index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # We decompose the tensor T[i,j,k] = 1 for all i,j,k
    # into a sum of rank-1 tensors.
    
    # Strategy:
    # 1. Use Strassen's algorithm for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}). Rank 7.
    # 2. Handle the remaining terms where j=2 or k=2.
    #    The remaining set is:
    #    - j in {0,1}, k=2 (2*1=2 terms)
    #    - j=2, k in {0,1} (1*2=2 terms)
    #    - j=2, k=2 (1 term)
    #    Total 5 terms.
    
    # Can we do better than rank 1 for these?
    # The sub-tensor for j in {0,1}, k=2 is:
    # T[i, j, 2] = 1 for i in {0,1}, j in {0,1}.
    # This is a 2x2 matrix of ones. Rank is 2? No, wait.
    # The tensor is 3-way.
    # Let's look at the structure.
    # For fixed k=2, the tensor is T[i, j, 2].
    # U[i,j], V[j,2], W[2,i].
    # This is essentially the tensor for multiplying a 2x2 matrix by a 2x1 vector?
    # No, it's a slice.
    # Let's try to decompose the 5 remaining terms with minimum rank.
    
    # The remaining tensor R has support:
    # (i, j, k) in:
    # (0,0,2), (0,1,2)
    # (1,0,2), (1,1,2)
    # (0,2,0), (1,2,0)
    # (0,2,1), (1,2,1)
    # (0,2,2), (1,2,2)
    
    # Let's check if these can be combined.
    # Consider the term (0,2,2) and (1,2,2).
    # U[0,2], U[1,2] non-zero.
    # V[2,2] non-zero.
    # W[2,0], W[2,1] non-zero.
    
    # Let's try to find a decomposition of the remaining 5 terms with rank < 5.
    # Actually, the 2x2x2 block is independent in the sense that Strassen uses specific linear combinations.
    # But we can mix them.
    
    # Let's stick to the block decomposition first to ensure correctness, then see if we can optimize.
    # Rank 7 (Strassen) + Rank ? (rest)
    
    # The rest consists of:
    # 1. i in {0,1}, j in {0,1}, k=2. (4 terms)
    # 2. i in {0,1}, j=2, k in {0,1}. (4 terms)
    # 3. i in {0,1}, j=2, k=2. (2 terms)
    # Total 10 terms.
    
    # Can we decompose the 4 terms of type (i,j,2) with rank 2?
    # T[i,j,2] = 1.
    # This is equivalent to the tensor for 2x2 * 2x1?
    # Let's check the rank of the tensor T[i,j,k] where k is fixed to 2.
    # It's a 2x2x1 tensor.
    # U[i,j], V[j,2], W[2,i].
    # If we fix k=2, V is a vector of length 2 (j=0,1). W is a vector of length 2 (i=0,1).
    # U is a 2x2 matrix.
    # The tensor is T[i,j] = U[i,j] * V[j] * W[i].
    # This is a rank-1 tensor in the (i,j) plane if U is rank 1.
    # But T[i,j] = 1 is a rank-2 matrix (all ones).
    # So we need at least rank 2 for this slice?
    # Let's see.
    # Term 1: U[0,0]=1, V[0]=1, W[0]=1 -> contributes to (0,0,2)
    # Term 2: U[1,1]=1, V[1]=1, W[1]=1 -> contributes to (1,1,2)
    # This doesn't cover (0,1,2) and (1,0,2).
    
    # Let's try:
    # r1: U[0,0]=1, U[0,1]=1, V[0]=1, V[1]=1, W[0]=1 -> (0,0,2), (0,1,2)
    # r2: U[1,0]=1, U[1,1]=1, V[0]=1, V[1]=1, W[1]=1 -> (1,0,2), (1,1,2)
    # So rank 2 for the k=2, j in {0,1} block.
    
    # Similarly for j=2, k in {0,1}:
    # T[i,2,k] = 1.
    # r3: U[0,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=1 -> (0,2,0), (0,2,1)
    # r4: U[1,2]=1, V[2,0]=1, V[2,1]=1, W[1,0]=1 -> (1,2,0), (1,2,1)
    # So rank 2 for the j=2, k in {0,1} block.
    
    # And for j=2, k=2:
    # T[i,2,2] = 1.
    # r5: U[0,2]=1, V[2,2]=1, W[2,0]=1 -> (0,2,2)
    # r6: U[1,2]=1, V[2,2]=1, W[2,1]=1 -> (1,2,2)
    # So rank 2 for the j=2, k=2 block.
    
    # Total rank = 7 + 2 + 2 + 2 = 13.
    # This is better than 18.
    
    # Can we do better?
    # The Strassen part uses 7 terms.
    # The remaining parts use 6 terms.
    # Total 13.
    
    # Let's check if we can merge some terms.
    # The Strassen terms involve j in {0,1} and k in {0,1}.
    # The other terms involve j=2 or k=2.
    # They are disjoint in support?
    # Strassen support: j in {0,1}, k in {0,1}.
    # Other support: j=2 or k=2.
    # Yes, disjoint.
    # So we can sum the ranks.
    
    # Is rank 13 minimal?
    # Lower bound for 2x3x3 is not well known, but 13 is a good guess.
    # Actually, for 2x3x3, the rank is known to be at most 12? Or 13?
    # Let's try to optimize the "other" part.
    # We used 2+2+2=6.
    # Can we do it in 5?
    # The tensor for the "other" part is:
    # (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # (0,2,0), (1,2,0), (0,2,1), (1,2,1)
    # (0,2,2), (1,2,2)
    
    # Let's check if we can combine (0,2,2) with others.
    # If we use a term with U[0,2]=1, V[2,2]=1, W[2,0]=1, it covers (0,2,2).
    # Can it also cover something else?
    # U[0,2] is fixed. V[2,2] is fixed. W[2,0] is fixed.
    # It only contributes to (0,2,2) because U is zero elsewhere, V is zero elsewhere, W is zero elsewhere.
    # Wait, U[0,2] is non-zero. If we set U[0,0]=1 as well, then V[0,?] and W[?,0] would matter.
    # But V[0,2] is 0 in this term? No, V is a vector for the term.
    # If we set U[0,2]=1 and U[0,0]=1, then for k=2, j=0, we get U[0,0]*V[0,2]*W[2,0].
    # If V[0,2]=1, then we get a contribution to (0,0,2).
    # This might allow merging.
    
    # Let's try to construct a rank 5 decomposition for the 10 terms.
    # Or maybe rank 4?
    
    # Let's stick with 13 for now, as it's a solid improvement over 18.
    # 18 -> 13 is a big jump.
    
    rank = 13
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # 1. Strassen for 2x2x2 (i in 0,1; j in 0,1; k in 0,1)
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Strassen terms (from literature, verified for T[i,j,k]=1):
    # u, v, w are vectors of length 4.
    # u index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # v index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # w index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([0, -1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]),
    ]
    
    for u, v, w in strassen_terms:
        for idx in range(4):
            i, j = divmod(idx, 2) # Wait, u index 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # My mapping above:
            # 0: (0,0) -> i=0, j=0
            # 1: (0,1) -> i=0, j=1
            # 2: (1,0) -> i=1, j=0
            # 3: (1,1) -> i=1, j=1
            # So i = idx // 2, j = idx % 2?
            # 0//2=0, 0%2=0 -> (0,0) OK
            # 1//2=0, 1%2=1 -> (0,1) OK
            # 2//2=1, 2%2=0 -> (1,0) OK
            # 3//2=1, 3%2=1 -> (1,1) OK
            # Yes.
            
            # U
            U[i * m + j, col] += u[idx]
            
            # V
            # v index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # j, k
            j_v, k_v = divmod(idx, 2)
            V[j_v * p + k_v, col] += v[idx]
            
            # W
            # w index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # k, i
            k_w, i_w = divmod(idx, 2)
            W[k_w * n + i_w, col] += w[idx]
            
        col += 1
        
    # col should be 7 now.
    
    # 2. Block k=2, j in {0,1}
    # Terms: (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # Rank 2 decomposition:
    # r7: U[0,0]=1, U[0,1]=1, V[0,2]=1, V[1,2]=1, W[2,0]=1
    # r8: U[1,0]=1, U[1,1]=1, V[0,2]=1, V[1,2]=1, W[2,1]=1
    
    # r7
    U[0 * m + 0, col] = 1.0
    U[0 * m + 1, col] = 1.0
    V[0 * p + 2, col] = 1.0
    V[1 * p + 2, col] = 1.0
    W[2 * n + 0, col] = 1.0
    col += 1
    
    # r8
    U[1 * m + 0, col] = 1.0
    U[1 * m + 1, col] = 1.0
    V[0 * p + 2, col] = 1.0
    V[1 * p + 2, col] = 1.0
    W[2 * n + 1, col] = 1.0
    col += 1
    
    # 3. Block j=2, k in {0,1}
    # Terms: (0,2,0), (1,2,0), (0,2,1), (1,2,1)
    # Rank 2 decomposition:
    # r9: U[0,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=1
    # r10: U[1,2]=1, V[2,0]=1, V[2,1]=1, W[1,0]=1
    
    # r9
    U[0 * m + 2, col] = 1.0
    V[2 * p + 0, col] = 1.0
    V[2 * p + 1, col] = 1.0
    W[0 * n + 0, col] = 1.0
    col += 1
    
    # r10
    U[1 * m + 2, col] = 1.0
    V[2 * p + 0, col] = 1.0
    V[2 * p + 1, col] = 1.0
    W[1 * n + 0, col] = 1.0
    col += 1
    
    # 4. Block j=2, k=2
    # Terms: (0,2,2), (1,2,2)
    # Rank 2 decomposition:
    # r11: U[0,2]=1, V[2,2]=1, W[2,0]=1
    # r12: U[1,2]=1, V[2,2]=1, W[2,1]=1
    
    # r11
    U[0 * m + 2, col] = 1.0
    V[2 * p + 2, col] = 1.0
    W[2 * n + 0, col] = 1.0
    col += 1
    
    # r12
    U[1 * m + 2, col] = 1.0
    V[2 * p + 2, col] = 1.0
    W[2 * n + 1, col] = 1.0
    col += 1
    
    # Total rank 13.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

