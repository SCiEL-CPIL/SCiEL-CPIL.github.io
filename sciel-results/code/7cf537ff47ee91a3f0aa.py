import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Try to find a decomposition with rank < 28
    # Key insight: A is 2x2 (4 scalars), B is 2x8 (16 scalars)
    # The 8 columns of B can share A-side structure
    
    # Strategy: Use a "folded" Strassen approach where we group columns
    # of B and exploit that A coefficients are shared across all 8 columns.
    
    # Let's try rank 24: 3 groups of 8, or some other partition
    # Actually, let's think more carefully.
    
    # For M_{2,2,p}, the rank is known to be:
    # rank(M_{2,2,p}) for p=8: The schoolbook is 32, Strassen gives 28.
    # Can we do better?
    
    # Key idea: Since A has only 4 entries, we can write the computation as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For Strassen on 2x2 blocks, we use 7 multiplications per 2-column block.
    # But maybe we can share some multiplications across blocks.
    
    # Let's try a different approach: express the full 2x2 x 2x8 multiplication
    # using a smaller set of rank-1 tensors.
    
    # The tensor T has components T[i,j,k] where i=0,1; j=0,1; k=0..7
    # T[i,j,k] = 1 if i==j, else 0
    # 
    # A rank-r decomposition means: T[i,j,k] = sum_{s=1}^r U[i,s] * V[j,s] * W[k,s]
    # where U is 2x r (wait, no - U is n*m x r = 4 x r)
    
    # Let me re-read the interface:
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # The decomposition satisfies:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==j else 0
    # For all other (a,b,c) triples:
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # Let's try to find rank 24 by being clever about sharing.
    
    # Approach: Use the fact that for fixed i,j, the equation over k is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j)
    
    # Let's parameterize and try to solve.
    
    # Actually, let me try a constructive approach with rank 24.
    # Group the 8 columns into 4 pairs, but share 1 Strassen term per pair.
    
    # Alternative: Try rank 24 directly by construction.
    # 
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{2,2,4}, is rank 14 or can we do 12?
    # For M_{2,2,8}, is rank 28 or can we do 24?
    
    # Let me try a different decomposition structure.
    # 
    # Idea: Instead of independent Strassen blocks, use a "diagonal" approach
    # where some terms are shared across multiple k-values.
    
    # Let's try: rank 24 = 6 terms * 4 (shared) + 0, or some other split.
    
    # Actually, let me try to verify if rank 24 is possible by attempting
    # a numerical search with integer/rational constraints.
    
    # For now, let's stick with the known rank-28 but try to optimize
    # the construction to ensure correctness, and also try rank 24.
    
    deadline = time.time() + 140
    
    # First, try to construct rank 24
    # 
    # Key insight: The 2x2 matrix A multiplies 8 independent 2x1 column vectors.
    # We can think of this as 8 separate 2x2 x 1 multiplications, but they share A.
    # 
    # A 2x2 x 1 multiplication (A * b where b is 2x1) needs rank 2:
    # c_0 = a_00*b_0 + a_01*b_1
    # c_1 = a_10*b_0 + a_11*b_1
    # This is rank 2 (schoolbook) for each column.
    # 8 columns * 2 = 16? No, that's not right either.
    
    # Let me think again. The full tensor M_{2,2,8}:
    # Input A: 4 scalars (a00, a01, a10, a11)
    # Input B: 16 scalars (b00, b01, ..., b17) where b_{j,k}
    # Output C: 16 scalars (c00, c01, ..., c17) where c_{i,k}
    # c_{i,k} = sum_j a_{i,j} * b_{j,k}
    
    # This is a 4-16-16 tensor.
    
    # Rank of this tensor: We know rank(M_{2,2,2}) = 7.
    # rank(M_{2,2,4}) >= 2*rank(M_{2,2,2}) - something? 
    # Actually, rank is sub-multiplicative in some sense.
    
    # Known results: rank(M_{2,2,n}) for various n.
    # For n=8, the best known upper bound might be 28 (4*7).
    # But can we do better by exploiting the shared A?
    
    # Let me try a specific construction for rank 24.
    # 
    # Suppose we use the following structure:
    # For k=0..7, we have 8 "output pairs" (c_{0,k}, c_{1,k}).
    # 
    # Strassen for one 2x2 block uses 7 terms. If we could share some terms
    # across blocks, we'd reduce the total.
    
    # Let's try: use 6 shared terms + 18 block-specific = 24? 
    # Or 3 groups of 8?
    
    # Actually, let me just try to verify the rank-28 solution works correctly
    # and also attempt a rank-24 construction.
    
    # Rank-28 Strassen construction (verified approach):
    def strassen_228():
        n, m, p = 2, 2, 8
        rank = 28
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Strassen terms for 2x2 x 2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e  
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # i = M1 + M4 - M5 + M7
        # j = M3 + M5
        # k = M2 + M4
        # l = M1 - M2 + M3 + M6
        
        # U coefficients (for A entries a,b,c,d at indices 0,1,2,3)
        # V coefficients (for B entries e,f,g,h at local indices 0,1,2,3)
        # W coefficients (for C entries i,j,k,l at local indices 0,1,2,3)
        
        U_str = [
            [1, 0, 0, 1],  # M1: a+d
            [0, 0, 1, 1],  # M2: c+d
            [1, 0, 0, 0],  # M3: a
            [0, 0, 0, 1],  # M4: d
            [1, 1, 0, 0],  # M5: a+b
            [-1, 0, 1, 0], # M6: c-a
            [0, 1, 0, -1], # M7: b-d
        ]
        V_str = [
            [1, 0, 0, 1],  # M1: e+h
            [1, 0, 0, 0],  # M2: e
            [0, 1, 0, -1], # M3: f-h
            [-1, 0, 1, 0], # M4: g-e
            [0, 0, 0, 1],  # M5: h
            [1, 1, 0, 0],  # M6: e+f
            [0, 0, 1, 1],  # M7: g+h
        ]
        W_str = [
            [1, 0, 0, 1],  # M1: i + l
            [0, 0, 1, -1], # M2: k - l
            [0, 1, 0, 1],  # M3: j + l
            [1, 0, 1, 0],  # M4: i + k
            [-1, 1, 0, 0], # M5: j - i
            [0, 0, 0, 1],  # M6: l
            [1, 0, 0, 0],  # M7: i
        ]
        
        for block in range(4):
            for t in range(7):
                col = block * 7 + t
                for i in range(4):
                    U[i][col] = U_str[t][i]
                
                # V: B is 2x8, local block is columns 2*block, 2*block+1
                # Local indices: 0=(0,2b), 1=(0,2b+1), 2=(1,2b), 3=(1,2b+1)
                for li in range(4):
                    row = li // 2
                    col_local = li % 2
                    global_col = 2 * block + col_local
                    v_idx = row * 8 + global_col
                    V[v_idx][col] = V_str[t][li]
                
                # W: C is 2x8, convention W[k*n+i, r]
                # Local indices: 0=(i=0,k=2b), 1=(i=0,k=2b+1), 2=(i=1,k=2b), 3=(i=1,k=2b+1)
                for li in range(4):
                    i_local = li % 2
                    k_local = li // 2
                    global_k = 2 * block + k_local
                    w_idx = global_k * 2 + i_local
                    W[w_idx][col] = W_str[t][li]
        
        return U, V, W
    
    def verify(U, V, W, n=2, m=2, p=8):
        rank = len(U[0])
        # Check all (a,b,c) triples
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    expected = 1 if (a, b, c) in valid else 0
                    if s != expected:
                        return False
        return True
    
    # Try rank 28 first
    U, V, W = strassen_228()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If rank 28 fails (shouldn't), fallback to schoolbook
    rank = 32
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(8):
                U[i*2+j][col] = 1
                V[j*8+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
