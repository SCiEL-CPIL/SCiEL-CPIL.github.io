import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # We want to decompose the tensor for C = A * B
    # A is 2x2, B is 2x12, C is 2x12
    # Dimensions of tensor factors:
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Strategy:
    # The multiplication C = A * B can be viewed as:
    # For each column j of C: C[:, j] = A * B[:, j]
    # This is a 2x2 * 2x1 = 2x1 multiplication for each j.
    
    # The rank of 2x2 * 2x1 matrix multiplication is 2 (it's just 2 scalar multiplications per row... wait)
    # Actually, C[:,j] = A * B[:,j] where A is 2x2 and B[:,j] is 2x1.
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    # This is rank 2 for each j? No, it's rank 4 for each j if we consider the full tensor.
    # Let's check:
    # The bilinear map (A, b) -> A*b has rank 4 (schoolbook: a*e + b*h, c*e + d*h -> 4 terms)
    # Wait, the rank of 2x2 * 2x1 is exactly 4?
    # Let A = [[a,b],[c,d]], b = [e,h].
    # C[0] = a*e + b*h
    # C[1] = c*e + d*h
    # Rank-1 terms:
    # 1. a*e -> C[0]
    # 2. b*h -> C[0]
    # 3. c*e -> C[1]
    # 4. d*h -> C[1]
    # Yes, rank 4 for each column.
    
    # But we can do better by grouping columns.
    # If we group 2 columns: 2x2 * 2x2 -> rank 7 (Strassen)
    # Total rank = 6 groups * 7 = 42. This is the current approach.
    
    # Can we do better?
    # The tensor is T_{i,j,k,l} = delta_{i,k} (where i is row of A, k is row of C, j is col of A, l is col of B... wait)
    # Standard indexing:
    # C_{r,c} = sum_{k} A_{r,k} B_{k,c}
    # r in {0,1}, c in {0..11}, k in {0,1}
    #
    # The tensor is T_{r,c,k} A_{r,k} B_{k,c}
    # This is a 3-tensor of size 2*24*2 = 96 elements.
    #
    # The rank of the matrix multiplication tensor for (n,m) x (m,p) is denoted R(n,m,p).
    # R(2,2,12) = ?
    #
    # We know R(2,2,2) = 7.
    # R(2,2,1) = 4.
    #
    # Using the "block" argument:
    # R(2,2, 2k) >= k * R(2,2,2) is NOT necessarily true, but R(2,2, 2k) <= k * R(2,2,2).
    # So R(2,2,12) <= 6 * 7 = 42.
    #
    # Lower bound:
    # R(2,2,12) >= ?
    #
    # Let's try to find a decomposition with rank < 42.
    #
    # Consider the structure:
    # C_{r,c} = A_{r,0} B_{0,c} + A_{r,1} B_{1,c}
    #
    # We can write this as:
    # C = A_0 * B_0 + A_1 * B_1  (where A_0 is column 0 of A, B_0 is row 0 of B, etc.)
    # This is a sum of two rank-1 terms in the space of (A, B) -> C?
    # No, A_0 * B_0 is not a scalar product, it's an outer product.
    #
    # Let's look at the tensor T in the basis of the monomials.
    # U factors: vectors in A-space (dim 4)
    # V factors: vectors in B-space (dim 24)
    # W factors: vectors in C-space (dim 24)
    #
    # A rank-r decomposition is a sum of r rank-1 tensors.
    # T = sum_{t=1}^r u_t \otimes v_t \otimes w_t
    #
    # Let's try to use the fact that the columns of B are independent.
    #
    # Alternative approach:
    # Is R(2,2,12) = 4 * 12 / 2 = 24? (For large m, R(n,m,p) ~ nmp/m = np? No, R(n,m,p) >= max(nm, mp, np)?)
    # Actually, R(n,m,p) >= max(n*m, m*p, p*n) is a lower bound?
    # R(2,2,12) >= max(4, 24, 24) = 24.
    #
    # Can we achieve rank 24?
    # Rank 24 would be the "trivial" bound.
    #
    # Let's try to construct a rank-24 decomposition.
    #
    # If we can find U (4x24), V (24x24), W (24x24) such that T = U V^T W^T (in some sense), or rather T_{ijk} = sum_t U_{i,t} V_{j,t} W_{k,t}.
    #
    # Let's look at the slice for a specific column c of B.
    # The contribution to C[:, c] is A * B[:, c].
    # Let B[:, c] = [x, y].
    # C[:, c] = A * [x, y]^T = [a*x + b*y, c*x + d*y]^T.
    #
    # We need to express the vector [a*x+b*y, c*x+d*y] as a sum of r_t terms.
    # Each term is u_t(a) * v_t(x,y) * w_t.
    #
    # If we fix x,y and vary a,b,c,d, the map is linear.
    #
    # Let's try to share the V and W factors across columns.
    #
    # Suppose we use the same set of 24 V-factors and 24 W-factors for all columns?
    # No, V depends on the column index of B.
    #
    # Let's try a specific structure.
    #
    # For each column c, we have 2 B-variables: B[0,c] and B[1,c].
    # Let's call them x_c and y_c.
    #
    # We want to compute C[0,c] = a x_c + b y_c and C[1,c] = c x_c + d y_c.
    #
    # Consider the following rank-4 decomposition for a single column c:
    # Term 1: a x_c -> C[0,c]. u=[1,0,0,0], v=[1,0], w=[1,0] (local)
    # Term 2: b y_c -> C[0,c]. u=[0,1,0,0], v=[0,1], w=[1,0] (local)
    # Term 3: c x_c -> C[1,c]. u=[0,0,1,0], v=[1,0], w=[0,1] (local)
    # Term 4: d y_c -> C[1,c]. u=[0,0,0,1], v=[0,1], w=[0,1] (local)
    #
    # This uses 4 terms per column. Total rank = 4 * 12 = 48.
    # This is worse than Strassen (42).
    #
    # Why is Strassen better? Because it mixes columns.
    # By mixing 2 columns, we reduce the rank from 4*2=8 to 7.
    #
    # Can we mix more columns?
    # R(2,2,4)?
    # Schoolbook: 4*2 = 8.
    # Strassen (2 groups of 2): 7*2 = 14.
    # Can we do 12? 11? 10?
    #
    # It is known that R(2,2,4) = 14? Or maybe less?
    # Actually, R(n,m,p) is not fully understood.
    #
    # However, there is a lower bound: R(n,m,p) >= n*m*p / max(n,m,p) ?
    # For 2,2,12: 48/12 = 4.
    #
    # Let's stick with the Strassen decomposition for pairs, which gives 42.
    # Is it possible to do better?
    #
    # Let's check if R(2,2,12) < 42.
    # If we could find a rank-24 decomposition, it would be a breakthrough.
    #
    # Given the time limit, I will stick with the rank-42 Strassen decomposition,
    # but I will implement it cleanly using NumPy to ensure speed and correctness.
    
    # Rank-42 decomposition using Strassen for 2x2 blocks.
    rank = 42
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((24, rank), dtype=float)
    
    # Strassen 7 products for 2x2 x 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    #
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    #
    # C = [[M1 + M4 - M5 + M7, M3 + M5],
    #      [M2 + M4,             M1 - M2 + M3 + M6]]
    
    # U columns for each of the 7 Strassen products
    # A indices: 0=a, 1=b, 2=c, 3=d
    
    u_strassen = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # V factors for each of the 7 Strassen products
    # B indices (local for 2x2 block): 0=e, 1=f, 2=g, 3=h
    
    v_strassen = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, -1, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # W factors for each of the 7 Strassen products
    # C indices (local for 2x2 block): 0=C00, 1=C01, 2=C10, 3=C11
    
    w_strassen = [
        [1, 0, 0, -1], # M1: C00 + 1, C11 - 1
        [0, 0, 1, 1],  # M2: C10 + 1, C11 + 1 (Wait, M2 is in C10 and C11? No.)
        # Let's re-verify C formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00 (+1), C11 (-1) -> [1, 0, 0, -1]
        # M2: C10 (+1), C11 (-1) -> [0, 0, 1, -1]
        # M3: C01 (+1), C11 (+1) -> [0, 1, 0, 1]
        # M4: C00 (+1), C10 (+1) -> [1, 0, 1, 0]
        # M5: C00 (-1), C01 (+1) -> [-1, 1, 0, 0]
        # M6: C11 (+1) -> [0, 0, 0, 1]
        # M7: C00 (+1) -> [1, 0, 0, 0]
        
        [1, 0, 0, -1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    # Fill U, V, W
    # We have 6 blocks of 2 columns each.
    num_blocks = 6
    
    for block in range(num_blocks):
        # Block t uses B columns 2t and 2t+1
        # And C columns 2t and 2t+1
        
        for s in range(7):
            col_idx = block * 7 + s
            
            # U
            U[:, col_idx] = u_strassen[s]
            
            # V
            # Local B indices: 0=e(B[0, 2t]), 1=f(B[0, 2t+1]), 2=g(B[1, 2t]), 3=h(B[1, 2t+1])
            # Global V indices:
            # B[0, j] -> index j
            # B[1, j] -> index 12 + j
            
            v_local = v_strassen[s]
            # e (local 0) -> B[0, 2t] -> global 2t
            V[2 * block, col_idx] = v_local[0]
            # f (local 1) -> B[0, 2t+1] -> global 2t+1
            V[2 * block + 1, col_idx] = v_local[1]
            # g (local 2) -> B[1, 2t] -> global 12 + 2t
            V[12 + 2 * block, col_idx] = v_local[2]
            # h (local 3) -> B[1, 2t+1] -> global 12 + 2t+1
            V[12 + 2 * block + 1, col_idx] = v_local[3]
            
            # W
            # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
            # Global W indices: W[k*n+i] where k=col, i=row
            # C[0, 2t] -> k=2t, i=0 -> index 2t
            # C[0, 2t+1] -> k=2t+1, i=0 -> index 2t+1
            # C[1, 2t] -> k=2t, i=1 -> index 2t+1
            # C[1, 2t+1] -> k=2t+1, i=1 -> index 2t+2
            
            w_local = w_strassen[s]
            # C00 (local 0) -> global 2t
            W[2 * block, col_idx] = w_local[0]
            # C01 (local 1) -> global 2t+1
            W[2 * block + 1, col_idx] = w_local[1]
            # C10 (local 2) -> global 2t+1 (Wait, C[1, 2t] is k=2t, i=1. Index is 2t*2+1? No.
            # W index is k*n + i. n=2.
            # C[0, c] -> c*2 + 0 = 2c
            # C[1, c] -> c*2 + 1 = 2c+1
            
            # So:
            # C[0, 2t] -> 2*(2t) = 4t
            # C[0, 2t+1] -> 2*(2t+1) = 4t+2
            # C[1, 2t] -> 2*(2t)+1 = 4t+1
            # C[1, 2t+1] -> 2*(2t+1)+1 = 4t+3
            
            W[4 * block, col_idx] = w_local[0]     # C[0, 2t]
            W[4 * block + 2, col_idx] = w_local[1] # C[0, 2t+1]
            W[4 * block + 1, col_idx] = w_local[2] # C[1, 2t]
            W[4 * block + 3, col_idx] = w_local[3] # C[1, 2t+1]
            
            # Check time
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
