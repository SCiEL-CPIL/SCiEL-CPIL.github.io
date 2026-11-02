<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # A is n x m = 2x2, B is m x p = 2x12, C is n x p = 2x12
    # Partition p columns into pairs: 6 pairs
    num_pairs = p // 2  # 6
    
    # Strassen 7 products for 2x2 * 2x2:
    # Each product is (linear combo of A entries) * (linear combo of B entries)
    # A indices: A[0][0], A[0][1], A[1][0], A[1][1] -> U rows: 0,1,2,3 (i*m+j)
    # B entries per block: B[0][col], B[1][col] -> V rows: 0*p+col, 1*p+col
    # C output per block: C[0][col], C[1][col] -> W rows: col*n+0, col*n+1
    
    # Strassen formulas (for 2x2 * 2x2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each block t, columns are 2*t and 2*t+1
    # B11 = B[0][2t], B12 = B[0][2t+1], B21 = B[1][2t], B22 = B[1][2t+1]
    # C11 = C[0][2t], C12 = C[0][2t+1], C21 = C[1][2t], C22 = C[1][2t+1]
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # A element to U row mapping: A[i][j] -> row i*m+j
    # A[0][0]->0, A[0][1]->1, A[1][0]->2, A[1][1]->3
    
    # B element to V row mapping: B[j][k] -> row j*p+k
    # B[0][k]->k, B[1][k]->p+k = 12+k
    
    # C element to W row mapping: C[i][k] -> W[k*n+i] (trace convention)
    # C[0][k]->k*2+0 = 2k, C[1][k]->k*2+1 = 2k+1
    
    # Define the 7 Strassen products as (A_coeffs, B_coeffs)
    # A_coeffs: dict from A[i][j] to coefficient
    # B_coeffs: dict from B[j][k] to coefficient (k relative to block)
    
    # Product definitions (A_part, B_part) where each is a list of (element, coeff)
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    products = [
        # M1 = (A11+A22)(B11+B22)
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A21+A22)B11
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        # M3 = A11(B12-B22)
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # M4 = A22(B21-B11)
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        # M5 = (A11+A12)B22
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M6 = (A21-A11)(B11+B12)
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # M7 = (A12-A22)(B21+B22)
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    # Output contributions: for each product s, which C elements get which coefficient
    # C elements: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    # M1: C11+=1, C22+=1
    # M2: C21+=1, C22+=-1
    # M3: C12+=1, C22+=1
    # M4: C11+=1, C21+=1
    # M5: C11+=-1, C12+=1
    # M6: C22+=1
    # M7: C11+=1
    
    output_contribs = [
        # For product 0 (M1): C11+=1, C22+=1
        [((0,0), 1), ((1,1), 1)],
        # For product 1 (M2): C21+=1, C22+=-1
        [((1,0), 1), ((1,1), -1)],
        # For product 2 (M3): C12+=1, C22+=1
        [((0,1), 1), ((1,1), 1)],
        # For product 3 (M4): C11+=1, C21+=1
        [((0,0), 1), ((1,0), 1)],
        # For product 4 (M5): C11+=-1, C12+=1
        [((0,0), -1), ((0,1), 1)],
        # For product 5 (M6): C22+=1
        [((1,1), 1)],
        # For product 6 (M7): C11+=1
        [((0,0), 1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t  # starting column in B and C
        for s in range(7):
            # Set U entries (A coefficients)
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj  # i*m+j
                U[u_row][col] = coeff
            
            # Set V entries (B coefficients)
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel  # actual column in B
                v_row = bj * p + b_col  # j*p+k
                V[v_row][col] = coeff
            
            # Set W entries (C output coefficients)
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel  # actual column in C
                w_row = c_col * n + ci  # k*n+i (trace convention)
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = 1 if i*m+j and j*p+k produce C[k*n+i]
    # T[i,j,k] = 1 iff the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # The tensor T has T[u_row, v_row, w_row] = 1 where u_row=i*m+j, v_row=j*p+k, w_row=k*n+i
    # This holds when the j index matches: A[i,j] * B[j,k] = C[i,k]
    
    # Check: sum_r U[u,r] * V[v,r] * W[w,r] should equal T[u,v,w]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k) triple: u=i*m+j, v=j*p+k, w=k*n+i
    # T[u,v,w] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                # sum over r of U[u_row,r] * V[v_row,r] * W[w_row,r] should be 1
                val = np.dot(U_arr[u_row], V_arr[v_row] * W_arr[w_row])
                if abs(val - 1.0) > 1e-10:
                    return False
    # Also check that non-tensor entries are zero
    # For any (u,v,w) not in the tensor, sum should be 0
    # u = i*m+j, v = j'*p+k, w = k'*n+i'
    # Tensor entry exists when j==j' and i==i' and k==k'
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            u_row = i * m + j
                            v_row = j2 * p + k
                            w_row = k2 * n + i2
                            # Skip if this is a valid tensor entry
                            if j == j2 and i == i2 and k == k2:
                                continue
                            val = np.dot(U_arr[u_row], V_arr[v_row] * W_arr[w_row])
                            if abs(val) > 1e-10:
                                return False
    return True

def _strassen_block(U, V, W, base_col, n, m, p):
    """Add one Strassen 2x2*2x2 block starting at column base_col. Returns rank used."""
    # Strassen formulas for 2x2 * 2x2
    # A: rows 0,1,2,3 = A[0,0],A[0,1],A[1,0],A[1,1]
    # B within block: B[0,base_col], B[0,base_col+1], B[1,base_col], B[1,base_col+1]
    # C within block: C[0,base_col], C[0,base_col+1], C[1,base_col], C[1,base_col+1]
    
    products = [
        # M1 = (A11+A22)(B11+B22)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A21+A22)B11
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        # M3 = A11(B12-B22)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # M4 = A22(B21-B11)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        # M5 = (A11+A12)B22
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M6 = (A21-A11)(B11+B12)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # M7 = (A12-A22)(B21+B22)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    col = 0
    for s in range(7):
        r = col
        for (ai, aj), coeff in products[s][0]:
            U[ai * m + aj][r] = coeff
        for (bj, bk_rel), coeff in products[s][1]:
            b_col = base_col + bk_rel
            V[bj * p + b_col][r] = coeff
        for (ci, ck_rel), coeff in output_contribs[s]:
            c_col = base_col + ck_rel
            W[c_col * n + ci][r] = coeff
        col += 1
    return col

def _m223_block(U, V, W, base_col, n, m, p):
    """Add a rank-5 decomposition for 2x2 * 2x3 block starting at column base_col.
    
    For 2x2 * 2x3:
    A: a11, a12, a21, a22 (U rows 0,1,2,3)
    B: b11, b12, b13, b21, b22, b23 (V rows base_col..base_col+2, p+base_col..p+base_col+2)
    C: c11, c12, c13, c21, c22, c23 (W rows base_col*2..base_col*2+5)
    
    Rank-5 decomposition:
    M1 = (a11+a21) * b11
    M2 = (a11+a21+a22) * b12
    M3 = (a11+a21+a22) * b13
    M4 = a12 * (b21+b22)
    M5 = a12 * b23
    
    c11 = M1 - a21*b11 ... hmm this doesn't work directly.
    
    Let me use a verified rank-5 for M_{2,2,3}:
    M1 = a11*b11
    M2 = a12*b21
    M3 = (a21+a22)*b11
    M4 = (a11+a12)*(b12+b22)
    M5 = (a21+a22)*(b13+b23)
    
    c11 = M1 + M2 = a11*b11 + a12*b21 ✓
    c12 = a11*b12 + a12*b22 = (M4 - a11*b12 - a12*b22 + a11*b12 + a12*b22)... 
    
    This approach of guessing is error-prone. Let me use a systematic construction.
    
    Actually for 2x2*2x3, we can use:
    - 4 products for a 2x2*2x2 on first two columns (using a rank-4 subset of Strassen won't work since rank is 7)
    
    Let me just use 7 products for first 2 cols + 2 products for 3rd col = 9 per group of 3.
    That gives 4*9 = 36 total. Better than 42.
    
    Or: use Strassen (7) for cols 0,1 and simple (2) for col 2: total 9 per group of 3, 4 groups = 36.
    """
    # Use 7 Strassen products for first 2 columns + 2 simple products for 3rd column
    # Total: 9 products per group of 3 columns
    
    # First, add Strassen for columns base_col and base_col+1
    col_offset = 0
    rank_used = _strassen_block(U, V, W, base_col, n, m, p)
    col_offset = rank_used  # 7 columns used
    
    # Now add 2 simple products for column base_col+2
    # C[0][base_col+2] = a11*b13 + a12*b23
    # C[1][base_col+2] = a21*b13 + a22*b23
    # Product 8: a11 * b13 -> C[0][base_col+2] += 1
    # Product 9: a12 * b23 -> C[0][base_col+2] += 1
    # Product 10: a21 * b13 -> C[1][base_col+2] += 1
    # Product 11: a22 * b23 -> C[1][base_col+2] += 1
    # That's 4 more products, total 11 per group. 4*11=44. Worse than 42!
    
    # Better: use 2 products for the 3rd column:
    # M8 = a11 * b13
    # M9 = a12 * b23
    # C[0][base_col+2] = M8 + M9
    # But C[1][base_col+2] = a21*b13 + a22*b23 needs 2 more products
    # Total 4 extra = 11 per group. Still worse.
    
    # Alternative: share products across the 3 columns differently.
    # Use 5 products total for 2x2*2x3:
    # This requires a true rank-5 decomposition which I need to verify.
    
    # For now, let's just return the Strassen 7 for first 2 cols and 4 for last = 11
    # This gives 4*11 = 44 which is worse. Let me just use the original 6*7=42 approach.
    # 
    # Actually, let me try: partition into 3 groups of 4 columns.
    # For 2x2 * 2x4: use 2 Strassen blocks = 14 products per group. 3*14=42. Same.
    #
    # Or: 2 groups of 6 columns. For 2x2*2x6: 3 Strassen blocks = 21 per group. 2*21=42. Same.
    #
    # The fundamental issue is that 2x2*2x2 needs rank 7, so any partition into pairs gives 42.
    # To beat 42, we need cross-column sharing.
    
    # Let me try a different approach: use the fact that A is only 2x2 (4 entries).
    # The multiplication tensor M_{2,2,12} has 2*2*12 = 48 nonzero entries.
    # Lower bound: rank >= max(n*m, m*p, p*n)/max(n,m,p) ... various bounds.
    # Bounded rank: rank >= ceil(n*p/m) = ceil(24/2) = 12. Also rank >= n*m = 4, etc.
    # A better bound: rank >= (n*m*p) / max(n*m, m*p, p*n) = 48/24 = 2. Weak.
    # Coppersmith-Winograd type: for M_{n,n,p}, rank grows.
    
    # For now, stick with rank 42 as it's verified correct.
    # The function should just call _strassen_block 6 times.
    return rank_used

def construct():
    """Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication."""
    n, m, p = DIMENSIONS
    num_pairs = p // 2  # 6
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_pairs):
        base_col = 2 * t
        _strassen_block(U, V, W, base_col, n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize I'm overcomplicating this. Let me reconsider the problem more carefully.

The key question: can we do better than 42? The rank of M_{2,2,p} for p=12. Known results suggest that for small n,m, the rank is approximately n*m*p/max(n,m,p) but with Strassen improvements.

For M_{2,2,p}, a simple bound gives rank >= 2*p (since each output column needs at least 2 products). With Strassen, we get 7*p/2 = 42 for p=12.

Can we share products between adjacent column pairs? The issue is that Strassen's 7 products are "local" to the 2x2 block. To share, we'd need a different decomposition structure.

Let me try a different approach: what if we use a rank-5 decomposition for M_{2,2,3}? If it exists, 4 groups of 3 gives rank 20.

Let me carefully construct and verify a rank-5 for M_{2,2,3}.

For 2×2 · 2×3:
- Inputs: a11, a12, a21, a22, b11, b12, b13, b21, b22, b23
- Outputs: c11=a11·b11+a12·b21, c12=a11·b12+a12·b22, c13=a11·b13+a12·b23, c21=a21·b11+a22·b21, c22=a21·b12+a22·b22, c23=a21·b13+a22·b23

I'll search for a rank-5 decomposition programmatically within the construct function, with a fallback to rank 42.

