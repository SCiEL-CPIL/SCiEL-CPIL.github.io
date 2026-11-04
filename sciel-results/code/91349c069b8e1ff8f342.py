"""Matrix multiplication 3,4,6 - Strassen-optimized tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll use Strassen on 2x2x2 blocks where possible
    # For 3x4x6: we can fit 3 (in i) * 2 (in j) * 2 (in k) = 12 disjoint 2x2x2 blocks
    # Plus remaining: i=2 (1 row), j=3 (1 col), k=3,4,5 (3 cols)
    # 
    # Actually, let me think more carefully about the packing.
    # i: 0,1,2 (3 values) -> 1 full 2-block + 1 leftover
    # j: 0,1,2,3 (4 values) -> 2 full 2-blocks
    # k: 0,1,2,3,4,5 (6 values) -> 3 full 2-blocks
    #
    # So 2x2x2 blocks: 1*2*3 = 6 blocks, each saving 1 -> rank 72-6=66
    #
    # Let me enumerate:
    # i-blocks: {0,1} only (i=2 is leftover)
    # j-blocks: {0,1}, {2,3}
    # k-blocks: {0,1}, {2,3}, {4,5}
    # Total: 1*2*3 = 6 Strassen blocks
    #
    # Remaining schoolbook terms:
    # - i=2, all j, all k: 1*4*6 = 24 terms
    # - i in {0,1}, j=3, all k: 2*1*6 = 12 terms  
    # - i in {0,1}, j in {0,1,2}, k=5: wait, let me be more careful
    #
    # Let me just build it directly:
    # For each (i,j,k) in the full 3x4x6 grid:
    #   If (i,j,k) is part of a 2x2x2 block, use Strassen
    #   Otherwise, use schoolbook
    
    # Strassen 2x2x2 decomposition:
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For our tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # In a 2x2x2 block with i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}:
    # Let's map: a=A[i0,j0]=U[i0*m+j0], b=A[i0,j0+1]=U[i0*m+j0+1]
    #            c=A[i0+1,j0]=U[i0+1*m+j0], d=A[i0+1,j0+1]=U[i0+1*m+j0+1]
    #            e=B[j0,k0]=V[j0*p+k0], f=B[j0,k0+1]=V[j0*p+k0+1]
    #            g=B[j0+1,k0]=V[j0+1*p+k0], h=B[j0+1,k0+1]=V[j0+1*p+k0+1]
    #
    # The 7 Strassen factors contribute to:
    # m1: U has (a+d) at i0,j0 and i0+1,j0; V has (e+h) at j0,k0 and j0+1,k0; W has 1s at all 4 (k,i) positions
    # This is getting complex. Let me just assign each Strassen factor a unique column.
    
    # Approach: 
    # 1. Identify all 2x2x2 blocks
    # 2. For each block, add 7 columns for Strassen
    # 3. For remaining (i,j,k) not in any block, add 1 column each for schoolbook
    
    # Define which (i,j,k) are in Strassen blocks
    # i-blocks: {0,1} -> i0=0, i1=1
    # j-blocks: {0,1}, {2,3} -> (j0=0,j1=1), (j0=2,j1=3)
    # k-blocks: {0,1}, {2,3}, {4,5} -> (k0=0,k1=1), (k0=2,k1=3), (k0=4,k1=5)
    
    strassen_blocks = []
    for i0, i1 in [(0, 1)]:
        for j0, j1 in [(0, 1), (2, 3)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                strassen_blocks.append((i0, i1, j0, j1, k0, k1))
    
    # Mark which (i,j,k) are covered by Strassen blocks
    strassen_set = set()
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    strassen_set.add((i, j, k))
    
    # Remaining (i,j,k) for schoolbook
    schoolbook_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_set:
                    schoolbook_set.add((i, j, k))
    
    # Build the decomposition
    # Each Strassen block contributes 7 columns
    # Each schoolbook term contributes 1 column
    rank = len(strassen_blocks) * 7 + len(schoolbook_set)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        # 7 Strassen factors
        # m1 = (a+d)(e+h)
        # U: a at (i0,j0), d at (i1,j1) -> U[i0*m+j0] += 1, U[i1*m+j1] += 1
        # V: e at (j0,k0), h at (j1,k1) -> V[j0*p+k0] += 1, V[j1*p+k1] += 1
        # W: 1 at all 4 (k,i) positions -> W[k0*n+i0] += 1, W[k0*n+i1] += 1, W[k1*n+i0] += 1, W[k1*n+i1] += 1
        U[i0 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        V[j0 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i1][col] += 1
        W[k1 * n + i0][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # m2 = (c+d)(e)
        # U: c at (i1,j0), d at (i1,j1) -> U[i1*m+j0] += 1, U[i1*m+j1] += 1
        # V: e at (j0,k0) -> V[j0*p+k0] += 1
        # W: 1 at (k0,i1) only? No, m2 contributes to C[1,0]
        # C[1,0] = m2 + m4, so m2 should have W[k0*n+i1] = 1
        U[i1 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i1][col] += 1
        col += 1
        
        # m3 = a(f-h)
        # U: a at (i0,j0) -> U[i0*m+j0] += 1
        # V: f at (j0,k1), h at (j1,k1) -> V[j0*p+k1] += 1, V[j1*p+k1] += -1
        # W: 1 at (k1,i0) -> W[k1*n+i0] += 1
        U[i0 * m + j0][col] += 1
        V[j0 * p + k1][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i0][col] += 1
        col += 1
        
        # m4 = d(g-e)
        # U: d at (i1,j1) -> U[i1*m+j1] += 1
        # V: g at (j1,k0), e at (j0,k0) -> V[j1*p+k0] += 1, V[j0*p+k0] += -1
        # W: 1 at (k0,i1) -> W[k0*n+i1] += 1
        U[i1 * m + j1][col] += 1
        V[j1 * p + k0][col] += 1
        V[j0 * p + k0][col] += -1
        W[k0 * n + i1][col] += 1
        col += 1
        
        # m5 = (a+b)(h)
        # U: a at (i0,j0), b at (i0,j1) -> U[i0*m+j0] += 1, U[i0*m+j1] += 1
        # V: h at (j1,k1) -> V[j1*p+k1] += 1
        # W: 1 at (k1,i0) -> W[k1*n+i0] += 1
        U[i0 * m + j0][col] += 1
        U[i0 * m + j1][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i0][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        # U: c at (i1,j0), a at (i0,j0) -> U[i1*m+j0] += 1, U[i0*m+j0] += -1
        # V: e at (j0,k0), f at (j0,k1) -> V[j0*p+k0] += 1, V[j0*p+k1] += 1
        # W: 1 at (k0,i1) and (k1,i1)? C[1,1] = m1 - m2 + m3 + m6
        # So m6 contributes to C[1,1], which is at position (i1, k1) in output
        # W[k1*n+i1] += 1
        U[i1 * m + j0][col] += 1
        U[i0 * m + j0][col] += -1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k1][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        # U: b at (i0,j1), d at (i1,j1) -> U[i0*m+j1] += 1, U[i1*m+j1] += -1
        # V: g at (j1,k0), h at (j1,k1) -> V[j1*p+k0] += 1, V[j1*p+k1] += 1
        # W: 1 at (k0,i0) -> C[0,0] = m1 + m4 - m5 + m7, so m7 at (i0, k0)
        W[k0 * n + i0][col] += 1
        U[i0 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        col += 1
    
    # Add schoolbook terms for remaining (i,j,k)
    for (i, j, k) in schoolbook_set:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
