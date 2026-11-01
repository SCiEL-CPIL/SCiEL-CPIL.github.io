"""Matrix multiplication 2,4,5 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 145.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook rank: n*m*p = 40
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: Use block structure. M_{2,4,5} can be seen as blocks.
    # A(2x4) * B(4x5) = C(2x5)
    # Split B into B1(4x2), B2(4x2), B3(4x1)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bk is 2x2 or 2x1
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,2,1} has rank 4 (schoolbook, minimal)
    # But we can potentially share factors
    
    # Alternative: view as M_{2,4,5}
    # Known: r(M_{2,4,5}) <= 26 (from various bounds)
    # Let's try to construct a decomposition using Strassen on 2x2 blocks
    
    # Split the 4 intermediate dimension into two 2s:
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    # Each term is M_{2,2,5}
    # r(M_{2,2,5}) <= ? 
    # M_{2,2,5}: split B into B_a(2x2), B_b(2x2), B_c(2x1)
    # = A*B_a + A*B_b + A*B_c
    # r(M_{2,2,2})=7, r(M_{2,2,1})=4
    # Naive: 7+7+4=18 for M_{2,2,5}
    # Total: 18+18=36 for M_{2,4,5}
    
    # Can we do better with more global structure?
    # Try rank 26, 24, 22...
    
    # For now, implement the 36-rank construction and try to improve
    
    def strassen_222():
        """Return Strassen decomposition for 2x2*2x2. Rank 7."""
        # Standard Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # U rows: A11,A12,A21,A22 (indices 0,1,2,3)
        # V rows: B11,B12,B21,B22 (indices 0,1,2,3)
        # W rows: C11,C12,C21,C22 (indices 0,1,2,3)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W
    
    def schoolbook_221():
        """M_{2,2,1}: rank 4"""
        U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        V = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        # W: C = A11*B1 + A12*B2 + A21*B3 + A22*B4
        # Output is 2x1, so W has 2 rows
        W = [[1,0,0,0],[0,1,0,0]]
        return U, V, W
    
    def combine_m225():
        """Combine M_{2,2,2} twice and M_{2,2,1} to get M_{2,2,5}. Rank 7+7+4=18"""
        U1,V1,W1 = strassen_222()
        U2,V2,W2 = strassen_222()
        U3,V3,W3 = schoolbook_221()
        
        # Concatenate: 18 columns total
        U = []
        for r in range(4):
            U.append(U1[r] + U2[r] + U3[r])
        V = []
        for r in range(4):
            V.append(V1[r] + V2[r] + V3[r])
        # W for M_{2,2,5}: output is 2x5, so W has 10 rows
        # W1 contributes to cols 0-1 (4 rows: C11,C12 for 2x2 -> actually 4 entries)
        # Wait, M_{2,2,2} output is 2x2 = 4 entries. M_{2,2,1} output is 2x1 = 2 entries.
        # Total 10 output entries.
        W = []
        for c in range(10):
            row = []
            for r in range(18):
                row.append(0)
            W.append(row)
        # W1: 4 rows (C11,C12,C21,C22) -> output cols 0-3 (for 2x2 block)
        # Actually output is 2x5. First 2x2 block is cols 0-1, second 2x2 is cols 2-3, last col is col 4.
        # C11->(0,0), C12->(0,1), C21->(1,0), C22->(1,1)
        # For W1: W1[0] goes to output (0,0)=col 0, W1[1] to (0,1)=col 1, W1[2] to (1,0)=col 0+2? No.
        # Output indices: (0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4)
        # Flat: k*n+i where k=col, n=2 rows. So (i,k) -> k*2+i
        # (0,0)->0, (0,1)->2, (0,2)->4, (0,3)->6, (0,4)->8
        # (1,0)->1, (1,1)->3, (1,2)->5, (1,3)->7, (1,4)->9
        
        # W1 (M_{2,2,2}): rows are C11,C12,C21,C22
        # C11 at (0,0)->idx 0, C12 at (0,1)->idx 2, C21 at (1,0)->idx 1, C22 at (1,1)->idx 3
        w1_map = [0, 2, 1, 3]  # W1 row j -> output flat index
        for j in range(4):
            for r in range(7):
                W[w1_map[j]][r] = W1[j][r]
        
        # W2 (second M_{2,2,2}): cols 2-3
        # C11 at (0,2)->idx 4, C12 at (0,3)->idx 6, C21 at (1,2)->idx 5, C22 at (1,3)->idx 7
        w2_map = [4, 6, 5, 7]
        for j in range(4):
            for r in range(7):
                W[w2_map[j]][7+r] = W2[j][r]
        
        # W3 (M_{2,2,1}): col 4
        # Row 0 at (0,4)->idx 8, Row 1 at (1,4)->idx 9
        for j in range(2):
            for r in range(4):
                W[8+j][14+r] = W3[j][r]
        
        return U, V, W, 18
    
    # Build M_{2,4,5} from two M_{2,2,5}
    U_a, V_a, W_a, rank_a = combine_m225()
    U_b, V_b, W_b, rank_b = combine_m225()
    
    # A is 2x4: [A1, A2] where A1 is cols 0-1, A2 is cols 2-3
    # B is 4x5: [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    # C = A1*B1 + A2*B2
    # U_a acts on A1 (4 entries), V_a acts on B1 (10 entries)
    # U_b acts on A2 (4 entries), V_b acts on B2 (10 entries)
    
    total_rank = rank_a + rank_b
    U_full = [[0]*total_rank for _ in range(8)]  # 2x4 = 8 entries
    V_full = [[0]*total_rank for _ in range(20)]  # 4x5 = 20 entries
    W_full = [[0]*total_rank for _ in range(10)]  # 5x2 = 10 entries
    
    # U: A1 rows 0-3 (A11,A12,A21,A22), A2 rows 4-7
    for r in range(4):
        for c in range(rank_a):
            U_full[r][c] = U_a[r][c]
            U_full[r+4][c] = 0  # A1 doesn't affect A2
        for c in range(rank_b):
            U_full[r][rank_a+c] = 0
            U_full[r+4][rank_a+c] = U_b[r][c]
    
    # V: B1 rows 0-9, B2 rows 10-19
    # B1 is 2x5=10 entries, indexed by k*2+j for k=0,1 and j=0..4
    # Wait, B is 4x5. B1 is top 2 rows, B2 is bottom 2 rows.
    # B1 entries: B[0,0..4], B[1,0..4] -> 10 entries
    # B2 entries: B[2,0..4], B[3,0..4] -> 10 entries
    # In our M_{2,2,5}, the 2x2 B block has entries B11,B12,B21,B22 for a 2x2.
    # But B1 is 2x5, not 2x2. The M_{2,2,5} V matrix has 4 rows for the 2x2 part.
    # Hmm, I need to be more careful.
    
    # Let me redo: M_{2,2,5} means A is 2x2, B is 2x5, C is 2x5.
    # So V for M_{2,2,5} should have 10 rows (2x5), not 4.
    # My combine_m225 had V with 4 rows - that's wrong.
    
    # Let me rebuild properly.
    
    def strassen_222_v2():
        """M_{2,2,2}: U(4x7), V(4x7), W(4x7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[1][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def schoolbook_2x2x1():
        """M_{2,2,1}: A(2x2)*B(2x1)=C(2x1). Rank 4. U(4x4),V(2x4),W(2x4)"""
        U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        V = [[1,0,0,0],[0,1,0,0]]
        W = [[1,0,0,0],[0,1,0,0]]
        return U, V, W
    
    def m225_decomp():
        """M_{2,2,5}: A(2x2)*B(2x5)=C(2x5). 
        Split B into B_a(2x2), B_b(2x2), B_c(2x1).
        C = A*B_a + A*B_b + A*B_c
        Rank: 7+7+4=18
        U(4x18), V(10x18), W(10x18)
        """
        U1,V1,W1 = strassen_222_v2()
        U2,V2,W2 = strassen_222_v2()
        U3,V3,W3 = schoolbook_2x2x1()
        
        rank = 18
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(10)]  # B is 2x5 = 10 entries
        W = [[0]*rank for _ in range(10)]  # C is 2x5 = 10 entries
        
        # U: A is 2x2, same for all three parts
        for r in range(4):
            for c in range(7):
                U[r][c] = U1[r][c]
                U[r][7+c] = U2[r][c]
            for c in range(4):
                U[r][14+c] = U3[r][c]
        
        # V: B_a is rows 0-1 of B (first 2 cols), B_b is rows 2-3, B_c is row 4
        # B is 2x5: entries B[0,0..4], B[1,0..4]
        # B_a(2x2): B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0,1,2,3
        # B_b(2x2): B[0,2],B[0,3],B[1,2],B[1,3] -> V rows 4,5,6,7
        # B_c(2x1): B[0,4],B[1,4] -> V rows 8,9
        v_a_map = [0,1,2,3]
        v_b_map = [4,5,6,7]
        v_c_map = [8,9]
        
        for r in range(4):
            for c in range(7):
                V[v_a_map[r]][c] = V1[r][c]
                V[v_b_map[r]][7+c] = V2[r][c]
        for r in range(2):
            for c in range(4):
                V[v_c_map[r]][14+c] = V3[r][c]
        
        # W: C_a(2x2), C_b(2x2), C_c(2x1)
        # C is 2x5: C[0,0..4], C[1,0..4]
        # W flat index: k*2+i for (i,k)
        # C_a: (0,0)->0,(0,1)->2,(1,0)->1,(1,1)->3
        # C_b: (0,2)->4,(0,3)->6,(1,2)->5,(1,3)->7
        # C_c: (0,4)->8,(1,4)->9
        w_a_map = [0,2,1,3]
        w_b_map = [4,6,5,7]
        w_c_map = [8,9]
        
        for r in range(4):
            for c in range(7):
                W[w_a_map[r]][c] = W1[r][c]
                W[w_b_map[r]][7+c] = W2[r][c]
        for r in range(2):
            for c in range(4):
                W[w_c_map[r]][14+c] = W3[r][c]
        
        return U, V, W, rank
    
    # Now build M_{2,4,5}
    # A is 2x4: A[0,0..3], A[1,0..3]
    # Split into A1(2x2) cols 0-1, A2(2x2) cols 2-3
    # B is 4x5: B[0..3,0..4]
    # Split into B1(2x5) rows 0-1, B2(2x5) rows 2-3
    # C = A1*B1 + A2*B2
    # Each is M_{2,2,5}
    
    Ua, Va, Wa, ra = m225_decomp()
    Ub, Vb, Wb, rb = m225_decomp()
    
    total_rank = ra + rb
    U_full = [[0]*total_rank for _ in range(8)]
    V_full = [[0]*total_rank for _ in range(20)]
    W_full = [[0]*total_rank for _ in range(10)]
    
    # U: A1 rows 0-3 (A[0,0],A[0,1],A[1,0],A[1,1]), A2 rows 4-7
    for r in range(4):
        for c in range(ra):
            U_full[r][c] = Ua[r][c]
        for c in range(rb):
            U_full[r+4][ra+c] = Ub[r][c]
    
    # V: B1 rows 0-9 (B[0,0..4],B[1,0..4]), B2 rows 10-19
    for r in range(10):
        for c in range(ra):
            V_full[r][c] = Va[r][c]
        for c in range(rb):
            V_full[r+10][ra+c] = Vb[r][c]
    
    # W: same for both (C is 2x5, shared)
    for r in range(10):
        for c in range(ra):
            W_full[r][c] = Wa[r][c]
        for c in range(rb):
            W_full[r][ra+c] = Wb[r][c]
    
    result_36 = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    def verify(res):
        U = np.array(res["U"], dtype=float)
        V = np.array(res["V"], dtype=float)
        W = np.array(res["W"], dtype=float)
        n, m, p = 2, 4, 5
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val) > 1e-9:
                        return False
        return True
    
    if verify(result_36):
        best_result = result_36
        best_rank = total_rank
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 30:
        return best_result
    
    # Try to find lower rank via search (limited)
    # For now, return the 36-rank result
    return best_result
# EVOLVE-BLOCK-END
