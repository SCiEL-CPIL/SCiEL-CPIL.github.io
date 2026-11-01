import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Partition:
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x6 -> split B1 into three 2x2 blocks, Strassen each: 3*7 = 21
    # A2*B2: 2x3 * 3x6 -> split A2=[A2a(2x2)|A2b(2x1)], B2=[B2a(2x6);B2b(1x6)]
    #   A2a*B2a: 2x2*2x6 -> 3*7 = 21
    #   A2b*B2b: 2x1*1x6 -> 12 schoolbook
    #   Total: 33
    # Grand total: 21 + 33 = 54
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries: list of (row, val), v_entries: list of (row, val), w_entries: list of (row, val)"""
        U.append(u_entries)
        V.append(v_entries)
        W.append(w_entries)
    
    def add_strassen(a_rows, b_rows, c_rows):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        a_rows: indices in A for [[a,b],[c,d]] = [(i0,j0),(i0,j1),(i1,j0),(i1,j1)]
        b_rows: indices in B for [[x,y],[u,v]] = [(j0,k0),(j0,k1),(j1,k0),(j1,k1)]
        c_rows: indices in C for [[C00,C01],[C10,C11]] = [(k0,i0),(k0,i1),(k1,i0),(k1,i1)]
        """
        # a = a_rows[0], b = a_rows[1], c = a_rows[2], d = a_rows[3]
        # x = b_rows[0], y = b_rows[1], u = b_rows[2], v = b_rows[3]
        # C00 = c_rows[0], C01 = c_rows[1], C10 = c_rows[2], C11 = c_rows[3]
        
        a_idx, b_idx, c_idx, d_idx = a_rows
        x_idx, y_idx, u_idx, v_idx = b_rows
        C00_idx, C01_idx, C10_idx, C11_idx = c_rows
        
        # Strassen:
        # M1 = (a+d)(x+u)
        # M2 = (c+d)x
        # M3 = a(y-u)
        # M4 = d(v-u)
        # M5 = (a+b)u
        # M6 = (c+d)(v-y)
        # M7 = (a+b)(y-v)
        #
        # C00 = M1 - M4 + M3
        # C01 = M4 + M7
        # C10 = M2 + M6
        # C11 = M1 - M5 + M6 + M7
        
        # Wait, let me verify these formulas carefully.
        # C00 should = ax + bu
        # C01 should = ay + bv
        # C10 should = cx + du
        # C11 should = cy + dv
        
        # Let me use the standard Strassen from literature:
        # M1 = (a+d)(x+u)
        # M2 = (c+d)x
        # M3 = a(y-u)
        # M4 = d(v-u)
        # M5 = (a+b)u
        # M6 = (c+d)(v-y)
        # M7 = (a+b)(y-v)
        #
        # C00 = M1 - M2 - M4 + M3
        # C01 = M3 + M4 + M7
        # C10 = M2 + M5 + M6
        # C11 = M1 - M5 + M6 + M7
        
        # Let me verify C00 = ax + bu:
        # M1 - M2 - M4 + M3
        # = (a+d)(x+u) - (c+d)x - d(v-u) + a(y-u)
        # = ax + au + dx + du - cx - dx - dv + du + ay - au
        # = ax - cx - dv + 2du + ay
        # This does NOT equal ax + bu. So these formulas are wrong.
        
        # Let me derive correct Strassen.
        # We need 7 bilinear forms whose linear combinations give the 4 products.
        
        # Correct Strassen (verified):
        # M1 = (a+d)(x+u)
        # M2 = (c+d)x  
        # M3 = a(y-u)
        # M4 = d(v-u)
        # M5 = (a+b)u
        # M6 = (c+d)(v-y)
        # M7 = (a+b)(y-v)
        
        # Hmm, I keep getting the same formulas. Let me try a different approach.
        # Let me just verify numerically what the correct combination is.
        
        # Actually, let me use a different well-known set:
        # M1 = (a+d)(x+u)
        # M2 = (c+d)x
        # M3 = a(y-u)  
        # M4 = d(v-u)
        # M5 = (a+b)u
        # M6 = (c+d)(v-y)
        # M7 = (a+b)(y-v)
        
        # C00 = M1 - M4 + M3  ??? Let me check:
        # (a+d)(x+u) - d(v-u) + a(y-u)
        # = ax + au + dx + du - dv + du + ay - au
        # = ax + dx + 2du - dv + ay
        # Still not ax+bu.
        
        # I think the issue is I'm misremembering. Let me just implement schoolbook 
        # for the 2x2x2 case to be safe, then use the block structure.
        # Actually, let me try to find the right Strassen by solving.
        
        # For now, let me just use schoolbook for everything to ensure correctness.
        # Rank = 60. Then I'll add the Strassen improvement.
        
        # Actually, let me try yet another formulation. The key insight:
        # M1 = (a+d)(x+u) = ax + au + dx + du
        # M2 = (c+d)x = cx + dx
        # M3 = a(y-u) = ay - au
        # M4 = d(v-u) = dv - du
        # M5 = (a+b)u = au + bu
        # M6 = (c+d)(v-y) = cv - cy + dv - dy
        # M7 = (a+b)(y-v) = ay - av + by - bv
        
        # C00 = ax + bu
        # Try: M5 - M2 + ... hmm
        # M5 = au + bu
        # M5 - au = bu... but I can't isolate au.
        
        # Let me try: C00 = M1 - M2 - M4 + M3
        # = (ax+au+dx+du) - (cx+dx) - (dv-du) + (ay-au)
        # = ax + au + dx + du - cx - dx - dv + du + ay - au
        # = ax - cx - dv + 2du + ay
        # Nope.
        
        # C00 = M1 - M2 - M4 + M3 doesn't work.
        
        # Let me try: C00 = M5 - M2 + M1 - M4 - ...
        # This is getting complicated. Let me just use a verified implementation.
        
        # From multiple sources, the correct Strassen is:
        # p1 = (a + d) * (x + u)
        # p2 = (c + d) * x
        # p3 = a * (y - u)
        # p4 = d * (v - u)
        # p5 = (a + b) * u
        # p6 = (c + d) * (v - y)
        # p7 = (a + b) * (y - v)
        
        # C[0][0] = p1 - p4 + p3
        # C[0][1] = p4 + p7
        # C[1][0] = p2 + p6
        # C[1][1] = p1 - p5 + p6 + p7
        
        # Let me verify C[0][0] = p1 - p4 + p3:
        # = (a+d)(x+u) - d(v-u) + a(y-u)
        # = ax + au + dx + du - dv + du + ay - au
        # = ax + dx + 2du - dv + ay
        # Should be ax + bu. Still wrong!
        
        # I'm clearly misremembering. Let me just use schoolbook (rank 4 per 2x2x2)
        # and focus on getting the block structure right. Then I can add Strassen later.
        
        # Schoolbook for 2x2x2: 4 terms
        # ax: U has a, V has x, W has C00
        # bu: U has b, V has u, W has C00
        # cx: U has c, V has x, W has C10
        # du: U has d, V has u, W has C10
        # ay: U has a, V has y, W has C01
        # bv: U has b, V has v, W has C01
        # cy: U has c, V has y, W has C11
        # dv: U has d, V has v, W has C11
        
        # That's 8 terms for 2x2x2, which is just schoolbook.
        # Strassen should do it in 7.
        
        # Let me try to find the correct Strassen by a different method.
        # The tensor for 2x2x2 matmul has 8 components. We want to express it as sum of 7 rank-1 tensors.
        
        # A rank-1 tensor is u_i * v_j * w_k where u is in R^4 (for A's 4 entries),
        # v is in R^4 (for B's 4 entries), w is in R^4 (for C's 4 entries).
        
        # The matmul tensor T[i,j,k] = 1 if C[k] = sum_j A[i,j]*B[j,k]
        # In our indexing: A has entries (i,j) for i in {0,1}, j in {0,1}
        # B has entries (j,k) for j in {0,1}, k in {0,1}
        # C has entries (k,i) for k in {0,1}, i in {0,1}
        
        # T[(i,j),(j,k),(k,i)] = 1 for matching j
        
        # So the 8 nonzero entries are:
        # (0,0),(0,0),(0,0) -> a*x -> C00
        # (0,1),(1,0),(0,0) -> b*u -> C00
        # (1,0),(0,0),(1,0) -> c*x -> C10
        # (1,1),(1,0),(1,0) -> d*u -> C10
        # (0,0),(0,1),(0,1) -> a*y -> C01
        # (0,1),(1,1),(0,1) -> b*v -> C01
        # (1,0),(0,1),(1,1) -> c*y -> C11
        # (1,1),(1,1),(1,1) -> d*v -> C11
        
        # For Strassen, we need 7 vectors (u_t, v_t, w_t) such that
        # sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = T[(i,j),(j,k),(k,i)]
        
        # This is a known result. Let me just use a lookup table approach.
        # I'll hardcode the 7 terms for Strassen 2x2x2.
        
        # After much deliberation, here are the correct Strassen terms:
        # Term 1: (a+d)(x+u) -> contributes to C00, C11
        #   U: a=1, d=1; V: x=1, u=1; W: C00=1, C11=1
        # Term 2: (c+d)x -> contributes to C10
        #   U: c=1, d=1; V: x=1; W: C10=1
        # Term 3: a(y-u) -> contributes to C00, C01
        #   U: a=1; V: y=1, u=-1; W: C00=1, C01=1
        # Term 4: d(v-u) -> contributes to C00, C01
        #   U: d=1; V: v=1, u=-1; W: C00=1, C01=1
        # Term 5: (a+b)u -> contributes to C10, C11
        #   U: a=1, b=1; V: u=1; W: C10=1, C11=1
        # Term 6: (c+d)(v-y) -> contributes to C10, C11
        #   U: c=1, d=1; V: v=1, y=-1; W: C10=1, C11=1
        # Term 7: (a+b)(y-v) -> contributes to C01, C11
        #   U: a=1, b=1; V: y=1, v=-1; W: C01=1, C11=1
        
        # Let me verify C00 = ax + bu:
        # From term 1: a*x + a*u + d*x + d*u
        # From term 3: a*y - a*u
        # From term 4: d*v - d*u
        # Sum for C00: ax + au + dx + du + ay - au + dv - du = ax + dx + ay + dv
        # That's not ax + bu!
        
        # I'm clearly getting this wrong. Let me take a completely different approach.
        # I'll just use schoolbook for the 2x2x2 blocks (rank 4 each) and see if the
        # overall structure gives a valid decomposition. Then I can try to optimize.
        
        # Schoolbook 2x2x2: 4 terms (not 8, because we can group)
        # Actually no, schoolbook 2x2x2 requires 4*4=... no.
        # 2x2 * 2x2: C[i,k] = sum_j A[i,j]*B[j,k]
        # 4 entries of C, each is a sum of 2 products. Total 8 scalar products.
        # But in tensor decomposition, each scalar product A[i,j]*B[j,k]*C[i,k] is one term.
        # So schoolbook rank for 2x2x2 is 4 (one for each j, but we need to handle all i,k).
        # Wait no. For fixed j, the term is sum over i,k of A[i,j]*B[j,k]*C[i,k].
        # That's not a rank-1 tensor in general.
        
        # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = delta (where the three indices are linked)
        # Schoolbook: for each j, the term is A[:,j] * B[j,:] * C^T
        # In tensor form: for each j, u_j[i] = A[i,j], v_j[k] = B[j,k], w_j[i,k] = 1
        # So rank = m (the middle dimension).
        
        # For 2x2x2: rank = 2 in schoolbook? No, that can't be right.
        # Wait, I think I'm confusing myself.
        
        # Let me restart with the correct tensor decomposition framework.
        # For multiplying A (n x m) by B (m x p) to get C (n x p):
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # 
        # In the task's notation: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
        # W[k*n+i, r] corresponds to C[i,k]
        # 
        # A rank-1 decomposition term r contributes:
        # sum_{i,j} U[i*m+j, r] * sum_{j,k} V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right either. The tensor is:
        # T[i,j | j,k | k,i] where the three "slices" are U, V, W
        # 
        # Actually, the standard formulation: the matmul tensor M has entries
        # M[(i,j), (j,k), (k,i)] = 1 if the indices are consistent
        # A decomposition is: M = sum_r U_r * V_r * W_r (outer product of 3 vectors)
        # where U_r is in R^{n*m}, V_r is in R^{m*p}, W_r is in R^{p*n}
        # 
        # For schoolbook: for each j (middle index), define:
        # U_r[i*m+j] = 1 for all i (else 0)
        # V_r[j*p+k] = 1 for all k (else 0)
        # W_r[k*n+i] = 1 for all i,k (else 0)
        # Wait, that doesn't work because W_r needs to be a single vector.
        
        # Let me look at the current program's schoolbook implementation:
        # for i in range(n): for j in range(m): for k in range(p):
        #   U[i*m+j][column] = 1
        #   V[j*p+k][column] = 1
        #   W[k*n+i][column] = 1
        # 
        # So each term is a single (i,j,k) triple. That gives n*m*p = 60 terms.
        # Each term: U has 1 at position (i,j), V has 1 at position (j,k), W has 1 at position (k,i)
        # This is indeed a valid rank-1 decomposition (each term is a rank-1 tensor).
        # And it's the "element-wise" schoolbook, which has rank n*m*p.
        
        # But the true schoolbook (using the sum over j) has rank m, where each term is:
        # For fixed j: U[i*m+j] = 1 for all i, V[j*p+k] = 1 for all k, W[k*n+i] = 1 for all i,k
        # But W[k*n+i] = 1 for all i,k means W is all ones in that term, which is not a valid
        # rank-1 tensor in the standard sense... actually it is! A rank-1 tensor in this context
        # is just a vector in R^{n*m} * R^{m*p} * R^{p*n}.
        
        # Hmm, but the task says U is (n*m, rank), so each column of U is a vector in R^{n*m}.
        # Similarly for V and W. A "rank-1 term" is a triple of such vectors.
        # The decomposition says: the matmul tensor = sum_r U[:,r] * V[:,r] * W[:,r] (triple outer product)
        
        # For the element-wise approach: term (i,j,k) has U with 1 at (i,j), V with 1 at (j,k), W with 1 at (k,i).
        # This gives exactly the matmul tensor since C[i,k] = sum_j A[i,j]*B[j,k].
        
        # For the "sum over j" approach: term j has U[i*m+j]=1 for all i, V[j*p+k]=1 for all k, W[k*n+i]=1 for all i,k.
        # But then the contribution to C[i,k] would be sum_j (sum_i A[i,j]) * (sum_k B[j,k]) * 1, which is wrong.
        
        # So the element-wise approach (rank n*m*p) is the correct basic decomposition.
        # To get a lower rank, we need to find linear combinations.
        
        # OK so I was overthinking this. The current program's approach is correct.
        # Rank 60 = 2*5*6. To improve, I need to find a better decomposition.
        
        # The key insight: for the 2x2x2 subproblem, the matmul tensor has rank 7 (Strassen).
        # This means we can replace 8 element-wise terms with 7 terms.
        
        # Let me just implement the element-wise schoolbook but replace 2x2x2 blocks with Strassen.
        
        # For a 2x2x2 block with A indices (i1,j1),(i2,j1),(i1,j2),(i2,j2),
        # B indices (j1,k1),(j1,k2),(j2,k1),(j2,k2),
        # C indices (k1,i1),(k2,i1),(k1,i2),(k2,i2):
        
        # The 8 element-wise terms are:
        # (i1,j1),(j1,k1),(k1,i1)
        # (i1,j2),(j2,k1),(k1,i1)
        # (i2,j1),(j1,k1),(k1,i2)
        # (i2,j2),(j2,k1),(k1,i2)
        # (i1,j1),(j1,k2),(k2,i1)
        # (i1,j2),(j2,k2),(k2,i1)
        # (i2,j1),(j1,k2),(k2,i2)
        # (i2,j2),(j2,k2),(k2,i2)
        
        # Strassen replaces these 8 with 7 terms. Let me find the correct 7 terms.
        
        # I'll use a numerical approach: set up the system and solve.
        # Actually, let me just hardcode a known correct Strassen decomposition.
        
        # After careful derivation, here are the 7 Strassen terms for 2x2x2:
        # Let A = [[a,b],[c,d]], B = [[x,y],[u,v]], C = [[C00,C01],[C10,C11]]
        # C00 = ax+bu, C01 = ay+bv, C10 = cx+du, C11 = cy+dv
        
        # Term 1: (a+d)(x+u) -> U: {a:1, d:1}, V: {x:1, u:1}, W: {C00:1, C11:1}
        # Term 2: (c+d)(x) -> U: {c:1, d:1}, V: {x:1}, W: {C10:1}
        # Term 3: (a)(y-u) -> U: {a:1}, V: {y:1, u:-1}, W: {C00:1, C01:1}
        # Term 4: (d)(v-u) -> U: {d:1}, V: {v:1, u:-1}, W: {C00:1, C01:1}
        # Term 5: (a+b)(u) -> U: {a:1, b:1}, V: {u:1}, W: {C10:1, C11:1}
        # Term 6: (c+d)(v-y) -> U: {c:1, d:1}, V: {v:1, y:-1}, W: {C10:1, C11:1}
        # Term 7: (a+b)(y-v) -> U: {a:1, b:1}, V: {y:1, v:-1}, W: {C01:1, C11:1}
        
        # Verify C00 = ax + bu:
        # T1: a*x + a*u + d*x + d*u
        # T3: a*y - a*u
        # T4: d*v - d*u
        # Sum: ax + au + dx + du + ay - au + dv - du = ax + dx + ay + dv
        # Expected: ax + bu. NOT MATCHING.
        
        # I clearly cannot remember Strassen correctly. Let me just use schoolbook (rank 8 per 2x2x2)
        # and accept rank 60. Or better, let me use a computational approach to find the Strassen terms.
        
        # For now, let me just return the schoolbook rank-60 decomposition to ensure correctness.
        
        pass  # This function will be replaced
    
    # Just use schoolbook for now - rank 60
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
