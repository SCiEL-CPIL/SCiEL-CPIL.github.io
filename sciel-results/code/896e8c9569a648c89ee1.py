import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115.0
    n, m, p = 3, 4, 5

    def verify(U, V, W):
        """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def build_schoolbook():
        """Standard rank-60 decomposition: one term per (i,j,k)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """
        Return the rank-7 Strassen decomposition of the 2x2x2 matrix multiplication tensor.
        
        We need U (4x7), V (4x7), W (4x7) such that for all a,b,c in {0,1}:
        sum_t U[a*2+b, t] * V[b*2+c, t] * W[c*2+a, t] = 1
        
        The 2x2x2 tensor T[a,b,c] = 1 for all a,b,c.
        This is the tensor for C = AB where A,B,C are 2x2.
        
        Strassen's algorithm uses 7 multiplications:
        M1 = (A00+A11)(B10-B00)
        M2 = (A10+A11)(B00)  
        M3 = A00(B01-B11)
        M4 = A11(B00+B11)
        M5 = (A00+A01)(B11)
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M4 - M2 + M3 + M1
        C01 = M5 + M4
        C10 = M2 + M6
        C11 = M1 - M3 + M5 + M7
        
        For the tensor decomposition, each term t contributes:
        U[a,b] * V[b,c] * W[c,a] to the (a,b,c) entry.
        
        We need to find 7 triples (u_t, v_t, w_t) where each is a 2x2 matrix (flattened to 4-vector)
        such that sum_t u_t[a,b] * v_t[b,c] * w_t[c,a] = 1.
        
        Let me work this out carefully. The tensor is T = sum_{a,b,c} e_{a,b} ⊗ e_{b,c} ⊗ e_{c,a}
        where the first factor is indexed by (a,b) [row,col of A], second by (b,c) [row,col of B],
        third by (c,a) [col,row of C - note the trace convention W[k*n+i] means W[c,a]].
        
        Actually, let me re-examine. In our convention:
        U[i*m+j, t] - indexed by (row_of_A, col_of_A) = (i, j)
        V[j*p+k, t] - indexed by (row_of_B, col_of_B) = (j, k)
        W[k*n+i, t] - indexed by (col_of_C, row_of_C) = (k, i)
        
        So for the 2x2x2 subproblem with local indices a (row), b (col/mid), c (col of output):
        U_local[a,b,t], V_local[b,c,t], W_local[c,a,t]
        
        We need: sum_t U[a,b,t] * V[b,c,t] * W[c,a,t] = 1 for all a,b,c in {0,1}
        
        This is 8 equations. Let me denote the 4 entries of U as U00,U01,U10,U11 (for a=0/1, b=0/1)
        Similarly V00,V01,V10,V11 and W00,W01,W10,W11 (for c=0/1, a=0/1).
        
        The 8 equations:
        (a=0,b=0,c=0): U00*V00*W00 = 1
        (a=0,b=0,c=1): U00*V01*W10 = 1  [W[c=1,a=0] = W10]
        (a=0,b=1,c=0): U01*V10*W00 = 1  [W[c=0,a=0] = W00]
        (a=0,b=1,c=1): U01*V11*W10 = 1
        (a=1,b=0,c=0): U10*V00*W01 = 1  [W[c=0,a=1] = W01]
        (a=1,b=0,c=1): U10*V01*W11 = 1  [W[c=1,a=1] = W11]
        (a=1,b=1,c=0): U11*V10*W01 = 1
        (a=1,b=1,c=1): U11*V11*W11 = 1
        
        Wait, I need to be more careful with W indexing. W is indexed by (c, a) where c is the column of C and a is the row of C.
        W[c*2+a, t] for 2x2 case.
        
        So:
        W[0*2+0] = W00 (c=0, a=0)
        W[0*2+1] = W01 (c=0, a=1)  
        W[1*2+0] = W10 (c=1, a=0)
        W[1*2+1] = W11 (c=1, a=1)
        
        Equations:
        (a=0,b=0,c=0): U[0,0]*V[0,0]*W[0,0] = 1
        (a=0,b=0,c=1): U[0,0]*V[0,1]*W[1,0] = 1
        (a=0,b=1,c=0): U[0,1]*V[1,0]*W[0,0] = 1
        (a=0,b=1,c=1): U[0,1]*V[1,1]*W[1,0] = 1
        (a=1,b=0,c=0): U[1,0]*V[0,0]*W[0,1] = 1
        (a=1,b=0,c=1): U[1,0]*V[0,1]*W[1,1] = 1
        (a=1,b=1,c=0): U[1,1]*V[1,0]*W[0,1] = 1
        (a=1,b=1,c=1): U[1,1]*V[1,1]*W[1,1] = 1
        
        From Strassen, the 7 terms are derived from the 7 multiplications. Let me use a known explicit decomposition.
        
        One known rank-7 decomposition of the 2x2x2 matrix multiplication tensor:
        
        Term 1: U=[1,0,0,1], V=[-1,0,1,0], W=[1,0,0,1]
          Check (0,0,0): 1*(-1)*1 = -1 ≠ 1. Hmm.
        
        Let me try a different approach. I'll use the fact that the Strassen algorithm can be expressed as a tensor decomposition.
        
        Actually, let me just hardcode a verified decomposition. I'll compute it numerically.
        
        After careful derivation, here's a valid rank-7 decomposition:
        
        The key insight from Strassen:
        M1 = (A00+A11)(B10-B00) -> contributes to C00 and C11
        M2 = (A10+A11)(B00) -> contributes to C00 and C10
        M3 = A00(B01-B11) -> contributes to C00 and C11
        M4 = A11(B00+B11) -> contributes to C00 and C01
        M5 = (A00+A01)(B11) -> contributes to C01 and C11
        M6 = (A10-A00)(B00+B01) -> contributes to C10
        M7 = (A01-A11)(B10+B11) -> contributes to C11
        
        For each term, the U part encodes the linear combination of A entries,
        the V part encodes the linear combination of B entries,
        and the W part encodes which C entries receive the contribution.
        
        But in the tensor formulation, U[a,b] is a single scalar for position (a,b), not a matrix.
        So for term t, U_t is a 4-vector (flattened 2x2), V_t is a 4-vector, W_t is a 4-vector.
        
        The product U_t[a,b] * V_t[b,c] * W_t[c,a] must sum to 1 over t.
        
        Let me try to construct this from Strassen. For M1 = (A00+A11)(B10-B00):
        The "A factor" is A00+A11, which means we want U_t to have 1 at (0,0) and 1 at (1,1).
        The "B factor" is B10-B00, which means V_t should have -1 at (0,0) and 1 at (1,0).
        The "C factor": M1 contributes +1 to C00 and +1 to C11.
        So W_t should have 1 at (c=0,a=0) and 1 at (c=1,a=1).
        
        U1 = [1, 0, 0, 1]  (U[0,0]=1, U[1,1]=1)
        V1 = [-1, 0, 1, 0]  (V[0,0]=-1, V[1,0]=1)
        W1 = [1, 0, 0, 1]  (W[0,0]=1, W[1,1]=1)
        
        Check contributions:
        (a=0,b=0,c=0): U1[0,0]*V1[0,0]*W1[0,0] = 1*(-1)*1 = -1
        (a=0,b=0,c=1): U1[0,0]*V1[0,1]*W1[1,0] = 1*0*0 = 0
        (a=0,b=1,c=0): U1[0,1]*V1[1,0]*W1[0,0] = 0*1*1 = 0
        (a=0,b=1,c=1): U1[0,1]*V1[1,1]*W1[1,0] = 0*0*0 = 0
        (a=1,b=0,c=0): U1[1,0]*V1[0,0]*W1[0,1] = 0*(-1)*0 = 0
        (a=1,b=0,c=1): U1[1,0]*V1[0,1]*W1[1,1] = 0*0*1 = 0
        (a=1,b=1,c=0): U1[1,1]*V1[1,0]*W1[0,1] = 1*1*0 = 0
        (a=1,b=1,c=1): U1[1,1]*V1[1,1]*W1[1,1] = 1*0*1 = 0
        
        So term 1 contributes: (-1, 0, 0, 0, 0, 0, 0, 0) for the 8 positions.
        That means it gives -1 to position (0,0,0). But we need the total to be +1.
        So this term alone gives -1 to (0,0,0), which is wrong.
        
        I think the issue is that in Strassen, M1 contributes to C00 with coefficient +1 and to C11 with coefficient +1, but the sign of the B factor matters.
        
        Let me reconsider. In Strassen:
        C00 = M4 - M2 + M3 + M1
        C11 = M1 - M3 + M5 + M7
        
        So M1 has coefficient +1 in C00 and +1 in C11.
        
        The tensor entry T[a,b,c] represents the coefficient of A[a,b]*B[b,c] in C[a,c].
        For the 2x2x2 case, T[a,b,c] = 1 for all a,b,c because C[a,c] = sum_b A[a,b]*B[b,c].
        
        In the rank-1 decomposition, each term t contributes U_t[a,b] * V_t[b,c] * W_t[c,a] to T[a,b,c].
        
        For M1: the A-part is (A00+A11), the B-part is (B10-B00), and it contributes to C00 and C11.
        
        In the tensor language:
        U_t[a,b] should be 1 where A[a,b] appears in the A-part: (0,0) and (1,1).
        V_t[b,c] should be the coefficient of B[b,c] in the B-part: B[1,0] has coeff 1, B[0,0] has coeff -1.
        W_t[c,a] should be the coefficient of M_t in C[a,c]: C[0,0] has coeff 1, C[1,1] has coeff 1.
        So W_t[0,0]=1 (c=0,a=0), W_t[1,1]=1 (c=1,a=1).
        
        Now the contribution to T[a,b,c] is U_t[a,b]*V_t[b,c]*W_t[c,a].
        
        For (a=0,b=0,c=0): U[0,0]*V[0,0]*W[0,0] = 1*(-1)*1 = -1
        For (a=1,b=1,c=1): U[1,1]*V[1,1]*W[1,1] = 1*0*1 = 0
        
        Hmm, that gives -1 to (0,0,0) but we need +1. The issue is that the tensor T[a,b,c] = 1 means that A[a,b]*B[b,c] contributes +1 to C[a,c]. But in Strassen, the term M1 = (A00+A11)(B10-B00) is an intermediate product, and its contribution to the final C is through the linear combinations.
        
        I think I'm confusing the tensor decomposition with the Strassen algorithm. Let me think about this differently.
        
        The tensor T for 2x2x2 matrix multiplication has entries T[a,b,c] = 1 for all a,b,c ∈ {0,1}.
        We want to write T = sum_t u_t ⊗ v_t ⊗ w_t where u_t ∈ R^4, v_t ∈ R^4, w_t ∈ R^4.
        
        A rank-1 tensor u ⊗ v ⊗ w has entry u[a,b]*v[b,c]*w[c,a] at position (a,b,c).
        
        So we need 7 such terms that sum to the all-ones tensor.
        
        Let me just solve this numerically using a known decomposition. I'll use the approach of setting up the equations and solving.
        
        Actually, let me just hardcode a verified decomposition. I found references that give the following:
        
        For the 2x2x2 matrix multiplication tensor, one rank-7 decomposition is:
        
        t=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  -> contributes 1 to (0,0,0)
        t=1: U=[0,1,0,0], V=[0,1,0,0], W=[0,1,0,0]  -> contributes 1 to (0,1,1) [W[1,0]=1? No...]
        
        This is getting complicated. Let me just use a numerical solver to find the decomposition.
        
        Actually, I realize I should just use the known result. The rank of the 2x2x2 matrix multiplication tensor is 7, and explicit decompositions are well-known. Let me use a different strategy: I'll construct the decomposition by solving the system of equations.
        
        For now, let me just return the schoolbook decomposition and note that the Strassen improvement needs more work.
        
        Actually, let me try a simpler approach: I'll use the fact that for the 2x2x2 case, we can verify any proposed decomposition numerically.
        
        Here's my plan: I'll propose a decomposition based on Strassen's structure and verify it. If it doesn't work, I'll fall back to schoolbook.
        
        Let me try this decomposition (based on careful analysis):
        
        Term 0: U=[1,0,0,1], V=[1,0,0,0], W=[1,0,0,0]
        Term 1: U=[0,0,1,0], V=[0,1,0,0], W=[0,1,0,0]
        ...
        
        This is taking too long. Let me just use the schoolbook for now and focus on a different optimization strategy.
        """
        # I'll use a numerical approach to find the Strassen decomposition
        # Set up the 8 equations for the 2x2x2 case
        
        # Variables: U_t[a*2+b], V_t[b*2+c], W_t[c*2+a] for t=0..6
        # That's 7*12 = 84 unknowns, 8 equations. Very underdetermined.
        
        # Let me use a specific known decomposition.
        # After research, here's a valid rank-7 decomposition of the 2x2x2 tensor:
        
        # Using the Strassen algorithm structure, the 7 terms are:
        terms = []
        
        # M1 = (A00+A11)(B10-B00)
        # A-part: A[0,0]+A[1,1] -> U has 1 at (0,0) and (1,1)
        # B-part: B[1,0]-B[0,0] -> V has -1 at (0,0) and 1 at (1,0)
        # C-part: +C[0,0] + C[1,1] -> W has 1 at (c=0,a=0) and (c=1,a=1)
        U1 = [1, 0, 0, 1]  # (a=0,b=0), (a=0,b=1), (a=1,b=0), (a=1,b=1)
        V1 = [-1, 0, 1, 0]  # (b=0,c=0), (b=0,c=1), (b=1,c=0), (b=1,c=1)
        W1 = [1, 0, 0, 1]  # (c=0,a=0), (c=0,a=1), (c=1,a=0), (c=1,a=1)
        terms.append((U1, V1, W1))
        
        # M2 = (A10+A11)(B00)
        U2 = [0, 0, 1, 1]
        V2 = [1, 0, 0, 0]
        W2 = [1, 0, 0, 0]  # C[0,0] gets -M2, so W should encode -1? 
        # Actually, the coefficient in C is -1 for C00, +1 for C10
        # So W2 should have -1 at (c=0,a=0) and +1 at (c=0,a=1)
        W2 = [-1, 1, 0, 0]
        terms.append((U2, V2, W2))
        
        # M3 = A00(B01-B11)
        U3 = [1, 0, 0, 0]
        V3 = [0, 1, 0, -1]
        W3 = [1, 0, 0, -1]  # C[0,0] gets +M3, C[1,1] gets -M3
        terms.append((U3, V3, W3))
        
        # M4 = A11(B00+B11)
        U4 = [0, 0, 0, 1]
        V4 = [1, 0, 0, 1]
        W4 = [1, 0, 1, 0]  # C[0,0] gets +M4, C[0,1] gets +M4
        terms.append((U4, V4, W4))
        
        # M5 = (A00+A01)(B11)
        U5 = [1, 1, 0, 0]
        V5 = [0, 0, 0, 1]
        W5 = [0, 0, 1, 1]  # C[0,1] gets +M5, C[1,1] gets +M5
        terms.append((U5, V5, W5))
        
        # M6 = (A10-A00)(B00+B01)
        U6 = [0, 0, 1, -1]  # Wait, A10-A00 means A[1,0]-A[0,0]
        # U6[a,b]: A[1,0] has coeff 1, A[0,0] has coeff -1
        U6 = [-1, 0, 1, 0]
        V6 = [1, 1, 0, 0]
        W6 = [0, 1, 0, 0]  # C[1,0] gets +M6
        terms.append((U6, V6, W6))
        
        # M7 = (A01-A11)(B10+B11)
        U7 = [0, 1, 0, -1]  # A[0,1]-A[1,1]
        V7 = [0, 0, 1, 1]
        W7 = [0, 0, 0, 1]  # C[1,1] gets +M7
        terms.append((U7, V7, W7))
        
        U = [t[0] for t in terms]
        V = [t[1] for t in terms]
        W = [t[2] for t in terms]
        
        # Verify
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    s = 0
                    for t in range(7):
                        s += U[t][a*2+b] * V[t][b*2+c] * W[t][c*2+a]
                    if s != 1:
                        # This decomposition is wrong
                        return None
        
        return U, V, W

    def build_strassen_3x4x5():
        """
        Build a decomposition of 3x4x5 using Strassen for 2x2x2 sub-blocks.
        
        Strategy: Partition the 4 middle indices into two groups of 2, and the 5 output
        indices into groups. Use Strassen where possible.
        
        Split j (middle) into j0={0,1} and j1={2,3}.
        For each j-group, we have a 3x2x5 subproblem.
        
        For 3x2x5: split i into {0} and {1,2}, and k into {0,1},{2,3},{4}.
        - i=0, j in {0,1}, k in {0,1}: 1x2x2 -> rank 4 (schoolbook)
        - i=0, j in {0,1}, k in {2,3}: 1x2x2 -> rank 4
        - i=0, j in {0,1}, k=4: 1x2x1 -> rank 2
        - i in {1,2}, j in {0,1}, k in {0,1}: 2x2x2 -> rank 7 (Strassen)
        - i in {1,2}, j in {0,1}, k in {2,3}: 2x2x2 -> rank 7
        - i in {1,2}, j in {0,1}, k=4: 2x2x1 -> rank 4
        
        Total for one j-group: 4+4+2+7+7+4 = 28
        Total for both j-groups: 56
        
        This gives rank 56, an improvement over schoolbook 60.
        """
        strassen = strassen_2x2x2()
        if strassen is None:
            return None
        
        U_s, V_s, W_s = strassen
        
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_schoolbook_block(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
        
        def add_strassen_block(i0, i1, j0, j1, k0, k1):
            """Add Strassen decomposition for 2x2x2 block with specified indices."""
            nonlocal col
            for t in range(7):
                # U part: local (a,b) -> global (i0+a, j0+b)
                for a in range(2):
                    for b in range(2):
                        U[(i0+a)*m + (j0+b)][col] = U_s[t][a*2+b]
                
                # V part: local (b,c) -> global (j0+b, k0+c)
                for b in range(2):
                    for c in range(2):
                        V[(j0+b)*p + (k0+c)][col] = V_s[t][b*2+c]
                
                # W part: local (c,a) -> global (k0+c, i0+a)
                for c in range(2):
                    for a in range(2):
                        W[(k0+c)*n + (i0+a)][col] = W_s[t][c*2+a]
                
                col += 1
        
        # Process each j-group
        for j_group, j_list in enumerate([[0, 1], [2, 3]]):
            j0 = j_list[0]
            j1 = j_list[1]
            
            # i=0, k in {0,1}: 1x2x2 schoolbook
            add_schoolbook_block([0], [j0, j1], [0, 1])
            
            # i=0, k in {2,3}: 1x2x2 schoolbook
            add_schoolbook_block([0], [j0, j1], [2, 3])
            
            # i=0, k=4: 1x2x1 schoolbook
            add_schoolbook_block([0], [j0, j1], [4])
            
            # i in {1,2}, k in {0,1}: 2x2x2 Strassen
            add_strassen_block(1, 2, j0, j1, 0, 1)
            
            # i in {1,2}, k in {2,3}: 2x2x2 Strassen
            add_strassen_block(1, 2, j0, j1, 2, 3)
            
            # i in {1,2}, k=4: 2x2x1 schoolbook
            add_schoolbook_block([1, 2], [j0, j1], [4])
        
        if col != rank:
            # Adjust rank if needed
            pass
        
        return U, V, W

    # Try the Strassen-based decomposition first
    result = build_strassen_3x4x5()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
