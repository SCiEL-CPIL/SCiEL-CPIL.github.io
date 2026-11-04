"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 5, 6

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen-based decomposition for 4x5x6
    # Split: i into [0,1],[2,3]; j into [0,1],[2,3],[4]; k into [0,1],[2,3],[4,5]
    # 12 blocks of 2x2x2 (use Strassen rank 7) + 6 blocks of 2x1x2 (schoolbook rank 4)
    # Total rank: 12*7 + 6*4 = 84 + 24 = 108

    i_groups = [[0, 1], [2, 3]]
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4, 5]]

    U = []
    V = []
    W = []
    rank = 0

    def add_term(u_val, v_val, w_val, i, j, k):
        """Add a rank-1 term with given coefficients."""
        nonlocal rank
        # Append to existing rows or create new
        for _ in range(len(U)):
            U[_].append(0)
        for _ in range(len(V)):
            V[_].append(0)
        for _ in range(len(W)):
            W[_].append(0)
        U[i * m + j][rank] = u_val
        V[j * p + k][rank] = v_val
        W[k * n + i][rank] = w_val
        rank += 1

    # Initialize U, V, W with correct number of rows
    num_U = n * m  # 20
    num_V = m * p  # 30
    num_W = p * n  # 24
    U = [[0] * 0 for _ in range(num_U)]
    V = [[0] * 0 for _ in range(num_V)]
    W = [[0] * 0 for _ in range(num_W)]

    # Strassen's 7 terms for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For each 2x2x2 block, we need to express the 8 schoolbook terms as 7 Strassen terms.
    # Each Strassen term M_t is itself a product of two linear combinations, which in the
    # tensor decomposition means we need to expand each M_t into rank-1 terms.
    #
    # Actually, in the tensor decomposition framework, each "multiplication" of two 2x2 matrices
    # is itself a 2x2x2 tensor of rank 8 (schoolbook) or 7 (Strassen). So when we say we use
    # Strassen for a 2x2x2 block, we mean the 8 terms (i,j,k) in that block are expressed
    # using 7 rank-1 tensors instead of 8.
    #
    # The Strassen decomposition of the 2x2x2 matmul tensor:
    # For indices (a,b,c) in {0,1}^3, the tensor T[a,b,c] = 1 for all a,b,c.
    # Strassen expresses this as sum of 7 rank-1 tensors.

    # The 7 Strassen rank-1 terms for 2x2x2:
    # Each term is (u_a, v_b, w_c) for a,b,c in {0,1}
    strassen_terms_2x2x2 = [
        # (u0, u1, v0, v1, w0, w1)
        # M1: (a00+a11)(b00+b11) -> contributes to C00 and C11
        # As a rank-1 tensor: u = [1,1], v = [1,1], w = [1,-1] gives C00=1, C11=-1... 
        # Let me look up the exact Strassen tensor decomposition.
    ]

    # The standard Strassen decomposition of the 2x2x2 matrix multiplication tensor:
    # T[a,b,c] = sum_{t=1}^{7} u_t[a] * v_t[b] * w_t[c]
    # where T[a,b,c] = 1 for all a,b,c in {0,1}.
    #
    # The 7 terms (u, v, w) are:
    strassen_2x2x2 = [
        # (u0, u1, v0, v1, w0, w1)
        ([1, 1], [1, 1], [1, -1]),   # M1
        ([0, 1], [1, 0], [1, 0]),    # M2  
        ([1, 0], [0, 1], [1, -1]),   # M3
        ([0, 1], [1, 0], [-1, 1]),   # M4
        ([1, 1], [0, 1], [0, 1]),    # M5
        ([1, -1], [1, 1], [1, 0]),   # M6
        ([-1, 1], [1, 0], [1, 1]),   # M7
    ]

    # Verify: sum_t u_t[a]*v_t[b]*w_t[c] should = 1 for all a,b,c
    # Let me check a=0,b=0,c=0:
    # T1: 1*1*1 = 1
    # T2: 0*1*1 = 0
    # T3: 1*0*1 = 0
    # T4: 0*1*(-1) = 0
    # T5: 1*0*0 = 0
    # T6: 1*1*1 = 1
    # T7: -1*1*1 = -1
    # Sum = 1+0+0+0+0+1-1 = 1 ✓
    # a=0,b=0,c=1:
    # T1: 1*1*(-1) = -1
    # T2: 0*1*0 = 0
    # T3: 1*0*(-1) = 0
    # T4: 0*1*1 = 0
    # T5: 1*0*1 = 0
    # T6: 1*1*0 = 0
    # T7: -1*1*1 = -1
    # Sum = -1+0+0+0+0+0-1 = -2 ✗ (should be 1)

    # My Strassen terms are wrong. Let me use the correct ones.
    # The correct Strassen decomposition:
    # C = AB where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # In tensor form, T[i,j,k] = 1 means the (i,j)-th entry of A times (j,k)-th entry of B
    # contributes to (i,k)-th entry of C.
    #
    # For the 2x2x2 tensor with indices (i,j,k) in {0,1}^3:
    # T[i,j,k] = 1 for all i,j,k.
    #
    # We need: sum_t u_t[i] * v_t[j] * w_t[k] = 1 for all i,j,k.
    #
    # From Strassen:
    # C[0,0] = M1+M4-M5+M7 = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
    # This is getting complicated. Let me just directly verify by expanding.

    # Let me use a verified Strassen decomposition.
    # Reference: the 2x2x2 matmul tensor has rank 7 with these factors:
    strassen_2x2x2_correct = [
        # (u for i-index, v for j-index, w for k-index)
        ([1, 1], [1, 1], [1, -1]),
        ([1, 0], [1, 0], [1, 0]),
        ([1, 0], [0, 1], [1, -1]),
        ([0, 1], [1, 0], [-1, 1]),
        ([1, 1], [0, 1], [0, 1]),
        ([1, -1], [1, 1], [1, 0]),
        ([-1, 1], [1, 0], [1, 1]),
    ]

    # Verify all 8 entries
    for a in range(2):
        for b in range(2):
            for c in range(2):
                s = 0
                for (u, v, w) in strassen_2x2x2_correct:
                    s += u[a] * v[b] * w[c]
                if s != 1:
                    pass  # Will check below

    # Let me compute properly:
    # Term 1: u=[1,1], v=[1,1], w=[1,-1]
    # Term 2: u=[1,0], v=[1,0], w=[1,0]
    # Term 3: u=[1,0], v=[0,1], w=[1,-1]
    # Term 4: u=[0,1], v=[1,0], w=[-1,1]
    # Term 5: u=[1,1], v=[0,1], w=[0,1]
    # Term 6: u=[1,-1], v=[1,1], w=[1,0]
    # Term 7: u=[-1,1], v=[1,0], w=[1,1]

    # (0,0,0): 1*1*1 + 1*1*1 + 1*0*1 + 0*1*(-1) + 1*0*0 + 1*1*1 + (-1)*1*1 = 1+1+0+0+0+1-1 = 2 ✗

    # These aren't right either. Let me derive from scratch.
    # We need sum_t u_t[i]*v_t[j]*w_t[k] = 1 for all i,j,k in {0,1}.
    # This is 8 equations, 21 unknowns (7 terms x 3 vectors x 2 entries each).
    # 
    # Let me use the known Strassen formula directly.
    # In Strassen, we compute 7 products M1..M7 of linear combinations.
    # Each M_t = (sum_i alpha_t[i]*A[i,:]) * (sum_j beta_t[j]*B[j,:])
    # But in our tensor, A and B are 2x2, so:
    # M_t = (sum_{i,j} alpha_t[i,j] * A[i,j]) * (sum_{j',k} beta_t[j',k] * B[j',k])
    # Wait, that's not right either. In Strassen, the "products" are of 2x2 matrices.
    #
    # I think the cleanest approach is to just directly construct the rank-1 terms.
    # For the 2x2x2 case, I'll solve the system of equations.

    # Actually, let me just use a known correct decomposition.
    # From literature, the Strassen decomposition of 2x2x2 matmul tensor:
    # Using 0-indexed (i,j,k) where i=row of A, j=col of A / row of B, k=col of B:
    
    # Let me just brute-force verify and construct.
    # I'll use the following known-correct Strassen factors:
    # (verified by direct computation)
    S = [
        ([1, 1], [1, 1], [1, -1]),
        ([1, 0], [1, 0], [1, 0]),  
        ([1, 0], [0, 1], [1, -1]),
        ([0, 1], [1, 0], [-1, 1]),
        ([1, 1], [0, 1], [0, 1]),
        ([1, -1], [1, 1], [1, 0]),
        ([-1, 1], [1, 0], [1, 1]),
    ]
    
    # Check (0,0,0): 
    # 1*1*1=1, 1*1*1=1, 1*0*1=0, 0*1*(-1)=0, 1*0*0=0, 1*1*1=1, -1*1*1=-1 => 1+1+0+0+0+1-1=2
    # Not 1. So this is wrong.

    # Let me solve it properly. We need 7 vectors (u,v,w) such that
    # sum_t u_t[i]*v_t[j]*w_t[k] = 1 for all i,j,k in {0,1}.
    #
    # By symmetry and known results, one valid decomposition is:
    # I'll use the one from the Strassen algorithm directly.
    # 
    # Strassen computes:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # In the tensor decomposition, each M_t is a "product" of two linear combinations.
    # But in our framework, we need rank-1 tensors. Each M_t corresponds to a set of
    # rank-1 terms that together compute that product.
    #
    # Actually, I think I'm overcomplicating this. In the tensor decomposition of
    # matrix multiplication, a rank-1 term is just a single triple (i,j,k) with
    # coefficients. The "Strassen" approach combines multiple (i,j,k) triples into
    # fewer terms by using linear combinations.
    #
    # Let me just directly find 7 rank-1 tensors that sum to the all-ones 2x2x2 tensor.
    # This is equivalent to finding a rank-7 decomposition of the 2x2x2 all-ones tensor.
    # But wait, the all-ones 2x2x2 tensor has rank... let me think.
    #
    # The 2x2x2 matmul tensor T[i,j,k]=1 has the property that for any fixed i,j,
    # T[i,j,:]=[1,1], etc. It's actually the tensor that represents C=AB where
    # C[i,k] = sum_j A[i,j]*B[j,k], and T[i,j,k]=1 means A[i,j]*B[j,k] contributes to C[i,k].
    #
    # The rank of this tensor is 7 (Strassen). So there exist 7 triples of vectors
    # (u_t, v_t, w_t) such that sum_t u_t[i]*v_t[j]*w_t[k] = 1 for all i,j,k.
    #
    # Let me just solve this system numerically and then round to integers/half-integers.
    
    import numpy as np
    
    # Set up the linear system: 8 equations, 7*6=42 unknowns
    # Variables: u_t[0], u_t[1], v_t[0], v_t[1], w_t[0], w_t[1] for t=0..6
    # Equation (i,j,k): sum_t u_t[i]*v_t[j]*w_t[k] = 1
    
    # This is nonlinear (products of variables). Use a different approach.
    # 
    # Known result: one valid Strassen decomposition uses these factors:
    # I'll look it up / derive carefully.
    
    # From Strassen (1969), the 7 products are:
    # M1 = (a+d)(e+h)  where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Now, each M_t is a scalar product of two linear forms. In the tensor decomposition,
    # we need to express the contribution of M_t to each C[i,k] entry.
    #
    # M1 contributes: +1 to C[0,0], +1 to C[1,1]
    # M2 contributes: +1 to C[1,0], -1 to C[1,1]
    # M3 contributes: +1 to C[0,1], +1 to C[1,1]
    # M4 contributes: +1 to C[0,0], +1 to C[1,0]
    # M5 contributes: -1 to C[0,0], +1 to C[0,1]
    # M6 contributes: +1 to C[1,1]
    # M7 contributes: +1 to C[0,0], +1 to C[1,1]
    #
    # Wait, let me recompute:
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    #
    # So the contribution of M_t to C[i,k] is the coefficient of M_t in C[i,k].
    #
    # Now, M_t = (linear form in A) * (linear form in B)
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # In the tensor decomposition T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]:
    # Each rank-1 term c has U[i,j,c], V[j,k,c], W[k,i,c].
    #
    # For M_t, the "A factor" is a linear form in A[*,*], and the "B factor" is a linear
    # form in B[*,*]. The contribution to C[i,k] is coeff_t[i,k] * M_t.
    #
    # So for each M_t, we need to create rank-1 terms that:
    # - Have U[i,j] = alpha_t[i,j] (coefficient of A[i,j] in the A-factor of M_t)
    # - Have V[j,k] = beta_t[j,k] (coefficient of B[j,k] in the B-factor of M_t)
    # - Have W[k,i] = coeff_t[i,k] (coefficient of M_t in C[i,k])
    #
    # But wait, that's not quite right either. A single rank-1 term has fixed U[i,j], V[j,k], W[k,i]
    # for all i,j,k. So we can't have different coefficients for different (i,j) in U.
    #
    # I think the correct interpretation is:
    # The tensor T[i,j,k] = 1 (for matmul) can be decomposed as:
    # T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]
    #
    # where each t is a "Strassen term" and U_t, V_t, W_t are 2x2 matrices (for the 2x2x2 case).
    #
    # For M1: A-factor = [1,0;0,1] (identity, since A[0,0]+A[1,1]), B-factor = [1,1;1,1]... no.
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # The A-factor picks out A[0,0] and A[1,1] with coefficient 1.
    # The B-factor picks out B[0,0] and B[1,1] with coefficient 1.
    #
    # But in the tensor decomposition, U_t[i,j] is a single number for each (i,j).
    # If U_t = [[1,0],[0,1]] (identity), V_t = [[1,1],[1,1]], W_t = [[1,-1],[1,1]]... 
    # then sum over j: U_t[i,j]*V_t[j,k]*W_t[k,i]... no, that's not how it works.
    #
    # I think I've been confusing myself. Let me restart with a clear formulation.
    #
    # The matrix multiplication tensor for n x m x p:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # (This represents: C[i,k] = sum_j A[i,j]*B[j,k], and each term A[i,j]*B[j,k] 
    #  contributes with coefficient 1 to C[i,k].)
    #
    # A rank-r decomposition is: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c]*V[j,k,c]*W[k,i,c]
    #
    # For the 2x2x2 case, we need 7 terms. Let me just find them by solving.
    #
    # Due to the small size, I'll hardcode a known-correct decomposition.
    # After careful derivation, here's a valid rank-7 decomposition of the 2x2x2 matmul tensor:

    # I'll verify each one computationally in the code below.
    # Let me use this set (derived from Strassen's algorithm):
    
    # For the 2x2x2 tensor, the following 7 terms work:
    # (u0,u1,v0,v1,w0,w1) where u is for A indices, v for B indices, w for C indices
    # Note: in our convention, U[i,j,c] where i is row of A, j is col of A (=row of B)
    # V[j,k,c] where j is row of B, k is col of B
    # W[k,i,c] where k is col of B, i is row of C (=row of A)
    
    # Let me just verify by computation and use a known set.
    # I'll test several candidate sets and use the first one that works.
    
    candidates = [
        # Candidate 1: from direct Strassen translation
        [
            ([1, 1], [1, 1], [1, -1]),
            ([1, 0], [1, 0], [1, 0]),
            ([1, 0], [0, 1], [1, -1]),
            ([0, 1], [1, 0], [-1, 1]),
            ([1, 1], [0, 1], [0, 1]),
            ([1, -1], [1, 1], [1, 0]),
            ([-1, 1], [1, 0], [1, 1]),
        ],
        # Candidate 2: alternative
        [
            ([1, 0], [1, 0], [1, 0]),
            ([0, 1], [0, 1], [1, 0]),
            ([1, 0], [0, 1], [0, 1]),
            ([0, 1], [1, 0], [0, 1]),
            ([1, 1], [1, 1], [1, -1]),
            ([1, -1], [1, 1], [1, 0]),
            ([-1, 1], [1, 0], [1, 1]),
        ],
    ]
    
    valid_strassen = None
    for cand in candidates:
        ok = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(u[i]*v[j]*w[k] for (u,v,w) in cand)
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            valid_strassen = cand
            break
    
    if valid_strassen is None:
        # Fallback: use schoolbook (rank 8) for 2x2x2 blocks
        # This means total rank = 12*8 + 6*4 = 96+24 = 120 (no improvement)
        # But let's try harder to find a valid Strassen decomposition.
        # 
        # Let me derive it properly. We need 7 triples (u,v,w) with u,v,w in R^2
        # such that sum_t u_t[i]*v_t[j]*w_t[k] = 1 for all i,j,k in {0,1}.
        #
        # This is 8 equations. Let me set up and solve.
        # Due to symmetry, let's try:
        # u1=[1,0], v1=[1,0], w1=[1,0]  -> gives 1 for (0,0,0)
        # u2=[0,1], v2=[0,1], w2=[1,0]  -> gives 1 for (1,1,0)
        # u3=[1,0], v3=[0,1], w3=[0,1]  -> gives 1 for (0,1,1)
        # u4=[0,1], v4=[1,0], w4=[0,1]  -> gives 1 for (1,0,1)
        # These 4 give 1 for (0,0,0),(1,1,0),(0,1,1),(1,0,1)
        # Need: (0,0,1),(0,1,0),(1,0,0),(1,1,1) to also sum to 1.
        # Currently they sum to 0. Need 3 more terms to fix all 4.
        #
        # Let me use a computational approach: solve the system.
        import itertools
        # Try small integer/half-integer values
        vals = [-1, -0.5, 0, 0.5, 1]
        found = False
        for trial in range(100000):
            if time.time() > deadline - 30:
                break
            # Random search with small values
            terms = []
            for t in range(7):
                u = [random.choice(vals) for _ in range(2)]
                v = [random.choice(vals) for _ in range(2)]
                w = [random.choice(vals) for _ in range(2)]
                terms.append((u,v,w))
            ok = True
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        s = sum(u[i]*v[j]*w[k] for (u,v,w) in terms)
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                valid_strassen = terms
                found = True
                break
        
        if not found:
            # Use schoolbook for 2x2x2 (rank 8)
            # Total rank will be 120
            valid_strassen = None
    
    # Build the full decomposition
    if valid_strassen is not None:
        # Use Strassen for 2x2x2 blocks
        for ig_idx, ig in enumerate(i_groups):
            for jg_idx, jg in enumerate(j_groups):
                for kg_idx, kg in enumerate(k_groups):
                    if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                        # 2x2x2 block: use Strassen (7 terms)
                        # Each Strassen term is ONE rank-1 term
                        for (u, v, w) in valid_strassen:
                            for _ in range(num_U):
                                U[_].append(0)
                            for _ in range(num_V):
                                V[_].append(0)
                            for _ in range(num_W):
                                W[_].append(0)
                            # Set values for all indices in this block
                            for di in range(2):
                                for dj in range(2):
                                    i = ig[di]
                                    j = jg[dj]
                                    U[i*m+j][rank] = u[di]
                            for dj in range(2):
                                for dk in range(2):
                                    j = jg[dj]
                                    k = kg[dk]
                                    V[j*p+k][rank] = v[dj]
                            for dk in range(2):
                                for di in range(2):
                                    k = kg[dk]
                                    i = ig[di]
                                    W[k*n+i][rank] = w[dk]
                            rank += 1
                    else:
                        # Schoolbook for this block
                        for i in ig:
                            for j in jg:
                                for k in kg:
                                    for _ in range(num_U):
                                        U[_].append(0)
                                    for _ in range(num_V):
                                        V[_].append(0)
                                    for _ in range(num_W):
                                        W[_].append(0)
                                    U[i*m+j][rank] = 1
                                    V[j*p+k][rank] = 1
                                    W[k*n+i][rank] = 1
                                    rank += 1
    else:
        # Fallback: full schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for _ in range(num_U):
                        U[_].append(0)
                    for _ in range(num_V):
                        V[_].append(0)
                    for _ in range(num_W):
                        W[_].append(0)
                    U[i*m+j][rank] = 1
                    V[j*p+k][rank] = 1
                    W[k*n+i][rank] = 1
                    rank += 1

    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
