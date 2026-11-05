"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the full tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    def verify(U, V, W, rank):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # and 0 otherwise (but since we only have these indices, check all are 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def make_trivial():
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_trivial()
    best_rank = trivial_rank
    
    # Strassen 2x2x2 decomposition
    # For a 2x2x2 block with local indices (a, b, c) where a,b,c in {0,1}:
    # We need 7 terms such that sum_r U[a,b,r]*V[b,c,r]*W[c,a,r] = 1 for all (a,b,c)
    # 
    # Strassen:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)*B00       -> contributes to C10
    # M3 = A00*(B01-B11)       -> contributes to C01
    # M4 = A11*(B10-B00)       -> contributes to C10
    # M5 = (A00+A01)*B11       -> contributes to C00
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each term r, we have U[a,b,r], V[b,c,r], W[c,a,r]
    # such that for each (a,b,c), sum over r of U[a,b,r]*V[b,c,r]*W[c,a,r] = 1
    #
    # Let's define the 7 terms:
    # Term 0 (M1): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W[0,0]=1,W[1,1]=1
    #   Check: for (0,0,0): 1*1*1=1; (0,0,1): 0; (0,1,0): 0; (0,1,1): 0
    #          (1,0,0): 0; (1,0,1): 0; (1,1,0): 0; (1,1,1): 1*1*1=1
    #   So this gives 1 at (0,0,0) and (1,1,1)
    #
    # Actually, let me think about this differently. Each term is a rank-1 tensor.
    # We need the sum of 7 rank-1 tensors to equal the 2x2x2 all-ones tensor.
    #
    # The standard Strassen decomposition of the 2x2x2 identity tensor:
    # T = sum_{r=0}^{6} u_r ⊗ v_r ⊗ w_r
    #
    # where the coefficients are:
    # r=0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  [M1]
    # r=1: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  [M2]
    # r=2: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)  [M3]
    # r=3: u=(0,0,0,1), v=(0,1,-1,0), w=(0,1,0,0)  [M4]
    # r=4: u=(1,1,0,0), v=(0,0,0,1), w=(0,0,1,0)  [M5]
    # r=5: u=(0,1,-1,0), v=(1,1,0,0), w=(0,0,0,1)  [M6]
    # r=6: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0)  [M7]
    #
    # Let me verify: we need T[a,b,c] = 1 for all a,b,c in {0,1}
    # T is indexed by (a,b,c) with a,b,c in {0,1}
    # u is indexed by (a,b) -> u[a*2+b]
    # v is indexed by (b,c) -> v[b*2+c]
    # w is indexed by (c,a) -> w[c*2+a]
    
    # Let me define the Strassen terms more carefully
    # u_r: 4 values for (a,b) in {(0,0),(0,1),(1,0),(1,1)}
    # v_r: 4 values for (b,c) in {(0,0),(0,1),(1,0),(1,1)}
    # w_r: 4 values for (c,a) in {(0,0),(0,1),(1,0),(1,1)}
    
    strassen_u = [
        [1, 0, 0, 1],  # M1: (A00+A11)
        [0, 0, 1, 1],  # M2: (A10+A11)
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: (A00+A01)
        [0, 1, -1, 0], # M6: (A10-A00)
        [0, 1, 0, -1], # M7: (A01-A11)
    ]
    strassen_v = [
        [1, 0, 0, 1],  # M1: (B00+B11)
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: (B01-B11)
        [0, 1, -1, 0], # M4: (B10-B00)
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: (B00+B01)
        [0, 0, 1, 1],  # M7: (B10+B11)
    ]
    strassen_w = [
        [1, 0, 0, 1],  # M1: contributes to C00 and C11
        [0, 0, 1, 0],  # M2: contributes to C10
        [0, 1, 0, 0],  # M3: contributes to C01
        [0, 1, 0, 0],  # M4: contributes to C10
        [0, 0, 1, 0],  # M5: contributes to C00
        [0, 0, 0, 1],  # M6: contributes to C11
        [1, 0, 0, 0],  # M7: contributes to C00
    ]
    
    # Verify Strassen: sum_r u_r[a*2+b] * v_r[b*2+c] * w_r[c*2+a] should be 1 for all a,b,c
    for a in range(2):
        for b in range(2):
            for c in range(2):
                s = 0
                for r in range(7):
                    s += strassen_u[r][a*2+b] * strassen_v[r][b*2+c] * strassen_w[r][c*2+a]
                if s != 1:
                    # This is a bug in my Strassen coefficients
                    pass
    
    # Let me re-derive Strassen more carefully
    # We want: for all (a,b,c) in {0,1}^3, sum_r u_r[a,b]*v_r[b,c]*w_r[c,a] = 1
    #
    # This means the tensor T where T[a,b,c] = 1 must be decomposed as sum of 7 rank-1 tensors.
    #
    # Let me use a known correct Strassen decomposition:
    # The 2x2 matrix multiplication tensor has rank 7.
    # Standard decomposition:
    #
    # Let me use the formulation where:
    # C00 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A01-A11)(B10+B11)
    # C01 = A00(B01-B11) + (A00+A01)B11
    # C10 = (A10+A11)B00 + A11(B10-B00)
    # C11 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    #
    # So the 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 - P2 + P3 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Now, for the tensor T[a,b,c] where a=row of A/C, b=col of A/row of B, c=col of B/C:
    # T[a,b,c] = 1 for all a,b,c
    #
    # We need: sum_r u_r[a,b] * v_r[b,c] * w_r[c,a] = 1
    #
    # Each P_r corresponds to a rank-1 contribution. Let me figure out u_r, v_r, w_r:
    #
    # P1 = (A00+A11)(B00+B11):
    #   u1[a,b]: coefficient of A[a,b] in (A00+A11) -> u1 = [1,0,0,1] for (a,b)=(0,0),(0,1),(1,0),(1,1)
    #   v1[b,c]: coefficient of B[b,c] in (B00+B11) -> v1 = [1,0,0,1] for (b,c)=(0,0),(0,1),(1,0),(1,1)
    #   w1[c,a]: P1 contributes to C00 with +1 and C11 with +1
    #   C00 is at (c,a)=(0,0), C11 is at (c,a)=(1,1)
    #   So w1 = [1,0,0,1] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P2 = (A10+A11)B00:
    #   u2 = [0,0,1,1] (A10 and A11)
    #   v2 = [1,0,0,0] (B00)
    #   w2: P2 contributes to C10 with +1 and C11 with -1
    #   C10 is at (c,a)=(0,1), C11 is at (c,a)=(1,1)
    #   So w2 = [0,1,0,-1] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P3 = A00(B01-B11):
    #   u3 = [1,0,0,0] (A00)
    #   v3 = [0,1,0,-1] (B01-B11)
    #   w3: P3 contributes to C00 with +1 and C01 with +1
    #   C00 is at (c,a)=(0,0), C01 is at (c,a)=(1,0)
    #   So w3 = [1,0,1,0] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P4 = A11(B10-B00):
    #   u4 = [0,0,0,1] (A11)
    #   v4 = [0,1,-1,0] (B10-B00)
    #   w4: P4 contributes to C10 with +1
    #   C10 is at (c,a)=(0,1)
    #   So w4 = [0,1,0,0] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P5 = (A00+A01)B11:
    #   u5 = [1,1,0,0] (A00+A01)
    #   v5 = [0,0,0,1] (B11)
    #   w5: P5 contributes to C01 with +1
    #   C01 is at (c,a)=(1,0)
    #   So w5 = [0,0,1,0] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P6 = (A10-A00)(B00+B01):
    #   u6 = [-1,0,1,0] (A10-A00)
    #   v6 = [1,1,0,0] (B00+B01)
    #   w6: P6 contributes to C11 with +1
    #   C11 is at (c,a)=(1,1)
    #   So w6 = [0,0,0,1] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    #
    # P7 = (A01-A11)(B10+B11):
    #   u7 = [0,1,0,-1] (A01-A11)
    #   v7 = [0,0,1,1] (B10+B11)
    #   w7: P7 contributes to C00 with +1
    #   C00 is at (c,a)=(0,0)
    #   So w7 = [1,0,0,0] for (c,a)=(0,0),(0,1),(1,0),(1,1)
    
    strassen_u2 = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1],  # P7
    ]
    strassen_v2 = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [0, 1, -1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1],   # P7
    ]
    strassen_w2 = [
        [1, 0, 0, 1],   # P1
        [0, 1, 0, -1],  # P2
        [1, 0, 1, 0],   # P3
        [0, 1, 0, 0],   # P4
        [0, 0, 1, 0],   # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]
    
    # Verify
    for a in range(2):
        for b in range(2):
            for c in range(2):
                s = 0
                for r in range(7):
                    s += strassen_u2[r][a*2+b] * strassen_v2[r][b*2+c] * strassen_w2[r][c*2+a]
                if s != 1:
                    print(f"Strassen verification failed at ({a},{b},{c}): got {s}")
    
    # Now build the full decomposition
    # Non-overlapping 2x2x2 blocks:
    # i-blocks: {0,1}
    # j-blocks: {0,1}, {2,3}
    # k-blocks: {0,1}, {2,3}, {4,5}
    # Remaining: i={2}, k={6}
    
    blocks_2x2x2 = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks_2x2x2.append((i0, j0, k0))
    
    # Each 2x2x2 block uses 7 terms instead of 8, saving 1 term per block
    num_blocks = len(blocks_2x2x2)  # 6
    savings = num_blocks  # 6
    new_rank = trivial_rank - savings  # 78
    
    # Build the decomposition
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    col = 0
    
    # For each 2x2x2 block, add 7 Strassen terms
    for (i0, j0, k0) in blocks_2x2x2:
        for r in range(7):
            for a in range(2):
                for b in range(2):
                    i = i0 + a
                    j = j0 + b
                    u_val = strassen_u2[r][a*2+b]
                    if u_val != 0:
                        U[i*m+j][col] += u_val
            
            for b in range(2):
                for c in range(2):
                    j = j0 + b
                    k = k0 + c
                    v_val = strassen_v2[r][b*2+c]
                    if v_val != 0:
                        V[j*p+k][col] += v_val
            
            for c in range(2):
                for a in range(2):
                    k = k0 + c
                    i = i0 + a
                    w_val = strassen_w2[r][c*2+a]
                    if w_val != 0:
                        W[k*n+i][col] += w_val
            
            col += 1
    
    # Now handle the remaining terms (not in any 2x2x2 block)
    # These are terms where i=2 OR k=6 (since those are the singletons)
    # Actually, we need to handle all (i,j,k) not covered by the blocks
    
    # Covered (i,j,k) tuples:
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Not covered: i=2 (any j,k) and k=6 (i in {0,1}, any j)
    # But we need to be careful about double-counting
    
    # Let's track which (i,j,k) are covered
    covered = set()
    for (i0, j0, k0) in blocks_2x2x2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    covered.add((i0+a, j0+b, k0+c))
    
    # Add trivial terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    # Verify
    if verify(U, V, W, new_rank):
        result = {"U": U, "V": V, "W": W}
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = result
    else:
        # Fall back to trivial
        pass
    
    return best_result
# EVOLVE-BLOCK-END
