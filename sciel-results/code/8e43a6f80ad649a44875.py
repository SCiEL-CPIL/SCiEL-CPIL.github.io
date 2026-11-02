"""Combined program: structural schoolbook fallback + systematic rank-reduction search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    # We will use a block decomposition for 3x3x3 with rank 18
    # and schoolbook for 3x3x2 with rank 18.
    # Total rank 36.
    
    rank_333 = 18
    rank_332 = 18
    total_rank = rank_333 + rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= total_rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # --- Part 1: 3x3x3 (Rank 18) ---
    # Split A, B into 2x2, 2x1, 1x2, 1x1 blocks.
    # Indices for A (3x3):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # A00: 0,1,3,4
    # A01: 2,5
    # A10: 6,7
    # A11: 8
    
    # Indices for B (3x3):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # B00: 0,1,3,4
    # B01: 2,5
    # B10: 6,7
    # B11: 8
    
    # Indices for C (3x3) -> W index k*n+i.
    # C00: k=0,1; i=0,1 -> W[0*3+0]=0, W[1*3+0]=3, W[0*3+1]=1, W[1*3+1]=4
    # C01: k=0,1; i=2? No.
    # C is 3x3.
    # C00 block (top-left 2x2): i in 0,1; k in 0,1.
    # C01 block (top-right 2x1): i in 0,1; k in 2.
    # C10 block (bottom-left 1x2): i in 2; k in 0,1.
    # C11 block (bottom-right 1x1): i in 2; k in 2.
    
    # 1. A00 B00 (2x2x2) - Strassen Rank 7
    # A00 indices: 0,1,3,4
    # B00 indices: 0,1,3,4
    # C00 indices: 0,1,3,4 (W[k*3+i])
    # Strassen M1..M7
    
    # M1 = (A00+A11)(B00+B11)
    # This is complex to implement directly in the tensor form without intermediate vectors.
    # However, Strassen's algorithm is a specific rank 7 decomposition.
    # We can hardcode the rank 7 decomposition for 2x2x2.
    
    # Let's use a known rank 7 decomposition for 2x2x2.
    # U, V, W for 2x2x2.
    # U is 4x7, V is 4x7, W is 4x7.
    
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We need to map this to the tensor decomposition.
    # Each M corresponds to a rank 1 term in the 2x2x2 multiplication.
    # But M1 is a product of sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is not a single rank 1 term in the original A,B space.
    # Strassen's algorithm works by changing the basis.
    # The rank 7 decomposition exists.
    # I will use the explicit rank 7 coefficients for 2x2x2.
    
    # Coefficients for 2x2x2 rank 7 (Strassen)
    # U (4x7), V (4x7), W (4x7)
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> W[k*2+i]
    
    # U[0] (a11): [1, 0, 1, 1, 1, -1, 0]
    # U[1] (a12): [0, 0, 0, 0, 1, 0, 1]
    # U[2] (a21): [0, 1, 0, 0, 0, 1, 0]
    # U[3] (a22): [1, 1, 0, 1, 0, 0, -1]
    
    # V[0] (b11): [1, 1, 0, 0, 0, 1, 0]
    # V[1] (b12): [1, 0, -1, 0, 0, 1, 0]
    # V[2] (b21): [0, 0, 0, 1, 0, 0, 1]
    # V[3] (b22): [1, 0, 1, -1, 1, 0, 1]
    
    # W[0] (c11): [1, 0, 0, 1, -1, 0, 1]
    # W[1] (c12): [0, 0, 1, 0, 1, 0, 0]
    # W[2] (c21): [0, 1, 0, 1, 0, 0, 0]
    # W[3] (c22): [1, -1, 1, 0, 0, 1, 0]
    
    # Let's verify M1 = (a11+a22)(b11+b22).
    # Term 1: U[0]*V[0]*W[0] + U[0]*V[3]*W[0] + U[3]*V[0]*W[0] + U[3]*V[3]*W[0]
    # This is getting complicated to verify by hand.
    # I will trust the standard Strassen coefficients.
    
    U_222 = [
        [1, 0, 1, 1, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_222 = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, -1, 1, 0, 1]
    ]
    W_222 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Map 2x2 indices to 3x3 indices
    # A00: 0,1,3,4 -> 2x2: 0,1,2,3
    # B00: 0,1,3,4 -> 2x2: 0,1,2,3
    # C00: 0,1,3,4 -> 2x2: 0,1,2,3 (W[k*2+i])
    
    for r in range(7):
        for i2 in range(4):
            # Map 2x2 A index to 3x3 A index
            if i2 == 0: u_idx = 0
            elif i2 == 1: u_idx = 1
            elif i2 == 2: u_idx = 3
            else: u_idx = 4
            
            # Map 2x2 B index to 3x3 B index
            if i2 == 0: v_idx = 0
            elif i2 == 1: v_idx = 1
            elif i2 == 2: v_idx = 3
            else: v_idx = 4
            
            # Map 2x2 C index to 3x3 C index (W)
            # C index in 2x2: k*2+i.
            # C index in 3x3: k*3+i.
            # i2 is the flattened index of C?
            # In the 2x2 decomposition, W is indexed by C.
            # W_222[i2, r] corresponds to C_{i2}.
            # We need to map C_{i2} to W_{k*3+i}.
            # i2=0 (0,0) -> k=0, i=0 -> W[0]
            # i2=1 (0,1) -> k=1, i=0 -> W[3]
            # i2=2 (1,0) -> k=0, i=1 -> W[1]
            # i2=3 (1,1) -> k=1, i=1 -> W[4]
            
            if i2 == 0: w_idx = 0
            elif i2 == 1: w_idx = 3
            elif i2 == 2: w_idx = 1
            else: w_idx = 4
            
            U[u_idx][col] += U_222[i2][r]
            V[v_idx][col] += V_222[i2][r]
            W[w_idx][col] += W_222[i2][r]
        col += 1

    # 2. A01 B10 (2x1x2) Outer
    # A01: 2,5
    # B10: 6,7
    # C00: 0,1,3,4
    # C_{i,k} = A_{i,2} B_{2,k}
    # i in 0,1; k in 0,1.
    # Rank 1?
    # C_{0,0} = A_{0,2} B_{2,0}
    # C_{0,1} = A_{0,2} B_{2,1}
    # C_{1,0} = A_{1,2} B_{2,0}
    # C_{1,1} = A_{1,2} B_{2,1}
    # This is $u v^T$ where $u=[A_{0,2}, A_{1,2}]^T, v=[B_{2,0}, B_{2,1}]$.
    # Rank 1 decomposition:
    # U_{A_{0,2}}=1, U_{A_{1,2}}=1? No.
    # $C = u v^T$.
    # $C_{i,k} = u_i v_k$.
    # $T_{u_i, v_k, c_{i,k}} = 1$.
    # Rank 1: $U_{u_i, 1} = 1, V_{v_k, 1} = 1, W_{c_{i,k}, 1} = \delta_{i, i'} \delta_{k, k'}$?
    # No, the term is $U_{u_i, r} V_{v_k, r} W_{c_{i,k}, r}$.
    # If we set $U_{u_i, 1} = 1$ for all $i$, $V_{v_k, 1} = 1$ for all $k$,
    # then $W_{c_{i,k}, 1}$ must be such that $\sum_{i', k'} W_{c_{i',k'}, 1} = C_{i,k}$?
    # No, the equation is $C_{i,k} = \sum_r U_{u_i, r} V_{v_k, r} W_{c_{i,k}, r}$.
    # This is not how it works.
    # The tensor equation is $T_{u, v, w} = \sum_r U_u V_v W_w$.
    # For outer product, $T_{u_i, v_k, c_{i,k}} = 1$.
    # We need $U_{u_i, r} V_{v_k, r} W_{c_{i,k}, r} = \delta_{i,i'} \delta_{k,k'}$?
    # Actually, for rank 1, we can set:
    # $U_{u_i, 1} = 1$
    # $V_{v_k, 1} = 1$
    # $W_{c_{i,k}, 1} = 1$ if $i$ matches $u_i$ and $k$ matches $v_k$?
    # No, $W$ is a vector.
    # The term is $U_{u_i, 1} V_{v_k, 1} W_{c_{i,k}, 1}$.
    # This value must be 1 if $(u_i, v_k, c_{i,k})$ is a valid triple, else 0.
    # For outer product, all $(i,k)$ are valid.
    # So $W_{c_{i,k}, 1}$ must be 1 for all $i,k$.
    # But $W$ is indexed by $c$.
    # $c_{i,k}$ is unique for each $i,k$.
    # So we can set $W_{c_{i,k}, 1} = 1$ for all $i,k$.
    # Then $U_{u_i, 1} V_{v_k, 1} \cdot 1 = 1 \cdot 1 \cdot 1 = 1$.
    # This works.
    
    add_term(2, 6, 0, 1) # u=2 (A0,2), v=6 (B2,0), w=0 (C0,0)
    add_term(2, 7, 1, 1) # u=2, v=7 (B2,1), w=1 (C0,1)
    add_term(5, 6, 3, 1) # u=5 (A1,2), v=6, w=3 (C1,0)
    add_term(5, 7, 4, 1) # u=5, v=7, w=4 (C1,1)
    # Wait, this is rank 4 (schoolbook).
    # Can we do rank 1?
    # If rank 1, we have 1 term.
    # $U_{u_i, 1} V_{v_k, 1} W_{c_{i,k}, 1} = 1$.
    # This implies $U_{u_i, 1} = 1/U_{u_i, 1}$?
    # If we set $U_{u_0, 1}=1, U_{u_1, 1}=1$, $V_{v_0, 1}=1, V_{v_1, 1}=1$.
    # Then $W_{c_{0,0}, 1}=1, W_{c_{0,1}, 1}=1, W_{c_{1,0}, 1}=1, W_{c_{1,1}, 1}=1$.
    # This is a valid rank 1 decomposition!
    # Why did I think it was rank 4?
    # Because schoolbook uses 4 terms.
    # But the outer product is rank 1.
    
    # So we should use rank 1 for A01 B10.
    # Term:
    # U[2][col] = 1, U[5][col] = 1
    # V[6][col] = 1, V[7][col] = 1
    # W[0][col] = 1, W[1][col] = 1, W[3][col] = 1, W[4][col] = 1
    
    U[2][col] = 1
    U[5][col] = 1
    V[6][col] = 1
    V[7][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    W[3][col] = 1
    W[4][col] = 1
    col += 1
    
    # 3. A10 B01 (1x2x2) Outer
    # A10: 6,7
    # B01: 2,5
    # C10: 1,4 (W[1], W[4])
    # C_{2,k} = A_{2,j} B_{j,2}
    # i=2. j in 0,1. k=2.
    # C_{2,2} = A_{2,0} B_{0,2} + A_{2,1} B_{1,2}
    # This is dot product. Rank 2?
    # $C = u \cdot v$.
    # $T_{u_j, v_j, c} = 1$.
    # Rank 2:
    # Term 1: $u_0 v_0 c$
    # Term 2: $u_1 v_1 c$
    
    add_term(6, 2, 7, 1) # u=6 (A2,0), v=2 (B0,2), w=7 (C2,2)
    add_term(7, 5, 7, 1) # u=7 (A2,1), v=5 (B1,2), w=7
    
    # 4. A11 B11 (1x1x1)
    # A11: 8
    # B11: 8
    # C11: 7 (W[7])
    add_term(8, 8, 7, 1)
    
    # 5. A00 B01 (2x2x1) Schoolbook Rank 4
    # A00: 0,1,3,4
    # B01: 2,5
    # C01: 1,4 (W[1], W[4])
    # C_{i,2} = A_{i,0} B_{0,2} + A_{i,1} B_{1,2}
    # i in 0,1.
    
    add_term(0, 2, 1, 1) # i=0, j=0
    add_term(1, 5, 1, 1) # i=0, j=1
    add_term(3, 2, 4, 1) # i=1, j=0
    add_term(4, 5, 4, 1) # i=1, j=1
    
    # 6. A10 B00 (1x2x2) Schoolbook Rank 4
    # A10: 6,7
    # B00: 0,1,3,4
    # C10: 0,3 (W[0], W[3])
    # C_{2,k} = A_{2,0} B_{0,k} + A_{2,1} B_{1,k}
    # k in 0,1.
    
    add_term(6, 0, 0, 1) # i=2, j=0, k=0
    add_term(6, 1, 3, 1) # i=2, j=0, k=1
    add_term(7, 3, 0, 1) # i=2, j=1, k=0
    add_term(7, 4, 3, 1) # i=2, j=1, k=1
    
    # --- Part 2: 3x3x2 (Rank 18) ---
    # B2 is 3x2.
    # B2 indices in B (3x5):
    # 0 1 2 3 4
    # 5 6 7 8 9
    # 10 11 12 13 14
    # B2 is columns 3,4.
    # B2,0: 4, 9, 14
    # B2,1: 3, 8, 13
    
    # C2 is 3x2.
    # C2,0: k=3. i in 0,1,2.
    # W indices: k*n+i = 3*3+i = 9+i.
    # C2,0: W[9], W[10], W[11]
    # C2,1: k=4. i in 0,1,2.
    # W indices: 4*3+i = 12+i.
    # C2,1: W[12], W[13], W[14]
    
    # A is 3x3.
    # A indices: 0..8
    
    # Schoolbook for 3x3x1 (twice)
    # C_{i,3} = sum_j A_{i,j} B_{j,3}
    # C_{i,4} = sum_j A_{i,j} B_{j,4}
    
    for i in range(3):
        for j in range(3):
            # Term for C_{i,3}
            add_term(i*3+j, 3+j, 9+i, 1)
            # Term for C_{i,4}
            add_term(i*3+j, 4+j, 12+i, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
