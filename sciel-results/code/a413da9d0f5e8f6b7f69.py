import numpy as np

# Rank-23 decomposition for 3x3 matrix multiplication
# Source: Landsberg, 2010, "The asymptotic rank of matrix multiplication"
# The decomposition is given by 23 terms (U_r, V_r, W_r)
# Each U_r, V_r, W_r is a 3x3 matrix.
# vec(Z) = sum_r (vec(U_r^T X)) (vec(V_r^T Y)) W_r? No.
# The standard form is T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} is for n=2?
# For n=3, indices are (i,j) for X, (j,k) for Y, (i,k) for Z.
# The tensor is T_{(i,j), (j,k), (i,k)}.
# The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
# u_r is a 9-dim vector (flattened 3x3 matrix U_r).
# v_r is a 9-dim vector (flattened 3x3 matrix V_r).
# w_r is a 9-dim vector (flattened 3x3 matrix W_r).
# The condition is: For all X, Y,
# XY = sum_r (trace(U_r^T X) trace(V_r^T Y)) W_r ? No.
# The bilinear form is:
# Z_{ik} = sum_{j} X_{ij} Y_{jk} = sum_r (sum_{i,j} U_{r,ij} X_{ij}) (sum_{j,k} V_{r,jk} Y_{jk}) W_{r,ik}
# This is NOT correct. The term (sum U X) (sum V Y) is a scalar times W.
# The correct identity is:
# XY = sum_r (U_r \circ X) (V_r \circ Y) W_r^T ? No.
# The decomposition of the multiplication tensor T is:
# T_{I, J, K} = sum_r U_{I, r} V_{J, r} W_{K, r}
# where I, J, K are multi-indices.
# I = (i, j), J = (j, k), K = (i, k).
# So for a fixed r, U_r is a 3x3 matrix, V_r is a 3x3 matrix, W_r is a 3x3 matrix.
# The equation is:
# (XY)_{ik} = sum_{j} X_{ij} Y_{jk} = sum_r W_{r, ik} ( sum_{i', j'} U_{r, i' j'} X_{i' j'} ) ( sum_{j', k'} V_{r, j' k'} Y_{j' k'} )
# This is also not quite right. The sums over i', j' and j', k' are independent.
# The correct expansion is:
# sum_r ( sum_{i,j} U_{r,ij} X_{ij} ) ( sum_{j,k} V_{r,jk} Y_{jk} ) W_{r,ik}
# This equals sum_{i,k} W_{r,ik} ( ... ) ( ... ).
# This works if the term is rank 1 in the tensor sense.
# Yes, a rank 1 tensor is u \otimes v \otimes w.
# T_{I, J, K} = u_I v_J w_K.
# So T_{(i,j), (j,k), (i,k)} = U_{r,ij} V_{r,jk} W_{r,ik}.
# Summing over r:
# sum_r U_{r,ij} V_{r,jk} W_{r,ik} = delta_{i, i} delta_{j, j} delta_{k, k} ?
# No, it must equal 1 if the indices match the multiplication structure.
# The multiplication tensor has 1 at T_{(i,j), (j,k), (i,k)} and 0 elsewhere.
# So we need:
# sum_r U_{r, i_1 j_1} V_{r, j_2 k_2} W_{r, i_3 k_3} = 1 if i_1=i_3, j_1=j_2, k_2=k_3.
# And 0 otherwise.

# I will use a known rank-23 decomposition.
# One such decomposition is available in literature.
# I will hardcode the 23 triples of 3x3 matrices.

# Note: The specific decomposition matters for correctness.
# I will use the one from "A note on the rank of the 3x3 matrix multiplication tensor".

# To save space and ensure correctness, I will use a known valid decomposition.
# If I cannot recall the exact coefficients, I might use the schoolbook rank 27 for 3x3?
# If I use rank 27 for 3x3, the total rank is 7 * 27 = 189.
# 189 is still better than 216.
# Score = 216 / 189 = 1.1428.
# If I use rank 23, score = 216 / 161 = 1.3416.
# I will try to implement rank 23. If I fail, I fall back to 27.

# Actually, I can just use the schoolbook for 3x3 (rank 27) to be safe,
# then the rank is 189.
# Is 189 better than 161? Yes.
# Is 189 better than 216? Yes.
# Can I easily find the rank 23 coefficients?
# Yes, they are public.
# But hardcoding 23 * 9 * 3 = 621 numbers is risky if I make a typo.
# Using rank 27 is safer.
# Rank 27 decomposition is trivial:
# For each i, j, k in 0..2:
# U[i*3+j, c] = 1 if c corresponds to (i,j,k)
# V[j*3+k, c] = 1 if c corresponds to (i,j,k)
# W[k*3+i, c] = 1 if c corresponds to (i,j,k)
# This is exactly the seed logic but for n=3.

# Let's use Rank 27 for 3x3 blocks.
# Total Rank = 7 * 27 = 189.
# This is a valid improvement.
# Can I do better?
# Maybe use a rank 23 decomposition if I can find it quickly.
# I will stick to Rank 27 for 3x3 to ensure validity.
# 189 < 216.

def construct():
    n, m, p = 6, 6, 6
    # Use 2x2 blocks of 3x3 matrices
    # Strassen's algorithm requires 7 multiplications of 3x3 matrices
    # Each 3x3 multiplication is done with schoolbook rank 27
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # s: 0..6
    # alpha[s][block_a]
    # beta[s][block_b]
    # gamma[c_block][s]
    
    alpha = [
        {0: 1, 3: 1},       # M1: A00 + A11 (0:00, 3:11)
        {2: 1, 3: 1},       # M2: A10 + A11 (2:10, 3:11)
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01 (0:00, 1:01)
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    beta = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    gamma = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00: M1 + M4 - M5 + M7 (0:M1, 3:M4, 4:M5, 6:M7)
        {2: 1, 4: 1},                 # C01: M3 + M5 (2:M3, 4:M5)
        {1: 1, 3: 1},                 # C10: M2 + M4 (1:M2, 3:M4)
        {0: 1, 1: -1, 2: 1, 5: 1}    # C11: M1 - M2 + M3 + M6 (0:M1, 1:M2, 2:M3, 5:M6)
    ]
    
    # Helper to get global index from block and local index
    # Block index b: 0:00, 1:01, 2:10, 3:11
    # Local index l: 0..8 (row 0..2, col 0..2)
    def get_global_index(block, local):
        # block = i*2 + j? No, 0:00, 1:01, 2:10, 3:11
        # i = block // 2, j = block % 2
        i = block // 2
        j = block % 2
        # local = p*3 + q
        p = local // 3
        q = local % 3
        # Global row = i*3 + p
        # Global col = j*3 + q
        return (i * 3 + p) * 6 + (j * 3 + q)
    
    def get_block_and_local(global_idx):
        row = global_idx // 6
        col = global_idx % 6
        i = row // 3
        j = col // 3
        p = row % 3
        q = col % 3
        block = i * 2 + j
        local = p * 3 + q
        return block, local
    
    # Fill U, V, W
    for s in range(7):
        for r in range(27):
            c = s * 27 + r
            # r corresponds to i, j, k in 0..2
            i3 = r // 9
            j3 = (r % 9) // 3
            k3 = r % 3
            
            # U part
            # U[global_idx, c] = alpha[s][block_a] * (1 if local_idx corresponds to i3, j3 else 0)
            # For a given s, the linear form is sum_{a} alpha[s][a] * (A_a)_{i3, j3}
            # So for each global index I in A:
            # I belongs to block a(I) and has local index (p, q).
            # If (p, q) == (i3, j3), then U[I, c] += alpha[s][a(I)]
            
            for I in range(36):
                block_a, local_a = get_block_and_local(I)
                p_a = local_a // 3
                q_a = local_a % 3
                if p_a == i3 and q_a == j3:
                    if block_a in alpha[s]:
                        U[I][c] = alpha[s][block_a]
            
            # V part
            # V[global_idx, c] = beta[s][block_b] * (1 if local_idx corresponds to j3, k3 else 0)
            for J in range(36):
                block_b, local_b = get_block_and_local(J)
                p_b = local_b // 3
                q_b = local_b % 3
                if p_b == j3 and q_b == k3:
                    if block_b in beta[s]:
                        V[J][c] = beta[s][block_b]
            
            # W part
            # W[global_idx, c] = gamma[c_block][s] * (1 if local_idx corresponds to i3, k3 else 0)
            for K in range(36):
                block_c, local_c = get_block_and_local(K)
                p_c = local_c // 3
                q_c = local_c % 3
                if p_c == i3 and q_c == k3:
                    if block_c in gamma and s in gamma[block_c]:
                        W[K][c] = gamma[block_c][s]
    
    return {"U": U, "V": V, "W": W}
