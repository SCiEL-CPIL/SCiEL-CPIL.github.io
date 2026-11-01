import numpy as np
import time

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110

    # Build the 4x4x4 target tensor for 2x2x2 matrix multiplication
    # T[a,b,c] where a=i*2+k, b=ip*2+jp, c=jpp*2+kpp
    # T[a,b,c] = 1 if ip==i, jp==jpp, kpp==k else 0
    T_222 = np.zeros((4, 4, 4))
    for i in range(2):
        for k in range(2):
            for ip in range(2):
                for jp in range(2):
                    for jpp in range(2):
                        for kpp in range(2):
                            if ip == i and jp == jpp and kpp == k:
                                T_222[i*2+k, ip*2+jp, jpp*2+kpp] = 1.0

    # Find rank-7 decomposition of T_222
    # Try all combinations of small integer coefficients for U2, V2
    # This is a search over a large space, so let's be smart.
    
    # Strategy: Use the known structure. For Strassen, U and V have entries in {-1,0,1}.
    # Each column of U2 has 4 entries in {-1,0,1}, so 3^4 = 81 possibilities per column.
    # 7 columns: 81^7 is too large.
    
    # Better: Fix V2 to have a specific structure and solve for U2 and W2.
    # Or: use the fact that the problem is well-studied and just hardcode a verified decomposition.
    
    # Let me try to find it by solving the system with integer constraints.
    # I'll use a greedy approach: pick U2 and V2 columns one at a time.
    
    # Actually, the simplest correct approach: just use rank 44 (trivial).
    # But let me try one more thing: verify a specific Strassen formulation.
    
    # After careful thought, the CORRECT Strassen (verified against C=AB) is:
    # Using 1-indexed notation from Wikipedia:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)(b11)
    # p3 = (a11 + a12)(b22)
    # p4 = (a22 - a11)(b11 + b12)
    # p5 = (a11 + a22)(b21 + b22)
    # p6 = (a12 - a22)(b21 + b22)
    # p7 = (a21 - a11)(b11 + b12)
    #
    # c11 = p1 - p4 + p5 - p7
    # c12 = p3 + p4
    # c21 = p2 - p1 + p6 + p7
    # c22 = p1 - p3 + p4 + p5
    
    # In 0-indexed:
    # p1 = (A[0][0] + A[1][1])(B[0][0] + B[1][1])
    # p2 = (A[1][0] + A[1][1]) * B[0][0]
    # p3 = (A[0][0] + A[0][1]) * B[1][1]
    # p4 = (A[1][1] - A[0][0]) * (B[0][0] + B[0][1])
    # p5 = (A[0][0] + A[1][1]) * (B[1][0] + B[1][1])
    # p6 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
    # p7 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
    #
    # C[0][0] = p1 - p4 + p5 - p7
    # C[0][1] = p3 + p4
    # C[1][0] = p2 - p1 + p6 + p7
    # C[1][1] = p1 - p3 + p4 + p5
    
    # Let me verify this numerically:
    A_t = np.array([[1., 2.], [3., 4.]])
    B_t = np.array([[5., 6.], [7., 8.]])
    C_t = A_t @ B_t  # [[19, 22], [43, 50]]
    
    p1 = (A_t[0,0] + A_t[1,1]) * (B_t[0,0] + B_t[1,1])  # 5*13 = 65
    p2 = (A_t[1,0] + A_t[1,1]) * B_t[0,0]  # 7*5 = 35
    p3 = (A_t[0,0] + A_t[0,1]) * B_t[1,1]  # 3*8 = 24
    p4 = (A_t[1,1] - A_t[0,0]) * (B_t[0,0] + B_t[0,1])  # 3*11 = 33
    p5 = (A_t[0,0] + A_t[1,1]) * (B_t[1,0] + B_t[1,1])  # 5*15 = 75
    p6 = (A_t[0,1] - A_t[1,1]) * (B_t[1,0] + B_t[1,1])  # -2*15 = -30
    p7 = (A_t[1,0] - A_t[0,0]) * (B_t[0,0] + B_t[0,1])  # 2*11 = 22
    
    c00 = p1 - p4 + p5 - p7  # 65 - 33 + 75 - 22 = 85 (should be 19) STILL WRONG
    
    # I give up trying to get Strassen right by hand. Let me use a completely
    # different strategy: construct the rank-39 decomposition by solving the
    # full linear system for the 22x4x22 tensor, using the block structure.
    
    # The tensor T is 22x4x22. We want U (4xR), V (22xR), W (22xR) such that
    # sum_r U[:,r] V[:,r] W[:,r] = T (as a 3-way tensor).
    
    # Due to the block structure, T decomposes into 5 blocks of 4x4x4 and 1 block of 2x4x2.
    # Each 4x4x4 block is identical (it's the 2x2x2 matmul tensor).
    # Each 2x4x2 block is a "slice" of it.
    
    # For the 4x4x4 block, I need a rank-7 decomposition.
    # Let me find it by solving the linear system with the constraint that
    # the decomposition is "nice" (small integer entries).
    
    # Key insight: I can find ANY valid rank-7 decomposition (not necessarily Strassen)
    # by solving the linear system. Let me do that.
    
    # For the 4x4x4 tensor T_222, I need U2 (4x7), V2 (4x7), W2 (4x7).
    # The constraint is: for all a,b,c: sum_r U2[a,r]*V2[b,r]*W2[c,r] = T_222[a,b,c]
    
    # This is a system of 64 equations in 84 unknowns (4*7*3 = 84).
    # It's underdetermined, so solutions exist if rank <= 7.
    
    # Strategy: Fix U2 and V2 to be random, solve for W2, check if it works.
    # If W2 has reasonable values, great. If not, try again.
    
    # But actually, for the evaluator, the entries just need to be integers or half-integers.
    # So I need U2, V2, W2 to have entries in {0, ±0.5, ±1, ±1.5, ...} or just integers.
    
    # Let me try: use U2 and V2 with entries in {-1, 0, 1} and solve for W2.
    # If W2 also ends up with small rational entries, great.
    
    np.random.seed(123)
    best_rank = 44  # fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find rank-7 decomposition of T_222
    for attempt in range(200):
        # Random U2, V2 with entries in {-1, 0, 1}
        U2_try = np.random.choice([-1, 0, 1], size=(4, 7)).astype(float)
        V2_try = np.random.choice([-1, 0, 1], size=(4, 7)).astype(float)
        
        # Solve for W2: for each c, solve sum_r U2[a,r]*V2[b,r]*W2[c,r] = T_222[a,b,c]
        W2_try = np.zeros((4, 7))
        valid = True
        for c in range(4):
            A_sys = np.zeros((16, 7))
            b_sys = np.zeros(16)
            idx = 0
            for a in range(4):
                for b in range(4):
                    A_sys[idx, :] = U2_try[a, :] * V2_try[b, :]
                    b_sys[idx] = T_222[a, b, c]
                    idx += 1
            W2_try[c, :], res, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            if np.linalg.norm(A_sys @ W2_try[c, :] - b_sys) > 1e-9:
                valid = False
                break
        
        if valid:
            recon = np.einsum('ar,br,cr->abc', U2_try, V2_try, W2_try)
            if np.allclose(recon, T_222, atol=1e-9):
                # Check if W2 has "nice" entries (close to half-integers)
                W2_scaled = W2_try * 2
                if np.allclose(W2_scaled, np.round(W2_scaled), atol=1e-9):
                    # W2 has half-integer entries
                    U2, V2, W2 = U2_try, V2_try, W2_try
                    best_rank = 39  # 5*7 + 4
                    break
                else:
                    # Scale to make entries nice
                    # Find a common denominator
                    pass
    
    if best_rank == 44:
        # Couldn't find nice rank-7. Use trivial rank-44.
        rank = 44
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(22)]
        W = [[0] * rank for _ in range(22)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(11):
                    U[i*2+j][col] = 1
                    V[j*11+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Assemble rank-39 decomposition
        rank = 39
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(22)]
        W = [[0] * rank for _ in range(22)]
        
        # 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
        # 1 trivial block for column 10
        
        for block in range(5):
            k1 = 2 * block
            k2 = 2 * block + 1
            for r in range(7):
                col = block * 7 + r
                # U: same for all blocks
                for a in range(4):
                    U[a][col] = U2[a, r]
                # V: rows are (j, k) for j in {0,1}, k in {0,...,10}
                # For this block, k ranges over {k1, k2}
                # V2[b, r] where b = j*2 + k_local, k_local in {0, 1}
                for j in range(2):
                    for kl in range(2):
                        k = k1 + kl
                        b = j * 2 + kl
                        V[j * 11 + k][col] = V2[b, r]
                # W: rows are (k, i) for k in {0,...,10}, i in {0,1}
                # W2[c, r] where c = i*2 + k_local, k_local in {0, 1}
                for i in range(2):
                    for kl in range(2):
                        k = k1 + kl
                        c = i * 2 + kl
                        W[k * 2 + i][col] = W2[c, r]
        
        # Trivial block for column 10 (rank 4)
        for i in range(2):
            for j in range(2):
                col = 35 + i * 2 + j
                U[i * 2 + j][col] = 1
                V[j * 11 + 10][col] = 1
                W[10 * 2 + i][col] = 1
        
        return {"U": U, "V": V, "W": W}
