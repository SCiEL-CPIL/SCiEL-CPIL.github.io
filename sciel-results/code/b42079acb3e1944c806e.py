import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The rank 56 solution using block Strassen
    # Structure:
    # Split P=8 into 4 blocks of size 2? No, the current code uses:
    # g in range(4): 4 blocks of width 2 in output (p=8 -> 4*2)
    # b in range(2): 2 blocks of width 2 in middle (m=4 -> 2*2)
    # s in range(7): Strassen steps
    
    # This corresponds to:
    # R(2, 4, 8) <= 4 (output blocks) * R(2, 2, 2) ... wait.
    # The current code has rank = 4 * 2 * 7 = 56.
    # This is 4 (p-blocks) * 2 (m-blocks) * 7 (Strassen) ?
    # No, the loops are:
    # for g in 0..3: (4 groups, covering p=8 as 4*2)
    #   for b in 0..1: (2 blocks, covering m=4 as 2*2)
    #     for s in 0..6: (Strassen)
    # This treats each (g, b) pair as a 2x2x2 multiplication?
    # A is 2x4. Split A into A1 (2x2), A2 (2x2).
    # B is 4x8. Split B into 4 blocks B0..B3, each 2x2?
    # No, B is 4x8.
    # The index b selects columns of A and rows of B?
    # A[i, b*2+...]
    # B[b*2+..., 2*g+...]
    # This is exactly 8 independent 2x2x2 multiplications?
    # Wait, A is shared.
    # If we do 8 independent 2x2x2, rank is 8*7=56.
    # But A is the same for all g!
    # The current code DOES share A?
    # U[a_idx][r] = X_forms[s][i * 2 + jp]
    # a_idx = i * m + b * 2 + jp
    # The index r depends on g, b, s.
    # So for each g, b, s, we have a unique rank term.
    # U has entries for all a_idx.
    # This means the linear combination of A is repeated for each g?
    # No, U is a matrix of size (n*m) x rank.
    # If U[a_idx][r] is non-zero for specific r, it means that rank term uses that element of A.
    # Since r iterates over g, b, s, each (g,b,s) creates a new column in U, V, W.
    # This effectively computes 8 separate 2x2x2 multiplications, but packs them.
    # Does it exploit the sharing of A?
    # In a tensor decomposition, the rank is the number of terms.
    # If we compute $C_{g} = A_{b} B_{g,b}$?
    # The structure is $C = \sum_{b} A_b B_b$.
    # Here $A_b$ is $2 \times 2$ (cols b*2, b*2+1 of A).
    # $B_b$ is $2 \times 8$ (rows b*2, b*2+1 of B).
    # So we have two $2 \times 2 \times 8$ multiplications?
    # $R(2, 2, 8) \times 2$?
    # If $R(2, 2, 8) = 28$, then $2 \times 28 = 56$.
    # And $R(2, 2, 8) \le 4 \times R(2, 2, 2) = 28$.
    # So yes, it is 2 blocks of $R(2, 2, 8)$.
    # And $R(2, 2, 8)$ is computed as 4 blocks of $R(2, 2, 2)$.
    # This is the standard Strassen decomposition.
    
    X_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    Y_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    num_groups = 4  # p blocks
    num_ablocks = 2  # m blocks
    num_strassen = 7
    
    rank = num_groups * num_ablocks * num_strassen # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
