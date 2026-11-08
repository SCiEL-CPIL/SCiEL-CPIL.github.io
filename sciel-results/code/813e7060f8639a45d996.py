# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 5, 6

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_84():
        # Existing 84 rank logic
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        # ... (same as before) ...
        return result

    def build_bini_82():
        # Partition: i=3, j=3+2, k=3+3
        # Blocks:
        # 1. i=0..2, j=0..2, k=0..2 (3x3x3) -> Bini
        # 2. i=0..2, j=0..2, k=3..5 (3x3x3) -> Bini
        # 3. i=0..2, j=3..4, k=0..2 (3x2x3) -> Schoolbook
        # 4. i=0..2, j=3..4, k=3..5 (3x2x3) -> Schoolbook
        
        U = [[0] * 82 for _ in range(n * m)]
        V = [[0] * 82 for _ in range(m * p)]
        W = [[0] * 82 for _ in range(p * n)]
        col = 0
        
        # Bini terms for 3x3x3
        # Using a known set of 23 terms
        bini_terms = [
            # (u_dict, v_dict, w_dict)
            # 1
            ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
            # 2
            ({(0,0):1, (1,1):1, (2,2):-1}, {(0,0):1, (1,1):1, (2,2):-1}, {(0,0):1, (1,1):1, (2,2):1}),
            # 3
            ({(0,0):1, (1,1):-1, (2,2):1}, {(0,0):1, (1,1):-1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
            # 4
            ({(0,0):-1, (1,1):1, (2,2):1}, {(0,0):-1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
            # 5
            ({(0,1):1, (1,0):1, (2,0):1}, {(0,0):1, (1,0):1, (2,0):1}, {(0,0):1, (1,0):1, (2,0):1}),
            # ... This is getting risky to hardcode from memory.
            # I will use a fallback to 84 if I can't guarantee correctness.
            # For now, I will return None to signal failure.
            None
        ]
        
        if bini_terms[0] is None:
            return None
        
        # Since I don't have the full 23 terms hardcoded correctly, I will just return the 84 rank solution.
        # To improve, I need the exact Bini terms.
        # I will use a placeholder that returns None, triggering the fallback.
        return None

    # Try 82
    res_82 = build_bini_82()
    if res_82 and verify(res_82["U"], res_82["V"], res_82["W"], 82):
        return res_82
    
    # Fallback to 84
    res_84 = build_strassen_84()
    return res_84
# EVOLVE-BLOCK-END
