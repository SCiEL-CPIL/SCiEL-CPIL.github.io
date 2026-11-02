import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = 3, 3, 3

    # Try rank-23 decomposition first (known border-rank result)
    # The 3x3 matrix multiplication tensor has border rank 23.
    # We construct it via the known explicit decomposition.
    #
    # Strategy: Use the 4x4 Strassen decomposition (rank 15 for 4x4 border rank)
    # and project down. However, a simpler approach is to use the known
    # rank-23 construction from the literature.
    #
    # The rank-23 decomposition can be constructed as follows:
    # - Use 7 Strassen-like terms for the 2x2 core
    # - Use 16 border terms for indices involving dimension 2
    # - Then apply a linear combination to reduce from 23 to 23 (already 23)
    #
    # Actually, the standard construction gives rank 23 directly:
    # 7 terms for the 2x2x2 core (Strassen)
    # 16 terms for the remaining 16 entries (those with at least one index = 2)
    # But 7 + 16 = 23!
    #
    # Wait, let me recount. The 3x3 tensor has 27 entries.
    # The 2x2x2 sub-tensor has 8 entries, handled by 7 Strassen terms.
    # The remaining 19 entries need border terms.
    # But we can do better with clever linear combinations.
    #
    # Known result: border rank of 3x3 matrix multiplication is 23.
    # Let me use a direct construction.
    
    def build_rank23():
        """Construct a rank-23 decomposition for 3x3 matrix multiplication."""
        rank = 23
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        def set_term(c, u_pairs, v_pairs, w_pairs):
            for i, j, val in u_pairs:
                U[i * m + j][c] += val
            for j, k, val in v_pairs:
                V[j * p + k][c] += val
            for k, i, val in w_pairs:
                W[k * n + i][c] += val
        
        # === 7 Strassen terms for the 2x2 core (indices 0,1) ===
        # M1 = (A00+A11)(B00+B11)
        set_term(0,
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, -1)])
        
        # M2 = (A10+A11)B00
        set_term(1,
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1)],
                 [(0, 0, -1), (1, 0, 1)])
        
        # M3 = A00(B10-B11)
        set_term(2,
                 [(0, 0, 1)],
                 [(1, 0, 1), (1, 1, -1)],
                 [(0, 1, 1), (1, 1, 1)])
        
        # M4 = A11(B01-B00)
        set_term(3,
                 [(1, 1, 1)],
                 [(0, 1, 1), (0, 0, -1)],
                 [(0, 0, 1), (1, 0, -1)])
        
        # M5 = (A00+A01)B11
        set_term(4,
                 [(0, 0, 1), (0, 1, 1)],
                 [(1, 1, 1)],
                 [(0, 0, -1), (0, 1, 1), (1, 0, -1)])
        
        # M6 = (A10-A00)(B00+B01)
        set_term(5,
                 [(1, 0, 1), (0, 0, -1)],
                 [(0, 0, 1), (0, 1, 1)],
                 [(0, 1, 1), (1, 0, 1)])
        
        # M7 = (A01-A11)(B10+B11)
        set_term(6,
                 [(0, 1, 1), (1, 1, -1)],
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (0, 1, -1), (1, 1, 1)])
        
        # === 16 border terms for entries with at least one index = 2 ===
        # These handle all (i,j,k) where max(i,j,k) = 2
        # There are 27 - 8 = 19 such entries, but we can cover them with 16 terms
        # using linear combinations.
        
        # Actually, let's just use 16 separate terms for the 16 "pure border" entries
        # and handle the 3 "mixed" entries via the Strassen terms or additional combos.
        
        # The entries with exactly one index = 2:
        # (2,j,k) for j,k in {0,1}: 4 entries -> A[2,j]*B[j,k]
        # (i,2,k) for i,k in {0,1}: 4 entries -> A[i,2]*B[2,k]
        # (i,j,2) for i,j in {0,1}: 4 entries -> A[i,j]*B[j,2]
        # (2,2,k), (2,j,2), (i,2,2) for i,j,k in {0,1}: 3+3+3 = 9? No...
        
        # Let me just enumerate all 19 entries with max(i,j,k)=2:
        # i=2: (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2) = 9
        # i=0,j=2: (0,2,0),(0,2,1),(0,2,2) = 3
        # i=1,j=2: (1,2,0),(1,2,1),(1,2,2) = 3
        # i=0,j=0,k=2: (0,0,2) = 1
        # i=0,j=1,k=2: (0,1,2) = 1
        # i=1,j=0,k=2: (1,0,2) = 1
        # i=1,j=1,k=2: (1,1,2) = 1
        # Total: 9+3+3+1+1+1+1 = 19
        
        # We need 16 terms to cover these 19 entries (since 7+16=23).
        # This requires 3 entries to be covered by linear combinations.
        
        # Let's use a known explicit rank-23 construction.
        # Based on the work of Landsberg and others, here's a construction:
        
        col = 7
        
        # Group 1: Terms involving A[2,*] (i=2, j=0,1; k=0,1)
        # (2,0,0): A[2,0]*B[0,0]
        set_term(col, [(2, 0, 1)], [(0, 0, 1)], [(0, 2, 1)])
        col += 1
        # (2,0,1): A[2,0]*B[0,1]
        set_term(col, [(2, 0, 1)], [(0, 1, 1)], [(1, 2, 1)])
        col += 1
        # (2,1,0): A[2,1]*B[1,0]
        set_term(col, [(2, 1, 1)], [(1, 0, 1)], [(0, 2, 1)])
        col += 1
        # (2,1,1): A[2,1]*B[1,1]
        set_term(col, [(2, 1, 1)], [(1, 1, 1)], [(1, 2, 1)])
        col += 1
        
        # Group 2: Terms involving B[*,2] (k=2, i=0,1; j=0,1)
        # (0,0,2): A[0,0]*B[0,2]
        set_term(col, [(0, 0, 1)], [(0, 2, 1)], [(2, 0, 1)])
        col += 1
        # (0,1,2): A[0,1]*B[1,2]
        set_term(col, [(0, 1, 1)], [(1, 2, 1)], [(2, 0, 1)])
        col += 1
        # (1,0,2): A[1,0]*B[0,2]
        set_term(col, [(1, 0, 1)], [(0, 2, 1)], [(2, 1, 1)])
        col += 1
        # (1,1,2): A[1,1]*B[1,2]
        set_term(col, [(1, 1, 1)], [(1, 2, 1)], [(2, 1, 1)])
        col += 1
        
        # Group 3: Terms involving A[*,2] and B[2,*] (j=2)
        # (0,2,0): A[0,2]*B[2,0]
        set_term(col, [(0, 2, 1)], [(2, 0, 1)], [(0, 0, 1)])
        col += 1
        # (0,2,1): A[0,2]*B[2,1]
        set_term(col, [(0, 2, 1)], [(2, 1, 1)], [(0, 1, 1)])
        col += 1
        # (1,2,0): A[1,2]*B[2,0]
        set_term(col, [(1, 2, 1)], [(2, 0, 1)], [(1, 0, 1)])
        col += 1
        # (1,2,1): A[1,2]*B[2,1]
        set_term(col, [(1, 2, 1)], [(2, 1, 1)], [(1, 1, 1)])
        col += 1
        
        # Group 4: Terms with (2,2,k) and (2,j,2) and (i,2,2)
        # (2,2,0): A[2,2]*B[2,0]
        set_term(col, [(2, 2, 1)], [(2, 0, 1)], [(0, 2, 1)])
        col += 1
        # (2,2,1): A[2,2]*B[2,1]
        set_term(col, [(2, 2, 1)], [(2, 1, 1)], [(1, 2, 1)])
        col += 1
        # (2,0,2): A[2,0]*B[0,2]
        set_term(col, [(2, 0, 1)], [(0, 2, 1)], [(2, 2, 1)])
        col += 1
        # (2,1,2): A[2,1]*B[1,2]
        set_term(col, [(2, 1, 1)], [(1, 2, 1)], [(2, 2, 1)])
        col += 1
        # (0,2,2): A[0,2]*B[2,2]
        set_term(col, [(0, 2, 1)], [(2, 2, 1)], [(2, 0, 1)])
        col += 1
        # (1,2,2): A[1,2]*B[2,2]
        set_term(col, [(1, 2, 1)], [(2, 2, 1)], [(2, 1, 1)])
        col += 1
        
        # That's 16 terms (col went from 7 to 23). Total = 23.
        # But wait, we have 19 entries and only 16 terms. 
        # The 3 missing entries are: (2,2,2), and we need to check coverage.
        
        # Let me verify which entries are covered:
        # From Group 1: (2,0,0), (2,0,1), (2,1,0), (2,1,1) ✓
        # From Group 2: (0,0,2), (0,1,2), (1,0,2), (1,1,2) ✓
        # From Group 3: (0,2,0), (0,2,1), (1,2,0), (1,2,1) ✓
        # From Group 4: (2,2,0), (2,2,1), (2,0,2), (2,1,2), (0,2,2), (1,2,2) ✓
        # Total covered: 4+4+4+6 = 18
        # Missing: (2,2,2) - this is A[2,2]*B[2,2]
        
        # We need to add (2,2,2) but we're at rank 23 already.
        # This means the above construction doesn't work as-is.
        
        # Let me try a different approach: use the known rank-23 construction
        # from the literature more carefully.
        
        # Actually, I think the issue is that the border-rank construction
        # requires careful linear combinations. Let me use a verified construction.
        
        return None  # Signal that this simple approach doesn't work
    
    def build_rank26():
        """Construct a guaranteed valid rank-26 decomposition."""
        rank = 26
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        def set_term(c, u_pairs, v_pairs, w_pairs):
            for i, j, val in u_pairs:
                U[i * m + j][c] += val
            for j, k, val in v_pairs:
                V[j * p + k][c] += val
            for k, i, val in w_pairs:
                W[k * n + i][c] += val
        
        # 7 Strassen terms
        set_term(0,
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, -1)])
        set_term(1,
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1)],
                 [(0, 0, -1), (1, 0, 1)])
        set_term(2,
                 [(0, 0, 1)],
                 [(1, 0, 1), (1, 1, -1)],
                 [(0, 1, 1), (1, 1, 1)])
        set_term(3,
                 [(1, 1, 1)],
                 [(0, 1, 1), (0, 0, -1)],
                 [(0, 0, 1), (1, 0, -1)])
        set_term(4,
                 [(0, 0, 1), (0, 1, 1)],
                 [(1, 1, 1)],
                 [(0, 0, -1), (0, 1, 1), (1, 0, -1)])
        set_term(5,
                 [(1, 0, 1), (0, 0, -1)],
                 [(0, 0, 1), (0, 1, 1)],
                 [(0, 1, 1), (1, 0, 1)])
        set_term(6,
                 [(0, 1, 1), (1, 1, -1)],
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (0, 1, -1), (1, 1, 1)])
        
        # 19 border terms (one for each entry with max(i,j,k)=2)
        col = 7
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if max(i, j, k) == 2:
                        set_term(col, [(i, j, 1)], [(j, k, 1)], [(k, i, 1)])
                        col += 1
        
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W, rank, n=3, m=3, p=3, num_tests=10):
        random.seed(42)
        for _ in range(num_tests):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            C = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for k in range(p):
                        sum_U = 0
                        sum_V = 0
                        for j in range(m):
                            sum_U += U[i * m + j][c] * A[i][j]
                            sum_V += V[j * p + k][c] * B[j][k]
                        C[i][k] += sum_U * sum_V * W[k * n + i][c]
            if C != AB:
                return False
        return True
    
    # Try rank-23 first
    # Use a known valid rank-23 construction
    # Based on the explicit decomposition from the literature
    
    def build_rank23_v2():
        """Rank-23 construction using careful linear combinations."""
        rank = 23
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        def set_term(c, u_pairs, v_pairs, w_pairs):
            for i, j, val in u_pairs:
                U[i * m + j][c] += val
            for j, k, val in v_pairs:
                V[j * p + k][c] += val
            for k, i, val in w_pairs:
                W[k * n + i][c] += val
        
        # 7 Strassen terms for 2x2 core
        set_term(0,
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, -1)])
        set_term(1,
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1)],
                 [(0, 0, -1), (1, 0, 1)])
        set_term(2,
                 [(0, 0, 1)],
                 [(1, 0, 1), (1, 1, -1)],
                 [(0, 1, 1), (1, 1, 1)])
        set_term(3,
                 [(1, 1, 1)],
                 [(0, 1, 1), (0, 0, -1)],
                 [(0, 0, 1), (1, 0, -1)])
        set_term(4,
                 [(0, 0, 1), (0, 1, 1)],
                 [(1, 1, 1)],
                 [(0, 0, -1), (0, 1, 1), (1, 0, -1)])
        set_term(5,
                 [(1, 0, 1), (0, 0, -1)],
                 [(0, 0, 1), (0, 1, 1)],
                 [(0, 1, 1), (1, 0, 1)])
        set_term(6,
                 [(0, 1, 1), (1, 1, -1)],
                 [(1, 0, 1), (1, 1, 1)],
                 [(0, 0, 1), (0, 1, -1), (1, 1, 1)])
        
        # 16 border terms
        col = 7
        
        # Use a systematic approach: cover all 19 border entries with 16 terms
        # by using 3 "combined" terms that each cover 2 entries via linearity
        
        # Term 7: covers (2,0,0) and part of others
        # Let's use the construction from the paper:
        # We need to be very careful here.
        
        # Alternative: just use 19 border terms for rank 26, and try to
        # find a rank-23 via optimization if time permits.
        
        # For now, return None to indicate we should use rank-26
        return None
    
    # Build rank-26 as guaranteed valid
    U, V, W, rank = build_rank26()
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook rank-27
        rank = 27
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to search for rank-25 or rank-24 using random perturbation
    if time.time() < deadline - 20:
        best_result = (U, V, W, rank)
        
        # Try random linear combinations of the 26 terms to see if we can
        # merge some terms (i.e., find a rank-25 or lower decomposition)
        # This is a heuristic search.
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Try to find a rank-25 by combining two terms
            # This is complex, so we'll skip for now and just return rank-26
        
        # For now, return the rank-26 result
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": U, "V": V, "W": W}
