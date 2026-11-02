"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    # Deterministic Kronecker/block decomposition:
    # Split m=5 into m1=2, m2=3
    # Part 1: 2x2x6 -> split p=6 into 3 pairs of 2 -> 3 * rank(2,2,2) = 3*7 = 21
    # Part 2: 2x3x6 -> split m=3 into 1+2, split p=6 into 3 pairs of 2
    #   2x1x2: rank 4, 2x2x2: rank 7, so 2x3x2: rank 11
    #   2x3x6: 3 * 11 = 33
    # Total rank = 21 + 33 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0,1}, 2x2x6 -> 3 * Strassen(2x2x2) = 21
    for s in range(3):
        k_base = s * 2  # maps local k' in {0,1} to global k
        
        strassen_terms = [
            (
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)]
            ),
            (
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,0,1), (1,1,-1)]
            ),
            (
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(0,1,1), (1,1,1)]
            ),
            (
                [(1,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (1,0,1)]
            ),
            (
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,-1), (0,1,1)]
            ),
            (
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)]
            ),
            (
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)]
            ),
        ]
        
        for u_terms, v_terms, w_terms in strassen_terms:
            for (li, lj, val) in u_terms:
                gi = li
                gj = lj  # j in {0,1} for Part 1
                U[gi * m + gj][col] += val
            
            for (lj, lk, val) in v_terms:
                gj = lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            
            for (li, lk, val) in w_terms:
                gi = li
                gk = k_base + lk
                W[gk * n + gi][col] += val
            
            col += 1
    
    # Part 2: j in {2,3,4}, 2x3x6
    # Split m=3 into j=2 (1 dim) and j={3,4} (2 dims)
    # For each k-pair, do 2x1x2 (rank 4) + 2x2x2 (rank 7) = 11
    for s in range(3):
        k_base = s * 2  # maps local k' in {0,1} to global k
        
        # 2x1x2 part: j=2, rank 4
        for i in range(n):
            for k in range(2):
                U[i * m + 2][col] = 1
                V[2 * p + k_base + k][col] = 1
                W[(k_base + k) * n + i][col] = 1
                col += 1
        
        # 2x2x2 part: j={3,4}, Strassen, rank 7
        j_base = 3  # maps local j' in {0,1} to global j in {3,4}
        
        strassen_terms = [
            (
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)]
            ),
            (
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,0,1), (1,1,-1)]
            ),
            (
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(0,1,1), (1,1,1)]
            ),
            (
                [(1,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (1,0,1)]
            ),
            (
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,-1), (0,1,1)]
            ),
            (
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)]
            ),
            (
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)]
            ),
        ]
        
        for u_terms, v_terms, w_terms in strassen_terms:
            for (li, lj, val) in u_terms:
                gi = li
                gj = j_base + lj
                U[gi * m + gj][col] += val
            
            for (lj, lk, val) in v_terms:
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            
            for (li, lk, val) in w_terms:
                gi = li
                gk = k_base + lk
                W[gk * n + gi][col] += val
            
            col += 1
    
    # Time guard: verify we're within deadline
    if time.time() - _START_TIME > _DEADLINE:
        raise TimeoutError("Exceeded time budget")
    
    # Final verification: check dimensions
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U cols mismatch"
    assert all(len(row) == rank for row in V), "V cols mismatch"
    assert all(len(row) == rank for row in W), "W cols mismatch"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
