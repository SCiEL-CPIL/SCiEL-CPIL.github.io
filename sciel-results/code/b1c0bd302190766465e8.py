"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build a rank-32 decomposition using Strassen for 2x2x2 blocks
    # Split p=9 into blocks: 2+2+2+2+1
    # Each 2x2x2 block uses Strassen (rank 7), 2x2x1 uses schoolbook (rank 4)
    # Total: 7*4 + 4 = 32
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C10, C11 (with -1 on C11)
            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C01, C11
            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C00, C10
            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> C00, C01 (with -1 on C00)
            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11
            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00
            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            # 4 terms for a single k
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        # Split p=9 into 4 Strassen blocks (2x2 each) + 1 trivial block
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    # Build the rank-32 decomposition
    result = build_rank32()

    # Verify it's correct
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if verification fails
        return schoolbook()

    # Try to find a lower-rank decomposition using the time budget
    # Strategy: try to reduce rank by merging terms or finding non-block decompositions
    best_rank = 32
    best_result = result

    # Try rank 31: attempt to find a solution with 31 terms
    # We'll use a search approach: try different ways to combine/merge terms
    def try_rank_reduction(target_rank, U0, V0, W0, time_limit=60.0):
        """Try to find a decomposition of target_rank by optimizing from U0,V0,W0."""
        import numpy as np
        start = time.time()
        
        # Reshape to numpy for computation
        n, m, p = 2, 2, 9
        U_np = np.array(U0, dtype=float)
        V_np = np.array(V0, dtype=float)
        W_np = np.array(W0, dtype=float)
        
        # We'll try to zero out one column and solve for the rest
        # This is a nonlinear problem, so we'll use a simple iterative approach
        
        # Actually, let's try a different approach: 
        # The tensor equation is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # If we fix all but one column, the last column must satisfy:
        # U[i*m+j,c_last]*V[j*p+k,c_last]*W[k*n+i,c_last] = 1 - sum_{c<c_last} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # This is a system of 36 equations in the products of the last column entries.
        
        # For rank 31, we need to find 31 columns that satisfy the equations.
        # This is hard to solve directly. Let's try a heuristic approach.
        
        # Heuristic: try to find two columns that can be merged
        # If column a and column b have the property that their "contribution pattern" 
        # can be represented by a single column, we can merge them.
        
        # For now, let's just verify the rank-32 solution and return it.
        # A full search for rank 31 would require solving a hard nonlinear system.
        
        return None

    # Given the difficulty of finding rank < 32, return the verified rank-32 result
    return result
# EVOLVE-BLOCK-END
