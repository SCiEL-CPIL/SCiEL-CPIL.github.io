"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for name, M in (("U", U), ("V", V), ("W", W)):
            for row in M:
                for val in row:
                    if val != int(val) and 2 * val != int(2 * val):
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_block_decomp():
        """Build a rank-82 decomposition using Strassen on 2x2x2 blocks."""
        rank_total = 82
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0

        for b in range(2):
            j0 = 2 * b

            # --- Strassen 2x2x2 for A[0:2, j0:j0+2] * B[j0:j0+2, 0:2] -> C[0:2, 0:2] ---
            # 7 rank slots for Strassen
            # M1 = (A00+A11)*(B00+B11)
            U[0 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M2 = (A10+A11)*B00
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M3 = A00*(B11-B00)
            U[0 * m + (j0 + 0)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 0][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M4 = A11*(B01-B10)
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 1][col] = 1
            V[(j0 + 1) * p + 0][col] = -1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M5 = (A00+A01)*B10
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = -1  # C[0,0]
            W[0 * n + 1][col] = 1   # C[0,1]
            col += 1

            # M6 = (A10+A11)*(B11-B01)
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 1][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            W[1 * n + 1][col] = 1  # C[1,1]
            col += 1

            # M7 = (A00-A01)*(B00+B10)
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = -1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            col += 1

            # --- Direct 2x2x5 for A[0:2, j0:j0+2] * B[j0:j0+2, 2:7] -> C[0:2, 2:7] ---
            # 20 rank slots
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        U[i * m + (j0 + j)][col] = 1
                        V[(j0 + j) * p + (2 + k)][col] = 1
                        W[(2 + k) * n + i][col] = 1
                        col += 1

            # --- Direct 1x2x7 for A[2, j0:j0+2] * B[j0:j0+2, 0:7] -> C[0:7, 2] ---
            # 14 rank slots
            for j in range(2):
                for k in range(7):
                    U[2 * m + (j0 + j)][col] = 1
                    V[(j0 + j) * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1

        assert col == rank_total, f"col={col}, rank_total={rank_total}"
        return {"U": U, "V": V, "W": W}

    def build_3x2x7_decomp():
        """
        Build a rank-42 decomposition for 3x2x7 matrix multiplication.
        Uses Strassen for the 2x2x2 sub-block (rows 0-1, col 0, output cols 0-1)
        and direct for the rest.
        This is a helper to potentially assemble a better global decomposition.
        """
        # For 3x2x7: A is 3x2, B is 2x7, C is 3x7
        # Split A into A_top (2x2) and A_bottom (1x2)
        # Split B into B_left (2x2) and B_right (2x5)
        # C_top = A_top * B (2x7), C_bottom = A_bottom * B (1x7)
        
        # C_top_left (2x2) via Strassen: rank 7
        # C_top_right (2x5) direct: rank 10
        # C_bottom (1x7) direct: rank 14
        # Total: 7 + 10 + 14 = 31
        
        rank_total = 31
        U = [[0] * rank_total for _ in range(3 * 2)]  # 6 rows
        V = [[0] * rank_total for _ in range(2 * 7)]  # 14 rows
        W = [[0] * rank_total for _ in range(7 * 3)]  # 21 rows
        
        col = 0
        # Strassen for A[0:2, :] * B[:, 0:2] -> C[0:2, 0:2]
        # A indices: 0,1 (i=0,1), j=0,1
        # B indices: j=0,1; k=0,1
        # C indices: i=0,1; k=0,1
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[0][col] = 1; V[6][col] = 1
        W[0][col] = 1; W[3][col] = 1
        col += 1
        # M2 = (A10+A11)*B00
        U[2][col] = 1; U[3][col] = 1
        V[0][col] = 1
        W[1][col] = 1
        col += 1
        # M3 = A00*(B11-B00)
        U[0][col] = 1
        V[6][col] = 1; V[0][col] = -1
        W[3][col] = 1
        col += 1
        # M4 = A11*(B01-B10)
        U[3][col] = 1
        V[1][col] = 1; V[7][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        # M5 = (A00+A01)*B10
        U[0][col] = 1; U[1][col] = 1
        V[7][col] = 1
        W[0][col] = -1; W[1][col] = 1
        col += 1
        # M6 = (A10+A11)*(B11-B01)
        U[2][col] = 1; U[3][col] = 1
        V[6][col] = 1; V[1][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        # M7 = (A00-A01)*(B00+B10)
        U[0][col] = 1; U[1][col] = -1
        V[0][col] = 1; V[7][col] = 1
        W[0][col] = 1
        col += 1
        
        # Direct for A[0:2, :] * B[:, 2:7] -> C[0:2, 2:7]
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U[i * 2 + j][col] = 1
                    V[j * 7 + (2 + k)][col] = 1
                    W[(2 + k) * 3 + i][col] = 1
                    col += 1
                    
        # Direct for A[2, :] * B[:, :] -> C[2, :]
        for j in range(2):
            for k in range(7):
                U[4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + 2][col] = 1
                col += 1
                
        assert col == rank_total
        return {"U": U, "V": V, "W": W}

    # Try the block decomposition first
    result = build_block_decomp()
    if verify(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
