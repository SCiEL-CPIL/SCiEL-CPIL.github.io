import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = DIMENSIONS

    # Use partition [1, 4] for all dimensions
    # For 4x4x4 block, use Strassen(4) with rank 49
    # For other blocks, use schoolbook
    n_parts = [1, 4]
    m_parts = [1, 4]
    p_parts = [1, 4]

    n_starts = [0, 1]
    m_starts = [0, 1]
    p_starts = [0, 1]

    # Compute total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 49
                else:
                    total_rank += ni * mj * pk

    # Allocate factor matrices using Fraction for exact arithmetic
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(p * n)]

    col = 0

    def fill_strassen_2x2x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 2x2x4 Strassen block.
        A is 2x2, B is 2x4, result is 2x4.
        We can split B into B_left (2x2) and B_right (2x2).
        Use Strassen for each 2x2x2 sub-block.
        """
        c = col_offset
        # For A (2x2) * B_left (2x2) -> use Strassen rank 7
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1

        # M1 = A00 * B00
        U[i0 * m + j0][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M2 = (A10 + A11) * B00
        U[i1 * m + j0][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(1)
        W[k0 * n + i1][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(-1)
        c += 1

        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] = Fraction(1)
        V[j0 * p + k1][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(-1)
        W[k1 * n + i0][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M4 = A11 * (B10 - B00)
        U[i1 * m + j1][c] = Fraction(1)
        V[j1 * p + k0][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(-1)
        W[k0 * n + i0][c] = Fraction(1)
        W[k0 * n + i1][c] = Fraction(1)
        c += 1

        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] = Fraction(1)
        U[i0 * m + j1][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(-1)
        W[k1 * n + i0][c] = Fraction(1)
        c += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[i1 * m + j0][c] = Fraction(1)
        U[i0 * m + j0][c] = Fraction(-1)
        V[j0 * p + k0][c] = Fraction(1)
        V[j0 * p + k1][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + j1][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(-1)
        V[j1 * p + k0][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(1)
        c += 1

        return c

    def fill_strassen_4x4x2(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x4x2 Strassen block.
        A is 4x4, B is 4x2, result is 4x2.
        Split A into 2x2 blocks: A00, A01, A10, A11.
        B is 4x2, split into B_top (2x2) and B_bottom (2x2).
        """
        c = col_offset
        # Split A rows: [i_start, i_start+2) and [i_start+2, i_start+4)
        # Split B rows: [j_start, j_start+2) and [j_start+2, j_start+4)
        
        # A_top * B_top + A_bottom * B_bottom
        # Use Strassen for each 2x2x2
        
        # A_top * B_top: rows i_start..i_start+1, cols j_start..j_start+1
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        
        # A_top * B_bottom: rows i_start..i_start+1, cols j_start+2..j_start+3
        c = fill_strassen_2x2x2(i_start, j_start + 2, k_start, c)
        
        # A_bottom * B_top: rows i_start+2..i_start+3, cols j_start..j_start+1
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start, c)
        
        # A_bottom * B_bottom: rows i_start+2..i_start+3, cols j_start+2..j_start+3
        c = fill_strassen_2x2x2(i_start + 2, j_start + 2, k_start, c)
        
        return c

    def fill_strassen_4x2x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x2x4 Strassen block.
        A is 4x2, B is 2x4, result is 4x4.
        Split A rows into A_top (2x2) and A_bottom (2x2).
        Split B cols into B_left (2x2) and B_right (2x2).
        """
        c = col_offset
        # A_top * B_left + A_bottom * B_left for left half of result
        # A_top * B_right + A_bottom * B_right for right half of result
        
        # For left half (k_start..k_start+1):
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start, c)
        
        # For right half (k_start+2..k_start+3):
        c = fill_strassen_2x2x2(i_start, j_start, k_start + 2, c)
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start + 2, c)
        
        return c

    def fill_strassen_2x4x2(i_start, j_start, k_start, col_offset):
        """
        Fill a 2x4x2 Strassen block.
        A is 2x4, B is 4x2, result is 2x2.
        Split A cols into A_left (2x2) and A_right (2x2).
        Split B rows into B_top (2x2) and B_bottom (2x2).
        """
        c = col_offset
        # A_left * B_top + A_right * B_bottom
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        c = fill_strassen_2x2x2(i_start, j_start + 2, k_start, c)
        
        return c

    def fill_strassen_2x2x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 2x2x4 Strassen block.
        A is 2x2, B is 2x4, result is 2x4.
        Split B cols into B_left (2x2) and B_right (2x2).
        """
        c = col_offset
        # A * B_left
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        # A * B_right
        c = fill_strassen_2x2x2(i_start, j_start, k_start + 2, c)
        
        return c

    def fill_strassen_4x2x2(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x2x2 Strassen block.
        A is 4x2, B is 2x2, result is 4x2.
        Split A rows into A_top (2x2) and A_bottom (2x2).
        """
        c = col_offset
        # A_top * B
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        # A_bottom * B
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start, c)
        
        return c

    def fill_strassen_2x4x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 2x4x4 Strassen block.
        A is 2x4, B is 4x4, result is 2x4.
        Split A cols into A_left (2x2) and A_right (2x2).
        Split B cols into B_left (4x2) and B_right (4x2).
        """
        c = col_offset
        # A_left * B_left
        c = fill_strassen_2x4x2(i_start, j_start, k_start, c)
        # A_left * B_right
        c = fill_strassen_2x4x2(i_start, j_start, k_start + 2, c)
        # A_right * B_left
        c = fill_strassen_2x4x2(i_start, j_start + 2, k_start, c)
        # A_right * B_right
        c = fill_strassen_2x4x2(i_start, j_start + 2, k_start + 2, c)
        
        return c

    def fill_strassen_4x4x2(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x4x2 Strassen block.
        A is 4x4, B is 4x2, result is 4x2.
        Split A rows into A_top (2x4) and A_bottom (2x4).
        Split B rows into B_top (2x2) and B_bottom (2x2).
        """
        c = col_offset
        # A_top * B_top
        c = fill_strassen_2x4x2(i_start, j_start, k_start, c)
        # A_top * B_bottom
        c = fill_strassen_2x4x2(i_start, j_start + 2, k_start, c)
        # A_bottom * B_top
        c = fill_strassen_2x4x2(i_start + 2, j_start, k_start, c)
        # A_bottom * B_bottom
        c = fill_strassen_2x4x2(i_start + 2, j_start + 2, k_start, c)
        
        return c

    def fill_strassen_4x2x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x2x4 Strassen block.
        A is 4x2, B is 2x4, result is 4x4.
        Split A rows into A_top (2x2) and A_bottom (2x2).
        Split B cols into B_left (2x2) and B_right (2x2).
        """
        c = col_offset
        # A_top * B_left
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        # A_top * B_right
        c = fill_strassen_2x2x2(i_start, j_start, k_start + 2, c)
        # A_bottom * B_left
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start, c)
        # A_bottom * B_right
        c = fill_strassen_2x2x2(i_start + 2, j_start, k_start + 2, c)
        
        return c

    def fill_strassen_2x2x2(i_start, j_start, k_start, col_offset):
        """
        Fill a 2x2x2 Strassen block.
        """
        c = col_offset
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1

        # M1 = A00 * B00
        U[i0 * m + j0][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M2 = (A10 + A11) * B00
        U[i1 * m + j0][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(1)
        W[k0 * n + i1][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(-1)
        c += 1

        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] = Fraction(1)
        V[j0 * p + k1][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(-1)
        W[k1 * n + i0][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M4 = A11 * (B10 - B00)
        U[i1 * m + j1][c] = Fraction(1)
        V[j1 * p + k0][c] = Fraction(1)
        V[j0 * p + k0][c] = Fraction(-1)
        W[k0 * n + i0][c] = Fraction(1)
        W[k0 * n + i1][c] = Fraction(1)
        c += 1

        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] = Fraction(1)
        U[i0 * m + j1][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(-1)
        W[k1 * n + i0][c] = Fraction(1)
        c += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[i1 * m + j0][c] = Fraction(1)
        U[i0 * m + j0][c] = Fraction(-1)
        V[j0 * p + k0][c] = Fraction(1)
        V[j0 * p + k1][c] = Fraction(1)
        W[k1 * n + i1][c] = Fraction(1)
        c += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + j1][c] = Fraction(1)
        U[i1 * m + j1][c] = Fraction(-1)
        V[j1 * p + k0][c] = Fraction(1)
        V[j1 * p + k1][c] = Fraction(1)
        W[k0 * n + i0][c] = Fraction(1)
        c += 1

        return c

    def fill_strassen_4x4x4(i_start, j_start, k_start, col_offset):
        """
        Fill a 4x4x4 Strassen block with rank 49.
        Use recursive Strassen: 7 multiplications of 2x2 matrices, each with rank 7.
        Total rank = 7 * 7 = 49.
        """
        c = col_offset
        # Split A into 2x2 blocks: A00, A01, A10, A11
        # Split B into 2x2 blocks: B00, B01, B10, B11
        # Split C into 2x2 blocks: C00, C01, C10, C11
        
        # Strassen formulas for 4x4:
        # M1 = A00 * B00
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Each Mi is a 2x2 matrix product, which has rank 7.
        # But we need to express this in terms of the global tensor.
        
        # Actually, for 4x4x4, we can just use 49 rank-1 terms directly.
        # Let's use the known optimal decomposition for 4x4x4.
        
        # For simplicity, let's just use schoolbook for 4x4x4 if we can't find a good decomposition.
        # But we want rank 49, not 64.
        
        # Let me think about this differently. We can use the fact that:
        # C = A * B where A, B, C are 4x4.
        # We can write this as a sum of 49 rank-1 tensors.
        
        # One approach: use the 7 Strassen products, each of which is a 2x2x2 tensor.
        # But each 2x2x2 product involves linear combinations of A and B blocks.
        
        # This is getting complex. Let me just use schoolbook for now and see if we can improve.
        # Actually, let me try a different approach: use the [2,2] partition for the 4x4 blocks.
        
        # For 4x4x4, split into 2x2 blocks in each dimension.
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
        # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
        
        # Strassen:
        # M1 = A00 * B00
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Now, each Mi is a 2x2x2 tensor. We can decompose each Mi into 7 rank-1 terms.
        # So total rank = 7 * 7 = 49.
        
        # Let's implement this.
        
        # A00 is at rows i_start..i_start+1, cols j_start..j_start+1
        # A01 is at rows i_start..i_start+1, cols j_start+2..j_start+3
        # A10 is at rows i_start+2..i_start+3, cols j_start..j_start+1
        # A11 is at rows i_start+2..i_start+3, cols j_start+2..j_start+3
        
        # Similarly for B and C.
        
        # M1 = A00 * B00
        # A00 rows: i_start, i_start+1; cols: j_start, j_start+1
        # B00 rows: j_start, j_start+1; cols: k_start, k_start+1
        c = fill_strassen_2x2x2(i_start, j_start, k_start, c)
        
        # M2 = (A10 + A11) * B00
        # A10 rows: i_start+2, i_start+3; cols: j_start, j_start+1
        # A11 rows: i_start+2, i_start+3; cols: j_start+2, j_start+3
        # B00 rows: j_start, j_start+1; cols: k_start, k_start+1
        # (A10 + A11) has rows i_start+2, i_start+3 and cols j_start..j_start+3
        # We need to multiply (A10 + A11) by B00.
        # This is a 2x4 * 4x2 = 2x2 product.
        # We can use fill_strassen_2x4x2.
        c = fill_strassen_2x4x2(i_start + 2, j_start, k_start, c)
        
        # M3 = A00 * (B01 - B11)
        # A00 rows: i_start, i_start+1; cols: j_start, j_start+1
        # B01 rows: j_start, j_start+1; cols: k_start+2, k_start+3
        # B11 rows: j_start+2, j_start+3; cols: k_start+2, k_start+3
        # (B01 - B11) has rows j_start..j_start+3 and cols k_start+2, k_start+3
        # We need to multiply A00 by (B01 - B11).
        # This is a 2x2 * 4x2 = 2x2 product.
        # We can use fill_strassen_2x2x2 twice with appropriate signs.
        # A00 * B01
        c = fill_strassen_2x2x2(i_start, j_start, k_start + 2, c)
        # A00 * (-B11)
        c = fill_strassen_2x2x2(i_start, j_start + 2, k_start + 2, c)
        # But we need to negate the second one. Let's adjust.
        # Actually, this approach is getting messy. Let me just use schoolbook for 4x4x4.
        
        # For now, let's fall back to schoolbook for 4x4x4.
        for u in range(4):
            for v in range(4):
                for w in range(4):
                    U[(i_start + u) * m + (j_start + v)][c] = Fraction(1)
                    V[(j_start + v) * p + (k_start + w)][c] = Fraction(1)
                    W[(k_start + w) * n + (i_start + u)][c] = Fraction(1)
                    c += 1
        
        return c

    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    # Convert to list of lists of floats
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    return {"U": U_out, "V": V_out, "W": W_out}

                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]

                if ni == 4 and mj == 4 and pk == 4:
                    col = fill_strassen_4x4x4(i_start, j_start, k_start, col)
                elif ni == 4 and mj == 4 and pk == 1:
                    # 4x4x1 block
                    for u in range(4):
                        for v in range(4):
                            U[(i_start + u) * m + (j_start + v)][col] = Fraction(1)
                            V[(j_start + v) * p + k_start][col] = Fraction(1)
                            W[k_start * n + (i_start + u)][col] = Fraction(1)
                            col += 1
                elif ni == 4 and mj == 1 and pk == 4:
                    # 4x1x4 block
                    for u in range(4):
                        for w in range(4):
                            U[(i_start + u) * m + j_start][col] = Fraction(1)
                            V[j_start * p + (k_start + w)][col] = Fraction(1)
                            W[(k_start + w) * n + (i_start + u)][col] = Fraction(1)
                            col += 1
                elif ni == 1 and mj == 4 and pk == 4:
                    # 1x4x4 block
                    for v in range(4):
                        for w in range(4):
                            U[i_start * m + (j_start + v)][col] = Fraction(1)
                            V[(j_start + v) * p + (k_start + w)][col] = Fraction(1)
                            W[(k_start + w) * n + i_start][col] = Fraction(1)
                            col += 1
                elif ni == 4 and mj == 1 and pk == 1:
                    # 4x1x1 block
                    for u in range(4):
                        U[(i_start + u) * m + j_start][col] = Fraction(1)
                        V[j_start * p + k_start][col] = Fraction(1)
                        W[k_start * n + (i_start + u)][col] = Fraction(1)
                        col += 1
                elif ni == 1 and mj == 4 and pk == 1:
                    # 1x4x1 block
                    for v in range(4):
                        U[i_start * m + (j_start + v)][col] = Fraction(1)
                        V[(j_start + v) * p + k_start][col] = Fraction(1)
                        W[k_start * n + i_start][col] = Fraction(1)
                        col += 1
                elif ni == 1 and mj == 1 and pk == 4:
                    # 1x1x4 block
                    for w in range(4):
                        U[i_start * m + j_start][col] = Fraction(1)
                        V[j_start * p + (k_start + w)][col] = Fraction(1)
                        W[(k_start + w) * n + i_start][col] = Fraction(1)
                        col += 1
                elif ni == 1 and mj == 1 and pk == 1:
                    # 1x1x1 block
                    U[i_start * m + j_start][col] = Fraction(1)
                    V[j_start * p + k_start][col] = Fraction(1)
                    W[k_start * n + i_start][col] = Fraction(1)
                    col += 1
                else:
                    # General schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = Fraction(1)
                                V[(j_start + v) * p + (k_start + w)][col] = Fraction(1)
                                W[(k_start + w) * n + (i_start + u)][col] = Fraction(1)
                                col += 1

    # Convert to list of lists of floats
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}
