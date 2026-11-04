import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _add_strassen_2x2(U, V, W, a_rows, b_rows, c_rows, start_col):
    """Add a 2x2x2 Strassen decomposition.
    a_rows: [A00, A01, A10, A11] U row indices
    b_rows: [B00, B01, B10, B11] V row indices  
    c_rows: [C00, C01, C10, C11] W row indices
    Returns next column index.
    """
    c = start_col
    # S1 = (A00+A11)*(B00+B11)
    U[a_rows[0], c] = 1; U[a_rows[3], c] = 1
    V[b_rows[0], c] = 1; V[b_rows[3], c] = 1
    W[c_rows[0], c] = 1; W[c_rows[3], c] = 1
    c += 1
    # S2 = (A10+A11)*B00
    U[a_rows[2], c] = 1; U[a_rows[3], c] = 1
    V[b_rows[0], c] = 1
    W[c_rows[2], c] = 1; W[c_rows[3], c] = -1
    c += 1
    # S3 = A00*(B10-B11)
    U[a_rows[0], c] = 1
    V[b_rows[2], c] = 1; V[b_rows[3], c] = -1
    W[c_rows[1], c] = 1; W[c_rows[3], c] = 1
    c += 1
    # S4 = A11*(B01-B00)
    U[a_rows[3], c] = 1
    V[b_rows[1], c] = 1; V[b_rows[0], c] = -1
    W[c_rows[0], c] = 1; W[c_rows[2], c] = 1
    c += 1
    # S5 = (A00+A01)*B11
    U[a_rows[0], c] = 1; U[a_rows[1], c] = 1
    V[b_rows[3], c] = 1
    W[c_rows[0], c] = -1; W[c_rows[1], c] = 1
    c += 1
    # S6 = (A10-A00)*(B00+B01)
    U[a_rows[2], c] = 1; U[a_rows[0], c] = -1
    V[b_rows[0], c] = 1; V[b_rows[1], c] = 1
    W[c_rows[3], c] = 1
    c += 1
    # S7 = (A01-A11)*(B10+B11)
    U[a_rows[1], c] = 1; U[a_rows[3], c] = -1
    V[b_rows[2], c] = 1; V[b_rows[3], c] = 1
    W[c_rows[0], c] = 1
    c += 1
    return c


def _add_strassen_4x4x4(U, V, W, a_base, b_base, c_base, start_col):
    """Add a 4x4x4 Strassen (rank 49) decomposition.
    a_base: base row offset in U for the 4x4 A block (row i maps to a_base + i*4 + j)
    b_base: base row offset in V for the 4x4 B block (row k maps to b_base + k*4 + j)
    c_base: base row offset in W for the 4x4 C block (row i maps to c_base + j*4 + i)
    Wait, I need to be more careful with the indexing.
    
    Actually, let me just pass the specific row indices.
    """
    pass  # Will implement below
