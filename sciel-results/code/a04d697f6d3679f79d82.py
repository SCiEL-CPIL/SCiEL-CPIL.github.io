import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    U = np.zeros((n * m, 36), dtype=np.int64)
    V = np.zeros((m * p, 36), dtype=np.int64)
    W = np.zeros((p * n, 36), dtype=np.int64)

    def add_strassen_2x2x2(base_col, j_off, k_off):
        # Strassen for 2x2x2 block.
        # U row index: i * m + (j_off + j)
        # V row index: (j_off + j) * p + (k_off + k)
        # W row index: (k_off + k) * n + i
        # Strassen formula:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # M1: U has A11, A22; V has B11, B22; W has C11(+), C22(+)
        U[0*m + j_off+0, base_col+0] += 1
        U[1*m + j_off+1, base_col+0] += 1
        V[(j_off+0)*p + k_off+0, base_col+0] += 1
        V[(j_off+1)*p + k_off+1, base_col+0] += 1
        W[(k_off+0)*n + 0, base_col+0] += 1
        W[(k_off+1)*n + 1, base_col+0] += 1

        # M2: U has A21, A22; V has B11; W has C21(+), C22(-)
        U[1*m + j_off+0, base_col+1] += 1
        U[1*m + j_off+1, base_col+1] += 1
        V[(j_off+0)*p + k_off+0, base_col+1] += 1
        W[(k_off+0)*n + 1, base_col+1] += 1
        W[(k_off+1)*n + 1, base_col+1] += -1

        # M3: U has A11; V has B12, -B22; W has C12(+), C22(+)
        U[0*m + j_off+0, base_col+2] += 1
        V[(j_off+0)*p + k_off+1, base_col+2] += 1
        V[(j_off+1)*p + k_off+1, base_col+2] += -1
        W[(k_off+1)*n + 0, base_col+2] += 1
        W[(k_off+1)*n + 1, base_col+2] += 1

        # M4: U has A22; V has -B11, B21; W has C11(+), C21(+)
        U[1*m + j_off+1, base_col+3] += 1
        V[(j_off+0)*p + k_off+0, base_col+3] += -1
        V[(j_off+1)*p + k_off+0, base_col+3] += 1
        W[(k_off+0)*n + 0, base_col+3] += 1
        W[(k_off+0)*n + 1, base_col+3] += 1

        # M5: U has A11, A12; V has B22; W has C11(-), C12(+)
        U[0*m + j_off+0, base_col+4] += 1
        U[0*m + j_off+1, base_col+4] += 1
        V[(j_off+1)*p + k_off+1, base_col+4] += 1
        W[(k_off+0)*n + 0, base_col+4] += -1
        W[(k_off+1)*n + 0, base_col+4] += 1

        # M6: U has -A11, A21; V has B11, B12; W has C22(+)
        U[0*m + j_off+0, base_col+5] += -1
        U[1*m + j_off+0, base_col+5] += 1
        V[(j_off+0)*p + k_off+0, base_col+5] += 1
        V[(j_off+0)*p + k_off+1, base_col+5] += 1
        W[(k_off+1)*n + 1, base_col+5] += 1

        # M7: U has A12, -A22; V has B21, B22; W has C11(+)
        U[0*m + j_off+1, base_col+6] += 1
        U[1*m + j_off+1, base_col+6] += -1
        V[(j_off+1)*p + k_off+0, base_col+6] += 1
        V[(j_off+1)*p + k_off+1, base_col+6] += 1
        W[(k_off+0)*n + 0, base_col+6] += 1

    def add_schoolbook_2x2x1(base_col, j_off, k_off):
        # Schoolbook for 2x2x1 (k is fixed to k_off).
        # C_ij = A_ij * B_jk
        # U: A_ij, V: B_jk, W: C_ki (since W[k*n+i])
        U[0*m + j_off+0, base_col+0] += 1
        V[(j_off+0)*p + k_off, base_col+0] += 1
        W[k_off*n + 0, base_col+0] += 1

        U[0*m + j_off+1, base_col+1] += 1
        V[(j_off+1)*p + k_off, base_col+1] += 1
        W[k_off*n + 0, base_col+1] += 1

        U[1*m + j_off+0, base_col+2] += 1
        V[(j_off+0)*p + k_off, base_col+2] += 1
        W[k_off*n + 1, base_col+2] += 1

        U[1*m + j_off+1, base_col+3] += 1
        V[(j_off+1)*p + k_off, base_col+3] += 1
        W[k_off*n + 1, base_col+3] += 1

    col = 0
    # J1 (j=0,1)
    add_strassen_2x2x2(col, 0, 0); col += 7
    add_strassen_2x2x2(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4

    # J2 (j=2,3)
    add_strassen_2x2x2(col, 2, 0); col += 7
    add_strassen_2x2x2(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4

    return {"U": U, "V": V, "W": W}
