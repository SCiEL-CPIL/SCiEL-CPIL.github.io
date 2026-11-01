import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # Part 1: Strassen for cols 0, 1
    # Term 1: M1 = (A00 + A11)(B01 + B10)
    # U: A00(0)=1, A11(3)=1
    set_val(U, 0, 0, 1)
    set_val(U, 3, 0, 1)
    # V: B01(1)=1, B10(3)=1
    set_val(V, 1, 0, 1)
    set_val(V, 3, 0, 1)
    # W: C00(0)=1, C11(3)=1
    set_val(W, 0, 0, 1)
    set_val(W, 3, 0, 1)

    # Term 2: M2 = (A10 + A11)B00
    # U: A10(2)=1, A11(3)=1
    set_val(U, 2, 1, 1)
    set_val(U, 3, 1, 1)
    # V: B00(0)=1
    set_val(V, 0, 1, 1)
    # W: C10(1)=1, C11(3)=-1
    set_val(W, 1, 1, 1)
    set_val(W, 3, 1, -1)

    # Term 3: M3 = A00(B10 - B11)
    # U: A00(0)=1
    set_val(U, 0, 2, 1)
    # V: B10(3)=1, B11(4)=-1
    set_val(V, 3, 2, 1)
    set_val(V, 4, 2, -1)
    # W: C01(2)=1, C11(3)=1
    set_val(W, 2, 2, 1)
    set_val(W, 3, 2, 1)

    # Term 4: M4 = A11(B01 - B00)
    # U: A11(3)=1
    set_val(U, 3, 3, 1)
    # V: B01(1)=1, B00(0)=-1
    set_val(V, 1, 3, 1)
    set_val(V, 0, 3, -1)
    # W: C00(0)=1, C10(1)=1
    set_val(W, 0, 3, 1)
    set_val(W, 1, 3, 1)

    # Term 5: M5 = (A00 + A01)B11
    # U: A00(0)=1, A01(1)=1
    set_val(U, 0, 4, 1)
    set_val(U, 1, 4, 1)
    # V: B11(4)=1
    set_val(V, 4, 4, 1)
    # W: C00(0)=-1, C01(2)=1
    set_val(W, 0, 4, -1)
    set_val(W, 2, 4, 1)

    # Term 6: M6 = (A10 - A00)(B00 + B01)
    # U: A10(2)=1, A00(0)=-1
    set_val(U, 2, 5, 1)
    set_val(U, 0, 5, -1)
    # V: B00(0)=1, B01(1)=1
    set_val(V, 0, 5, 1)
    set_val(V, 1, 5, 1)
    # W: C11(3)=1
    set_val(W, 3, 5, 1)

    # Term 7: M7 = (A01 - A11)(B10 + B11)
    # U: A01(1)=1, A11(3)=-1
    set_val(U, 1, 6, 1)
    set_val(U, 3, 6, -1)
    # V: B10(3)=1, B11(4)=1
    set_val(V, 3, 6, 1)
    set_val(V, 4, 6, 1)
    # W: C00(0)=1, C11(3)=-1
    set_val(W, 0, 6, 1)
    set_val(W, 3, 6, -1)

    # Part 2: Schoolbook for col 2
    # Term 8: A00 B02
    # U: A00(0)=1
    set_val(U, 0, 7, 1)
    # V: B02(2)=1
    set_val(V, 2, 7, 1)
    # W: C02(4)=1
    set_val(W, 4, 7, 1)

    # Term 9: A01 B12
    # U: A01(1)=1
    set_val(U, 1, 8, 1)
    # V: B12(5)=1
    set_val(V, 5, 8, 1)
    # W: C02(4)=1
    set_val(W, 4, 8, 1)

    # Term 10: A10 B02
    # U: A10(2)=1
    set_val(U, 2, 9, 1)
    # V: B02(2)=1
    set_val(V, 2, 9, 1)
    # W: C12(5)=1
    set_val(W, 5, 9, 1)

    # Term 11: A11 B12
    # U: A11(3)=1
    set_val(U, 3, 10, 1)
    # V: B12(5)=1
    set_val(V, 5, 10, 1)
    # W: C12(5)=1
    set_val(W, 5, 10, 1)

    return {"U": U, "V": V, "W": W}
