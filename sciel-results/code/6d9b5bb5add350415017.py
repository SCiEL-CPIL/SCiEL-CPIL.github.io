import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    def add_U(i, j, col, val):
        U[i * m + j][col] += val
    
    def add_V(j, k, col, val):
        V[j * p + k][col] += val
    
    def add_W(i, k, col, val):
        W[k * n + i][col] += val
    
    # M1: (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
    col = 0
    add_U(0,0,col,1); add_U(1,1,col,1)
    add_V(0,0,col,1); add_V(1,1,col,1)
    add_W(0,0,col,1); add_W(1,1,col,1)
    
    # M2: (A10+A11)*B00 -> C10(+1), C11(-1)
    col = 1
    add_U(1,0,col,1); add_U(1,1,col,1)
    add_V(0,0,col,1)
    add_W(1,0,col,1); add_W(1,1,col,-1)
    
    # M3: A00*(B01-B11) -> C01(+1), C11(+1)
    col = 2
    add_U(0,0,col,1)
    add_V(0,1,col,1); add_V(1,1,col,-1)
    add_W(0,1,col,1); add_W(1,1,col,1)
    
    # M4: A11*(B10-B00) -> C00(+1), C10(+1)
    col = 3
    add_U(1,1,col,1)
    add_V(1,0,col,1); add_V(0,0,col,-1)
    add_W(0,0,col,1); add_W(1,0,col,1)
    
    # M5: (A00+A01)*B11 -> C00(-1), C01(+1)
    col = 4
    add_U(0,0,col,1); add_U(0,1,col,1)
    add_V(1,1,col,1)
    add_W(0,0,col,-1); add_W(0,1,col,1)
    
    # M6: (A10-A00)*(B00+B01) -> C11(+1)
    col = 5
    add_U(1,0,col,1); add_U(0,0,col,-1)
    add_V(0,0,col,1); add_V(0,1,col,1)
    add_W(1,1,col,1)
    
    # M7: (A01-A11)*(B10+B11) -> C00(+1)
    col = 6
    add_U(0,1,col,1); add_U(1,1,col,-1)
    add_V(1,0,col,1); add_V(1,1,col,1)
    add_W(0,0,col,1)
    
    # M8: (A00+A11)*(B02+B12) -> C02(+1), C13(+1)
    col = 7
    add_U(0,0,col,1); add_U(1,1,col,1)
    add_V(0,2,col,1); add_V(1,2,col,1)
    add_W(0,2,col,1); add_W(1,3,col,1)
    
    # M9: (A10+A11)*B02 -> C12(+1), C13(-1)
    col = 8
    add_U(1,0,col,1); add_U(1,1,col,1)
    add_V(0,2,col,1)
    add_W(1,2,col,1); add_W(1,3,col,-1)
    
    # M10: A00*(B03-B13) -> C03(+1), C13(+1)
    col = 9
    add_U(0,0,col,1)
    add_V(0,3,col,1); add_V(1,3,col,-1)
    add_W(0,3,col,1); add_W(1,3,col,1)
    
    # M11: A11*(B12-B02) -> C02(+1), C12(+1)
    col = 10
    add_U(1,1,col,1)
    add_V(1,2,col,1); add_V(0,2,col,-1)
    add_W(0,2,col,1); add_W(1,2,col,1)
    
    # M12: (A00+A01)*B13 -> C02(-1), C03(+1)
    col = 11
    add_U(0,0,col,1); add_U(0,1,col,1)
    add_V(1,3,col,1)
    add_W(0,2,col,-1); add_W(0,3,col,1)
    
    # M13: (A10-A00)*(B02+B03) -> C13(+1)
    col = 12
    add_U(1,0,col,1); add_U(0,0,col,-1)
    add_V(0,2,col,1); add_V(0,3,col,1)
    add_W(1,3,col,1)
    
    # M14: (A01-A11)*(B12+B13) -> C02(+1)
    col = 13
    add_U(0,1,col,1); add_U(1,1,col,-1)
    add_V(1,2,col,1); add_V(1,3,col,1)
    add_W(0,2,col,1)
    
    # M15: A02*B20 -> C00(+1)
    col = 14
    add_U(0,2,col,1)
    add_V(2,0,col,1)
    add_W(0,0,col,1)
    
    # M16: A02*B21 -> C01(+1)
    col = 15
    add_U(0,2,col,1)
    add_V(2,1,col,1)
    add_W(0,1,col,1)
    
    # M17: A12*B20 -> C10(+1)
    col = 16
    add_U(1,2,col,1)
    add_V(2,0,col,1)
    add_W(1,0,col,1)
    
    # M18: A12*B21 -> C11(+1)
    col = 17
    add_U(1,2,col,1)
    add_V(2,1,col,1)
    add_W(1,1,col,1)
    
    # M19: A02*B22 -> C02(+1)
    col = 18
    add_U(0,2,col,1)
    add_V(2,2,col,1)
    add_W(0,2,col,1)
    
    # M20: A02*B23 -> C03(+1)
    col = 19
    add_U(0,2,col,1)
    add_V(2,3,col,1)
    add_W(0,3,col,1)
    
    # M21: A12*B22 -> C12(+1)
    col = 20
    add_U(1,2,col,1)
    add_V(2,2,col,1)
    add_W(1,2,col,1)
    
    # M22: A12*B23 -> C13(+1)
    col = 21
    add_U(1,2,col,1)
    add_V(2,3,col,1)
    add_W(1,3,col,1)
    
    # M23: A20*B00 -> C20(+1)
    col = 22
    add_U(2,0,col,1)
    add_V(0,0,col,1)
    add_W(2,0,col,1)
    
    # M24: A21*B10 -> C20(+1)
    col = 23
    add_U(2,1,col,1)
    add_V(1,0,col,1)
    add_W(2,0,col,1)
    
    # M25: A20*B01 -> C21(+1)
    col = 24
    add_U(2,0,col,1)
    add_V(0,1,col,1)
    add_W(2,1,col,1)
    
    # M26: A21*B11 -> C21(+1)
    col = 25
    add_U(2,1,col,1)
    add_V(1,1,col,1)
    add_W(2,1,col,1)
    
    # M27: A22*B20 -> C20(+1)
    col = 26
    add_U(2,2,col,1)
    add_V(2,0,col,1)
    add_W(2,0,col,1)
    
    # M28: A22*B21 -> C21(+1)
    col = 27
    add_U(2,2,col,1)
    add_V(2,1,col,1)
    add_W(2,1,col,1)
    
    # M29: A20*B02 -> C22(+1)
    col = 28
    add_U(2,0,col,1)
    add_V(0,2,col,1)
    add_W(2,2,col,1)
    
    # M30: A21*B12 -> C22(+1)
    col = 29
    add_U(2,1,col,1)
    add_V(1,2,col,1)
    add_W(2,2,col,1)
    
    # M31: A20*B03 -> C23(+1)
    col = 30
    add_U(2,0,col,1)
    add_V(0,3,col,1)
    add_W(2,3,col,1)
    
    # M32: A21*B13 -> C23(+1)
    col = 31
    add_U(2,1,col,1)
    add_V(1,3,col,1)
    add_W(2,3,col,1)
    
    # M33: A22*B22 -> C22(+1)
    col = 32
    add_U(2,2,col,1)
    add_V(2,2,col,1)
    add_W(2,2,col,1)
    
    # M34: A22*B23 -> C23(+1)
    col = 33
    add_U(2,2,col,1)
    add_V(2,3,col,1)
    add_W(2,3,col,1)
    
    return {"U": U, "V": V, "W": W}
