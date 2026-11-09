DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0] * rank for _ in range(m * p)]  # 12 x 16
    W = [[0] * rank for _ in range(p * n)]  # 8 x 16
    
    # Helper to set values
    def set_U(i, j, r, val):
        U[i * m + j][r] += val
    
    def set_V(j, k, r, val):
        V[j * p + k][r] += val
    
    def set_W(k, i, r, val):
        W[k * n + i][r] += val
    
    # Terms 0-6: Strassen for A[:,0:2] * B[0:2, 0:2]
    # M1 (r=0): (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    set_U(0, 0, 0, 1)
    set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1)
    set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1)
    set_W(1, 1, 0, 1)
    
    # M2 (r=1): (A[1,0]+A[1,1])B[0,0]
    set_U(1, 0, 1, 1)
    set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1)
    set_W(1, 1, 1, -1)
    
    # M3 (r=2): A[0,0](B[0,1]-B[1,1])
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1)
    set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1)
    set_W(1, 1, 2, 1)
    
    # M4 (r=3): A[1,1](B[1,0]-B[0,0])
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1)
    set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1)
    set_W(0, 1, 3, 1)
    
    # M5 (r=4): (A[0,0]+A[0,1])B[1,1]
    set_U(0, 0, 4, 1)
    set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1)
    set_W(1, 0, 4, 1)
    
    # M6 (r=5): (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    set_U(1, 0, 5, 1)
    set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1)
    set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # M7 (r=6): (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    set_U(0, 1, 6, 1)
    set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1)
    set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)
    
    # Terms 7-13: Strassen for A[:,0:2] * B[0:2, 2:4]
    # M1' (r=7): (A[0,0]+A[1,1])(B[0,2]+B[1,3])
    set_U(0, 0, 7, 1)
    set_U(1, 1, 7, 1)
    set_V(0, 2, 7, 1)
    set_V(1, 3, 7, 1)
    set_W(2, 0, 7, 1)
    set_W(3, 1, 7, 1)
    
    # M2' (r=8): (A[1,0]+A[1,1])B[0,2]
    set_U(1, 0, 8, 1)
    set_U(1, 1, 8, 1)
    set_V(0, 2, 8, 1)
    set_W(2, 1, 8, 1)
    set_W(3, 1, 8, -1)
    
    # M3' (r=9): A[0,0](B[0,3]-B[1,3])
    set_U(0, 0, 9, 1)
    set_V(0, 3, 9, 1)
    set_V(1, 3, 9, -1)
    set_W(3, 0, 9, 1)
    set_W(3, 1, 9, 1)
    
    # M4' (r=10): A[1,1](B[1,2]-B[0,2])
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_V(0, 2, 10, -1)
    set_W(2, 0, 10, 1)
    set_W(2, 1, 10, 1)
    
    # M5' (r=11): (A[0,0]+A[0,1])B[1,3]
    set_U(0, 0, 11, 1)
    set_U(0, 1, 11, 1)
    set_V(1, 3, 11, 1)
    set_W(2, 0, 11, -1)
    set_W(3, 0, 11, 1)
    
    # M6' (r=12): (A[1,0]-A[0,0])(B[0,2]+B[0,3])
    set_U(1, 0, 12, 1)
    set_U(0, 0, 12, -1)
    set_V(0, 2, 12, 1)
    set_V(0, 3, 12, 1)
    set_W(3, 1, 12, 1)
    
    # M7' (r=13): (A[0,1]-A[1,1])(B[1,2]+B[1,3])
    set_U(0, 1, 13, 1)
    set_U(1, 1, 13, -1)
    set_V(1, 2, 13, 1)
    set_V(1, 3, 13, 1)
    set_W(2, 0, 13, 1)
    
    # Terms 14-15: 2x1 by 1x4 part (j=2)
    # Term 14 (r=14): A[0,2] * B[2,k]
    set_U(0, 2, 14, 1)
    for k in range(4):
        set_V(2, k, 14, 1)
        set_W(k, 0, 14, 1)
    
    # Term 15 (r=15): A[1,2] * B[2,k]
    set_U(1, 2, 15, 1)
    for k in range(4):
        set_V(2, k, 15, 1)
        set_W(k, 1, 15, 1)
    
    return {"U": U, "V": V, "W": W}
