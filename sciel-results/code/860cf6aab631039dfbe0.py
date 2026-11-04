"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * rank for _ in range(p * n)]  # (12, 33)
    
    def add_term(u_dict, v_dict, w_dict, col):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
    
    for g in range(3):
        col0 = 2 * g
        col1 = 2 * g + 1
        base = 7 * g
        
        # x11=A[0,0]=0, x12=A[0,1]=1, x21=A[1,0]=3, x22=A[1,1]=4
        # y11=B[0,col0]=col0, y12=B[0,col1]=col1, y21=B[1,col0]=6+col0, y22=B[1,col1]=6+col1
        # W: C[i,k] at k*2+i
        
        # m1 = (x11+x22)(y11+y22)
        # Z11 += m1, Z22 += m1
        add_term({0:1, 4:1}, {col0:1, 6+col1:1}, {2*col0:1, 2*col1+1:1}, base+0)
        
        # m2 = (x21+x22)y11
        # Z21 += m2, Z22 += m2
        add_term({3:1, 4:1}, {col0:1}, {2*col0+1:1, 2*col1+1:1}, base+1)
        
        # m3 = x11(y12-y22)
        # Z12 += m3, Z22 -= m3
        add_term({0:1}, {col1:1, 6+col1:-1}, {2*col1:1, 2*col1+1:-1}, base+2)
        
        # m4 = x22(y21-y11)
        # Z11 -= m4, Z21 += m4
        add_term({4:1}, {6+col0:1, col0:-1}, {2*col0:-1, 2*col0+1:1}, base+3)
        
        # m5 = (x11+x12)y22
        # Z11 += m5, Z12 += m5
        add_term({0:1, 1:1}, {6+col1:1}, {2*col0:1, 2*col1:1}, base+4)
        
        # m6 = (x21-x11)(y11+y12)
        # Z11 += m6
        add_term({3:1, 0:-1}, {col0:1, col1:1}, {2*col0:1}, base+5)
        
        # m7 = (x12-x22)(y21+y22)
        # Z22 += m7
        add_term({1:1, 4:-1}, {6+col0:1, 6+col1:1}, {2*col1+1:1}, base+6)
    
    # 12 schoolbook terms for A[:,2] * B[2,:]
    for i in range(2):
        for k in range(6):
            c = 21 + i * 6 + k
            U[i*3+2][c] = 1
            V[2*6+k][c] = 1
            W[k*2+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
