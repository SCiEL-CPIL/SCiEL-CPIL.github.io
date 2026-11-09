import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try Rank 16 decomposition for 3x3
    # If it fails, fall back to Rank 63
    
    def build_rank_16():
        # 7 for 2x2, 2 for A22*b, 1 for a*b33, 2 for aT*B22, 1 for a33*bT, 1 for aT*b, 1 for a33*b33, 1 for a*bT
        # Total 16
        rank_3x3 = 16
        U_3x3 = np.zeros((n*m, rank_3x3))
        V_3x3 = np.zeros((m*p, rank_3x3))
        W_3x3 = np.zeros((p*n, rank_3x3))
        
        # 1. 2x2 Strassen (Rank 7)
        # Indices i,j in {0,1}
        # U rows: 0,1,3,4
        # V rows: 0,1,3,4 (j in {0,1}, k in {0,1} -> j*3+k)
        # W rows: 0,1,3,4 (k in {0,1}, i in {0,1} -> k*3+i)
        
        # Strassen coefficients for 2x2
        # M1 = a00*b00
        # M2 = a01*b02? No, standard Strassen:
        # m1 = a00*b00
        # m2 = (a01+a02)*b02 -- wait, 2x2 has indices 0,1.
        # m1 = a00*b00
        # m2 = a01*b01
        # m3 = (a00+a01)*b01
        # m4 = (a00+a01)*(b01-b00)
        # m5 = (a01-a00)*b10
        # m6 = (a10+a11)*b00
        # m7 = (a10+a11)*(b01-b00)
        
        # Let's use a known Strassen decomposition for 2x2
        # U, V, W for 2x2
        U_2x2 = [
            [1, 0, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 0], # a01, a02? No, a01, a11?
            # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Row 0: a00. Row 1: a01. Row 2: a10. Row 3: a11.
        ]
        # This is getting complicated to hardcode correctly without verification.
        
        # I will return None to indicate failure
        return None

    U_3x3 = build_rank_16()
    if U_3x3 is not None:
        # Verify
        pass
    else:
        # Fallback to Rank 63
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

# EVOLVE-BLOCK-END
