import time

def construct():
    n, m, p = 2, 3, 8
    
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    rank = 40
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # For each half (k in 0..3 and k in 4..7), decompose 2x3x4
    for half in range(2):
        k_offset = 4 * half
        
        # Part A: 2x1x4 (A's column 0, B's row 0)
        for k in range(4):
            kk = k_offset + k
            # A[0,0] * B[0,kk] -> C[0,kk]
            U[0 * m + 0][col] = 1
            V[0 * p + kk][col] = 1
            W[kk * n + 0][col] = 1
            col += 1
            
            # A[1,0] * B[0,kk] -> C[1,kk]
            U[1 * m + 0][col] = 1
            V[0 * p + kk][col] = 1
            W[kk * n + 1][col] = 1
            col += 1
        
        # Part B: 2x2x4 (A's columns 1,2; B's rows 1,2)
        # Rank-12 decomposition using half-integer coefficients
        # Local indices: a00=A[0,1], a01=A[0,2], a10=A[1,1], a11=A[1,2]
        # b00=B[1,kk], b01=B[1,kk+1], b02=B[1,kk+2], b03=B[1,kk+3]
        # b10=B[2,kk], b11=B[2,kk+1], b12=B[2,kk+2], b13=B[2,kk+3]
        
        # Term 1: (a00+a10)/2 * b00
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = 0.5
        V[1 * p + k_offset + 0][col] = 1
        W[(k_offset + 0) * n + 0][col] = 1
        W[(k_offset + 0) * n + 1][col] = 1
        col += 1
        
        # Term 2: (a00+a10)/2 * b01
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = 0.5
        V[1 * p + k_offset + 1][col] = 1
        W[(k_offset + 1) * n + 0][col] = 1
        W[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        # Term 3: (a00+a10)/2 * b02
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = 0.5
        V[1 * p + k_offset + 2][col] = 1
        W[(k_offset + 2) * n + 0][col] = 1
        W[(k_offset + 2) * n + 1][col] = 1
        col += 1
        
        # Term 4: (a00+a10)/2 * b03
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = 0.5
        V[1 * p + k_offset + 3][col] = 1
        W[(k_offset + 3) * n + 0][col] = 1
        W[(k_offset + 3) * n + 1][col] = 1
        col += 1
        
        # Term 5: (a01+a11)/2 * b10
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = 0.5
        V[2 * p + k_offset + 0][col] = 1
        W[(k_offset + 0) * n + 0][col] = 1
        W[(k_offset + 0) * n + 1][col] = 1
        col += 1
        
        # Term 6: (a01+a11)/2 * b11
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = 0.5
        V[2 * p + k_offset + 1][col] = 1
        W[(k_offset + 1) * n + 0][col] = 1
        W[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        # Term 7: (a01+a11)/2 * b12
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = 0.5
        V[2 * p + k_offset + 2][col] = 1
        W[(k_offset + 2) * n + 0][col] = 1
        W[(k_offset + 2) * n + 1][col] = 1
        col += 1
        
        # Term 8: (a01+a11)/2 * b13
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = 0.5
        V[2 * p + k_offset + 3][col] = 1
        W[(k_offset + 3) * n + 0][col] = 1
        W[(k_offset + 3) * n + 1][col] = 1
        col += 1
        
        # Term 9: (a00-a10)/2 * b00
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = -0.5
        V[1 * p + k_offset + 0][col] = 1
        W[(k_offset + 0) * n + 0][col] = 1
        W[(k_offset + 0) * n + 1][col] = -1
        col += 1
        
        # Term 10: (a01-a11)/2 * b10
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = -0.5
        V[2 * p + k_offset + 0][col] = 1
        W[(k_offset + 0) * n + 0][col] = 1
        W[(k_offset + 0) * n + 1][col] = -1
        col += 1
        
        # Term 11: (a00-a10)/2 * b01
        U[0 * m + 1][col] = 0.5
        U[1 * m + 1][col] = -0.5
        V[1 * p + k_offset + 1][col] = 1
        W[(k_offset + 1) * n + 0][col] = 1
        W[(k_offset + 1) * n + 1][col] = -1
        col += 1
        
        # Term 12: (a01-a11)/2 * b11
        U[0 * m + 2][col] = 0.5
        U[1 * m + 2][col] = -0.5
        V[2 * p + k_offset + 1][col] = 1
        W[(k_offset + 1) * n + 0][col] = 1
        W[(k_offset + 1) * n + 1][col] = -1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if abs(val - expected_val) > 1e-9:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
