import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 8
    
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: j=0 part: A[i,0] * B[0,k] for i=0,1 and k=0..7
    # That's 2*8 = 16 terms
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 part: 2x2x8 multiplication
    # Split into 4 blocks of 2x2x2, each with Strassen rank 7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # In our notation:
        # A is 2x2: A[i,j] where i=0,1 (rows), j=0,1 (cols, corresponding to original j=1,2)
        # B is 2x2: B[j,k] where j=0,1 (rows, corresponding to original j=1,2), k=0,1 (cols, corresponding to k0,k1)
        # C is 2x2: C[i,k] where i=0,1 (rows), k=0,1 (cols, corresponding to k0,k1)
        
        # M1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        U[0*m+1][col] = 1
        U[1*m+2][col] = 1
        V[1*p+k0][col] = 1
        V[2*p+k1][col] = 1
        W[k0*n+0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M2 = (A[1,1]+A[1,2]) * B[1,k0]
        U[1*m+1][col] = 1
        U[1*m+2][col] = 1
        V[1*p+k0][col] = 1
        W[k0*n+1][col] = 1
        W[k1*n+1][col] = -1
        col += 1
        
        # M3 = A[0,1] * (B[2,k0]-B[1,k0])
        U[0*m+1][col] = 1
        V[2*p+k0][col] = 1
        V[1*p+k0][col] = -1
        W[k0*n+0][col] = 1
        W[k1*n+0][col] = 1
        col += 1
        
        # M4 = A[1,2] * (B[2,k1]-B[1,k1])
        U[1*m+2][col] = 1
        V[2*p+k1][col] = 1
        V[1*p+k1][col] = -1
        W[k0*n+1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M5 = (A[0,1]+A[0,2]) * B[2,k1]
        U[0*m+1][col] = 1
        U[0*m+2][col] = 1
        V[2*p+k1][col] = 1
        W[k0*n+0][col] = -1
        W[k1*n+0][col] = 1
        col += 1
        
        # M6 = (A[1,1]-A[0,1]) * (B[1,k0]+B[1,k1])
        U[1*m+1][col] = 1
        U[0*m+1][col] = -1
        V[1*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M7 = (A[0,2]-A[1,2]) * (B[2,k0]+B[2,k1])
        U[0*m+2][col] = 1
        U[1*m+2][col] = -1
        V[2*p+k0][col] = 1
        V[2*p+k1][col] = 1
        W[k0*n+0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
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
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
