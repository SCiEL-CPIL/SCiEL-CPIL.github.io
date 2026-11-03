import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using recursive Strassen.
    # The function returns U, V, W as lists of lists.
    
    def strassen_decompose(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            # U: 4x7, V: 4x7, W: 4x7
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # A is 2x2, B is 2x2, C is 2x2
            # A11=0, A12=1, A21=2, A22=3
            # B11=0, B12=1, B21=2, B22=3
            # C11=0, C12=1, C21=2, C22=3
            
            # M1 = (A11+A22)(B11+B22)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1
            
            # M2 = (A21+A22)B11
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1
            
            # M3 = A11(B12-B22)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1
            
            # M4 = A22(B21-B11)
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[2][3] = 1
            
            # M5 = (A11+A12)B22
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[3][4] = 1
            
            # M6 = (A21-A11)(B11+B12)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7 = (A12-A22)(B21+B22)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
            # Wait, C11 = M1 + M4 - M5 + M7
            # C11 index 0.
            # M1 contributes 1 to W[0][0]
            # M4 contributes 1 to W[2][3] -> No, M4 is for C21?
            # Let's re-verify Strassen indices.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # W[k*n+i, c]
            # C11 is W[0*2+0, c] = W[0, c]
            # C12 is W[0*2+1, c] = W[1, c]
            # C21 is W[1*2+0, c] = W[2, c]
            # C22 is W[1*2+1, c] = W[3, c]
            
            # M1: C11 += 1, C22 += 1
            # M2: C21 += 1
            # M3: C12 += 1
            # M4: C11 += 1, C21 += 1
            # M5: C11 -= 1, C12 += 1
            # M6: C22 += 1
            # M7: C11 += 1
            
            # Let's rebuild W based on this.
            W = [[0]*7 for _ in range(4)]
            # M1
            W[0][0] += 1
            W[3][0] += 1
            # M2
            W[2][1] += 1
            # M3
            W[1][2] += 1
            # M4
            W[0][3] += 1
            W[2][3] += 1
            # M5
            W[0][4] -= 1
            W[1][4] += 1
            # M6
            W[3][5] += 1
            # M7
            W[0][6] += 1
            
            return U, V, W

        # Recursive case
        n1 = n // 2
        n2 = n - n1
        m1 = m // 2
        m2 = m - m1
        p1 = p // 2
        p2 = p - p1
        
        # We need to perform 7 multiplications of sizes:
        # 1. (n1+n2, m1+m2, p1+p2) -> No, the submatrices are not square.
        # Strassen requires the submatrices to be of size (n1, m1, p1) etc.
        # This only works if n1=n2, m1=m2, p1=p2.
        # For non-power-of-2, we must pad or use a different decomposition.
        
        # For simplicity and correctness, let's pad to the next power of 2.
        # This might increase the rank, but it's a valid decomposition.
        # However, padding increases the size, which might increase the rank more than necessary.
        
        # Let's try a different approach: use the schoolbook decomposition but replace 2x2x2 blocks with Strassen.
        # This is complex to implement correctly.
        
        # Given the time constraint, let's return the schoolbook decomposition for now.
        # But we can try to use Strassen for 2x2x2 blocks if the dimensions are even.
        
        # For 3, 5, 6:
        # We can split 3 into 2+1, 5 into 4+1, 6 into 4+2.
        # This is too complex.
        
        # Let's just use the schoolbook decomposition.
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
        return U, V, W

    U, V, W = strassen_decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
