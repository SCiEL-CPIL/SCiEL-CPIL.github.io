DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Recursive Strassen-like decomposition generator
    # Returns U, V, W for the given dimensions
    def generate_strassen(n, m, p):
        # Base case: 2x2x2
        if n == 2 and m == 2 and p == 2:
            # Strassen 2x2x2 decomposition
            # U, V, W are 4x7, 4x7, 4x7
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # r=0
            U[0][0] += 1; U[3][0] += 1
            V[0][0] += 1; V[3][0] += 1
            W[0][0] += 1; W[3][0] += 1
            
            # r=1
            U[2][1] += 1; U[3][1] += 1
            V[0][1] += 1
            W[1][1] += 1; W[3][1] += -1
            
            # r=2
            U[0][2] += 1
            V[1][2] += 1; V[3][2] += -1
            W[2][2] += 1; W[3][2] += 1
            
            # r=3
            U[3][3] += 1
            V[2][3] += 1; V[0][3] += -1
            W[0][3] += 1; W[1][3] += 1
            
            # r=4
            U[0][4] += 1; U[1][4] += 1
            V[3][4] += 1
            W[0][4] += -1; W[2][4] += 1
            
            # r=5
            U[2][5] += 1; U[0][5] += -1
            V[0][5] += 1; V[1][5] += 1
            W[3][5] += 1
            
            # r=6
            U[1][6] += 1; U[3][6] += -1
            V[2][6] += 1; V[3][6] += 1
            W[0][6] += 1
            
            return U, V, W, rank

        # Recursive case
        # Split the largest dimension or use standard block decomposition
        # For simplicity, we split m and p if possible
        # We need to handle 2x4x8
        
        # If m is even, split m into m/2, m/2
        # If p is even, split p into p/2, p/2
        # We will split the dimension that allows recursion to base case 2x2x2
        
        # Strategy: Always split the dimension that is > 2
        # If n>2, split n. If m>2, split m. If p>2, split p.
        # Here n=2, m=4, p=8.
        # Split m=4 into 2,2.
        # Problem becomes two 2x2x8 multiplications.
        # Then split p=8 into 2,2.
        # Problem becomes four 2x2x4 multiplications.
        # Then split p=4 into 2,2.
        # Problem becomes eight 2x2x2 multiplications.
        
        # This will result in 8 * 7 = 56 rank.
        
        # Implementation:
        # We create empty U, V, W for the current size
        # We recurse on sub-problems and place their U, V, W blocks
        
        # Determine which dimension to split
        if m > 2:
            m1 = m // 2
            m2 = m - m1
            # Split A (n x m) into A1 (n x m1), A2 (n x m2)
            # Split B (m x p) into B1 (m1 x p), B2 (m2 x p)
            # C = A1 B1 + A2 B2
            
            # Recurse on (n, m1, p) and (n, m2, p)
            U1, V1, W1, r1 = generate_strassen(n, m1, p)
            U2, V2, W2, r2 = generate_strassen(n, m2, p)
            
            rank = r1 + r2
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Place U1, U2
            # U1 corresponds to A1 (rows 0..m1-1 of A)
            # U2 corresponds to A2 (rows m1..m-1 of A)
            for i in range(n):
                for j in range(m1):
                    U[i*m + j][0:r1] = U1[i*m1 + j]
                for j in range(m2):
                    U[i*m + m1 + j][r1:r1+r2] = U2[i*m2 + j]
                    
            # Place V1, V2
            # V1 corresponds to B1 (rows 0..m1-1 of B)
            # V2 corresponds to B2 (rows m1..m-1 of B)
            for j in range(m1):
                for k in range(p):
                    V[j*p + k][0:r1] = V1[j*p + k]
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p + k][r1:r1+r2] = V2[j*p + k]
                    
            # Place W1, W2
            # W1, W2 are identical in structure for C, but we just add them?
            # No, W is the same for both sub-problems because C is the same.
            # But the rank is additive.
            # W1 has rank r1, W2 has rank r2.
            # We need to combine them.
            # Actually, W1 and W2 are matrices of size p x n x rank.
            # The tensor decomposition is a sum of rank-1 terms.
            # The W matrix stores the coefficients for the output.
            # Since C = C1 + C2, the W blocks are just concatenated?
            # No, the W matrix is (p*n) x rank.
            # For the first r1 ranks, use W1.
            # For the next r2 ranks, use W2.
            for k in range(p):
                for i in range(n):
                    W[k*n + i][0:r1] = W1[k*n + i]
                    W[k*n + i][r1:r1+r2] = W2[k*n + i]
                    
            return U, V, W, rank
            
        elif p > 2:
            p1 = p // 2
            p2 = p - p1
            # Split B (m x p) into B1 (m x p1), B2 (m x p2)
            # C = C1 (n x p1) + C2 (n x p2)
            
            U1, V1, W1, r1 = generate_strassen(n, m, p1)
            U2, V2, W2, r2 = generate_strassen(n, m, p2)
            
            rank = r1 + r2
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Place U1, U2
            # U is the same for both sub-problems?
            # No, U is (n*m) x rank.
            # U1 has rank r1, U2 has rank r2.
            # We concatenate U1 and U2 horizontally?
            # Yes, because the input A is the same.
            for i in range(n):
                for j in range(m):
                    U[i*m + j][0:r1] = U1[i*m + j]
                    U[i*m + j][r1:r1+r2] = U2[i*m + j]
                    
            # Place V1, V2
            # V1 corresponds to B1 (cols 0..p1-1)
            # V2 corresponds to B2 (cols p1..p-1)
            for j in range(m):
                for k in range(p1):
                    V[j*p + k][0:r1] = V1[j*p1 + k]
                for k in range(p2):
                    V[j*p + p1 + k][r1:r1+r2] = V2[j*p2 + k]
                    
            # Place W1, W2
            # W1 corresponds to C1 (cols 0..p1-1)
            # W2 corresponds to C2 (cols p1..p-1)
            for k in range(p1):
                for i in range(n):
                    W[k*n + i][0:r1] = W1[k*n + i]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n + i][r1:r1+r2] = W2[k*n + i]
                    
            return U, V, W, rank
            
        elif n > 2:
            # Similar logic for n
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1, r1 = generate_strassen(n1, m, p)
            U2, V2, W2, r2 = generate_strassen(n2, m, p)
            
            rank = r1 + r2
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Place U1, U2
            for i in range(n1):
                for j in range(m):
                    U[i*m + j][0:r1] = U1[i*m + j]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m + j][r1:r1+r2] = U2[i*m + j]
                    
            # Place V1, V2
            for j in range(m):
                for k in range(p):
                    V[j*p + k][0:r1] = V1[j*p + k]
                    V[j*p + k][r1:r1+r2] = V2[j*p + k]
                    
            # Place W1, W2
            for k in range(p):
                for i in range(n1):
                    W[k*n + i][0:r1] = W1[k*n1 + i]
                for i in range(n2):
                    W[k*n + n1 + i][r1:r1+r2] = W2[k*n2 + i]
                    
            return U, V, W, rank
        else:
            # Should not reach here if base case is 2x2x2
            return None, None, None, 0

    U, V, W, rank = generate_strassen(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
