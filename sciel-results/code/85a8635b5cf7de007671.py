"""Matrix multiplication 3x3x3 - rank 26 exact tensor decomposition."""
import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 26
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # Indices:
    # A2: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 3, 4
    # B2: (0,0), (0,1), (1,0), (1,1) -> V indices 0, 1, 3, 4
    # C22: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 3, 1, 4 (k*n+i)
    # C_{00} -> 0*3+0 = 0
    # C_{01} -> 1*3+0 = 3
    # C_{10} -> 0*3+1 = 1
    # C_{11} -> 1*3+1 = 4
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    # a = A[0:2, 2] -> U indices 2, 5
    # b^T = B[2, 0:2] -> V indices 6, 7
    # C_{00} (W0) += A02 B20
    # C_{01} (W3) += A02 B21
    # C_{10} (W1) += A12 B20
    # C_{11} (W4) += A12 B21
    
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    # A2: U 0,1,3,4
    # b = B[0:2, 2] -> V indices 2, 5
    # C_{02} (W6) += A00 B02 + A01 B12
    # C_{12} (W7) += A10 B02 + A11 B12
    
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    # a: U 2,5
    # b3 = B[2,2] -> V index 8
    # C_{02} (W6) += A02 B22
    # C_{12} (W7) += A12 B22
    
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    # a^T = A[2, 0:2] -> U indices 6, 7
    # B2: V 0,1,3,4
    # C_{20} (W2) += A20 B00 + A21 B10
    # C_{21} (W5) += A20 B01 + A21 B11
    
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    # a3 = A[2,2] -> U index 8
    # b^T = B[2, 0:2] -> V indices 6, 7
    # C_{20} (W2) += A22 B20
    # C_{21} (W5) += A22 B21
    
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    # a^T: U 6,7
    # b: V 2,5
    # C_{22} (W8) += A20 B02 + A21 B12
    
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    # a3: U 8
    # b3: V 8
    # C_{22} (W8) += A22 B22
    
    add_term(8, 8, 8, 1)
    
    # Verify correctness
    # Check that the decomposition correctly computes matrix multiplication
    # by testing against a few random matrices
    import random
    
    def mat_mult(A, B):
        result = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(m):
                if A[i][k] != 0:
                    for j in range(p):
                        result[i][j] += A[i][k] * B[k][j]
        return result
    
    def decomposed_mult(A, B):
        # Compute using the decomposition
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m+k, r] = coefficient for A[i,k] in term r
        # V[k*p+j, r] = coefficient for B[k,j] in term r
        # W[j*n+i, r] = coefficient for C[i,j] in term r
        
        # Initialize result
        result = [[0.0] * p for _ in range(n)]
        
        for r in range(rank):
            # Compute the product of the two vectors in term r
            u_vec = [U[i * m + k][r] for i in range(n) for k in range(m)]
            v_vec = [V[k * p + j][r] for k in range(m) for j in range(p)]
            
            # Compute the product
            prod = 0.0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        prod += u_vec[i * m + k] * A[i][k] * v_vec[k * p + j] * B[k][j]
            
            # Add to result
            for i in range(n):
                for j in range(p):
                    result[i][j] += W[j * n + i][r] * prod
        
        return result
    
    # Test with a few random matrices
    for _ in range(3):
        A_test = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
        B_test = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
        
        C_direct = mat_mult(A_test, B_test)
        C_decomp = decomposed_mult(A_test, B_test)
        
        for i in range(n):
            for j in range(p):
                if abs(C_direct[i][j] - C_decomp[i][j]) > 1e-9:
                    raise ValueError(f"Decomposition incorrect at ({i},{j}): {C_direct[i][j]} vs {C_decomp[i][j]}")
    
    # Convert to integer/half-integer entries
    for i in range(n * m):
        for r in range(rank):
            if abs(U[i][r] - round(U[i][r])) < 1e-10:
                U[i][r] = round(U[i][r])
    for i in range(m * p):
        for r in range(rank):
            if abs(V[i][r] - round(V[i][r])) < 1e-10:
                V[i][r] = round(V[i][r])
    for i in range(p * n):
        for r in range(rank):
            if abs(W[i][r] - round(W[i][r])) < 1e-10:
                W[i][r] = round(W[i][r])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
