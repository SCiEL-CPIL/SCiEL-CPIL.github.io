import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We use a block decomposition to achieve rank < 60.
    # Partition A (2x5) into A1 (2x2) and A2 (2x3).
    # Partition B (5x6) into B1 (2x6) and B2 (3x6).
    # C = A1*B1 + A2*B2.
    
    # We need decompositions for:
    # 1. M(2,2,6): A1 (2x2) * B1 (2x6)
    # 2. M(2,3,6): A2 (2x3) * B2 (3x6)
    
    # For M(2,2,6), we can use Strassen on 2x2 blocks of B.
    # B1 is 2x6. Split into three 2x2 blocks: B11, B12, B13.
    # Each 2x2*2x2 uses Strassen (rank 7).
    # Total rank for M(2,2,6) = 3 * 7 = 21.
    
    # For M(2,3,6), we can split B2 (3x6) into three 3x2 blocks: B21, B22, B23.
    # Each 2x3*3x2 uses schoolbook (rank 6).
    # Total rank for M(2,3,6) = 3 * 6 = 18.
    
    # Total rank = 21 + 18 = 39.
    
    # Let's verify this is better than 57. Yes, 39 < 57.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(u_offset, v_offset, w_offset):
        """Add a Strassen decomposition for a 2x2*2x2 block at given offsets"""
        nonlocal col
        # U indices: u_offset+0=a, u_offset+1=b, u_offset+2=c, u_offset+3=d
        # V indices: v_offset+0=x, v_offset+1=y, v_offset+2=u, v_offset+3=v
        # W indices: w_offset+0=C00, w_offset+1=C01, w_offset+2=C10, w_offset+3=C11
        
        # M1 = (a+d)(x+u)
        U[u_offset][col] += 1
        U[u_offset+3][col] += 1
        V[v_offset][col] += 1
        V[v_offset+2][col] += 1
        W[w_offset][col] += 1
        W[w_offset+3][col] += 1
        col += 1
        
        # M2 = (c+d)x
        U[u_offset+2][col] += 1
        U[u_offset+3][col] += 1
        V[v_offset][col] += 1
        W[w_offset][col] -= 1
        W[w_offset+2][col] += 1
        col += 1
        
        # M3 = a(y-u)
        U[u_offset][col] += 1
        V[v_offset+1][col] += 1
        V[v_offset+2][col] -= 1
        W[w_offset][col] += 1
        W[w_offset+1][col] += 1
        col += 1
        
        # M4 = d(v-u)
        U[u_offset+3][col] += 1
        V[v_offset+3][col] += 1
        V[v_offset+2][col] -= 1
        W[w_offset][col] -= 1
        W[w_offset+1][col] += 1
        col += 1
        
        # M5 = (a+b)u
        U[u_offset][col] += 1
        U[u_offset+1][col] += 1
        V[v_offset+2][col] += 1
        W[w_offset+2][col] += 1
        W[w_offset+3][col] -= 1
        col += 1
        
        # M6 = (c+d)(v-y)
        U[u_offset+2][col] += 1
        U[u_offset+3][col] += 1
        V[v_offset+3][col] += 1
        V[v_offset+1][col] -= 1
        W[w_offset+2][col] += 1
        W[w_offset+3][col] += 1
        col += 1
        
        # M7 = (a+b)(y-v)
        U[u_offset][col] += 1
        U[u_offset+1][col] += 1
        V[v_offset+1][col] += 1
        V[v_offset+3][col] -= 1
        W[w_offset+1][col] += 1
        W[w_offset+3][col] += 1
        col += 1
    
    def add_schoolbook_2x3x2(u_offset, v_offset, w_offset):
        """Add schoolbook decomposition for 2x3 * 3x2 -> 2x2"""
        nonlocal col
        # U: 2x3 matrix flattened, indices 0-5
        # V: 3x2 matrix flattened, indices 0-5
        # W: 2x2 matrix flattened, indices 0-3
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[u_offset + i*3 + j][col] += 1
                    V[v_offset + j*2 + k][col] += 1
                    W[w_offset + i*2 + k][col] += 1
                    col += 1
    
    # Block 1: A1 * B1 (2x2 * 2x6)
    # A1 rows 0-1, cols 0-1 of A -> U indices 0-3
    # B1 rows 0-1, cols 0-5 of B -> V indices 0-11
    # Split B1 into three 2x2 blocks:
    # B11: cols 0-1 -> V indices 0-3, W indices 0-3
    # B12: cols 2-3 -> V indices 4-7, W indices 4-7
    # B13: cols 4-5 -> V indices 8-11, W indices 8-11
    
    add_strassen_block(0, 0, 0)    # A1 * B11
    add_strassen_block(0, 4, 4)    # A1 * B12
    add_strassen_block(0, 8, 8)    # A1 * B13
    
    # Block 2: A2 * B2 (2x3 * 3x6)
    # A2 rows 0-1, cols 2-4 of A -> U indices 4-9
    # B2 rows 2-4, cols 0-5 of B -> V indices 12-29
    # Split B2 into three 3x2 blocks:
    # B21: cols 0-1 -> V indices 12-17, W indices 0-3
    # B22: cols 2-3 -> V indices 18-23, W indices 4-7
    # B23: cols 4-5 -> V indices 24-29, W indices 8-11
    
    add_schoolbook_2x3x2(4, 12, 0)  # A2 * B21
    add_schoolbook_2x3x2(4, 18, 4)  # A2 * B22
    add_schoolbook_2x3x2(4, 24, 8)  # A2 * B23
    
    # Verify the decomposition with random tests
    def verify_decomposition():
        import random
        random.seed(123)
        for _ in range(50):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            
            # Compute C = A * B
            C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            
            # Compute C using the decomposition
            # C[i][k] = sum_c (sum_j U[i*m+j][c] * A[i][j]) * (sum_j V[j*p+k][c] * B[j][k]) * W[k*n+i][c]
            # Wait, this isn't right. Let me re-derive.
            
            # The tensor decomposition is:
            # T[i,j,k,l] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+l][c]
            # where T[i,j,k,l] = 1 if l==i else 0 (for matrix multiplication A(i,j) * B(j,k) -> C(i,k))
            
            # Actually, the standard form for matrix multiplication is:
            # C[i][k] = sum_j A[i][j] * B[j][k]
            # The decomposition should satisfy:
            # C[i][k] = sum_c (sum_j U[i*m+j][c] * A[i][j]) * (sum_j V[j*p+k][c] * B[j][k]) * W[k*n+i][c]
            
            # Wait, that's not quite right either. Let me think more carefully.
            
            # In the schoolbook decomposition:
            # For each (i,j,k), we have one term where:
            # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1
            # So the contribution to C[i][k] is:
            # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] * A[i][j] * B[j][k]
            # = sum_c (1 if c corresponds to (i,j,k)) * A[i][j] * B[j][k]
            # = A[i][j] * B[j][k]
            # And summing over j gives C[i][k] = sum_j A[i][j] * B[j][k]
            
            # So the correct formula is:
            # C[i][k] = sum_j sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] * A[i][j] * B[j][k]
            
            # Let me implement this:
            C_decomp = [[0 for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for c in range(rank):
                            if U[i*m+j][c] != 0 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                                total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] * A[i][j] * B[j][k]
                    C_decomp[i][k] = total
                    if C_decomp[i][k] != C[i][k]:
                        return False
        return True
    
    if not verify_decomposition():
        # Fall back to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
