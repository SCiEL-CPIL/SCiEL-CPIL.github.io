import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition is exact."""
        # Check tensor identity: for all i,k,j,jp:
        # sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r] = delta[j,jp]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        total = 0.0
                        for r in range(R):
                            total += W[k * n + i][r] * U[i * m + j][r] * V[jp * p + k][r]
                        expected = 1.0 if j == jp else 0.0
                        if abs(total - expected) > 1e-9:
                            return False
        return True
    
    def build_hybrid_decomposition():
        """
        Build a hybrid decomposition using Strassen-like blocks.
        
        Strategy:
        - Use 2 Strassen 2x2x2 blocks to cover i in {0,1}, j in {0,1}, k in {0,1}
        - Use schoolbook for remaining combinations
        - This gives rank = 2*2 + 3*4*5 - 2*2*2 = 8 + 60 - 8 = 60... 
        - Actually let me think more carefully.
        
        Better approach: Partition the computation.
        - Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
        - Block 2: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2, use Strassen (rank 7)  
        - Block 3: i in {0,1}, j in {0,3}, k in {2,4} -> 2x2x2, use Strassen (rank 7)
        - Block 4: i in {0,1}, j in {2,3}, k in {2,4} -> 2x2x2, use Strassen (rank 7)
        - Block 5: i = {2}, all j, all k -> 1x4x5, schoolbook (rank 20)
        
        Total: 7+7+7+7+20 = 48
        """
        R = 48
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        
        col = 0
        
        # Strassen 2x2x2 template
        # For multiplying 2x2 matrices:
        # C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # Strassen:
        # m1 = (a-d)(e+h)
        # m2 = (a+d)e  
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(e+f)
        # m7 = (b-c)(g+h)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            """Add a Strassen 2x2x2 block for rows i1,i2, cols j1,j2, output k1,k2."""
            nonlocal col
            # m1 = (A[i1,j1] - A[i2,j2]) * (B[j1,k1] + B[j2,k2])
            # m2 = (A[i1,j1] + A[i2,j1]) * B[j1,k1]
            # m3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
            # m4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
            # m5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
            # m6 = (A[i2,j1] - A[i2,j2]) * (B[j1,k1] + B[j1,k2])
            # m7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
            
            # m1
            U[i1 * m + j1][col] = 1
            U[i2 * m + j2][col] = -1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k1 * n + i2][col] = 0
            W[k2 * n + i1][col] = 0
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m2
            U[i1 * m + j1][col] = 1
            U[i2 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + i2][col] = 1
            W[k2 * n + i2][col] = -1
            col += 1
            
            # m3
            U[i1 * m + j1][col] = 1
            V[j1 * p + k2][col] = 1
            V[j2 * p + k2][col] = -1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m4
            U[i2 * m + j2][col] = 1
            V[j2 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            W[k1 * n + i1][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m5
            U[i1 * m + j1][col] = 1
            U[i1 * m + j2][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = -1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m6
            U[i2 * m + j1][col] = 1
            U[i2 * m + j2][col] = -1
            V[j1 * p + k1][col] = 1
            V[j1 * p + k2][col] = 1
            W[k2 * n + i2][col] = 1
            col += 1
            
            # m7
            U[i1 * m + j2][col] = 1
            U[i2 * m + j2][col] = -1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + i1][col] = 1
            W[k1 * n + i2][col] = -1
            col += 1
        
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        add_strassen_block(0, 1, 0, 1, 0, 1)
        
        # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
        add_strassen_block(0, 1, 2, 3, 0, 1)
        
        # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        add_strassen_block(0, 1, 0, 1, 2, 3)
        
        # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
        add_strassen_block(0, 1, 2, 3, 2, 3)
        
        # Block 5: i = {2}, all j in {0,1,2,3}, all k in {0,1,2,3,4}
        # Schoolbook: 1 * 4 * 5 = 20 terms
        for j in range(4):
            for k in range(5):
                U[2 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 2][col] = 1
                col += 1
        
        assert col == R, f"Expected {R} columns, got {col}"
        
        # Verify
        if not verify_decomposition(U, V, W, R):
            return None
        return U, V, W, R
    
    # First, build the hybrid decomposition
    result = build_hybrid_decomposition()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(dim_u)]
    V = [[0] * R for _ in range(dim_v)]
    W = [[0] * R for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
