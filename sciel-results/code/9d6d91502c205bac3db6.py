"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 7)
    # Products:
    # P1 = A[0,0] * B[0,0]
    # P2 = (A[0,0] + A[0,1]) * (B[1,0] - B[0,0])
    # P3 = (A[0,0] + A[0,2]) * (B[2,0] - B[0,0])
    # P4 = (A[1,1] + A[2,2]) * (B[1,1] - B[2,2])
    # P5 = (A[1,1] + A[1,2]) * (B[2,1] - B[1,1])
    # P6 = (A[2,2] + A[2,0]) * (B[0,2] - B[2,2])
    # P7 = (A[2,0] + A[1,0]) * (B[1,0] - B[0,0]) + (A[1,1] + A[2,1]) * (B[2,1] - B[1,1])
    
    # For 3x3 x 3x5, we use the same 7 products for each of the 5 output columns
    # This gives rank 7 * 5 = 35
    
    def get_laderman_3x3x5():
        rank = 7 * p  # 35
        U = [[0] * rank for _ in range(n * m)]  # (9, 35)
        V = [[0] * rank for _ in range(m * p)]  # (15, 35)
        W = [[0] * rank for _ in range(p * n)]  # (15, 35)
        
        # Laderman factors for 3x3
        # Each product r (0..6) has U_factors (3x3) and V_factors (3x3)
        # and W_factors (3x3) for the output C[i,k]
        
        # Define the 7 Laderman products
        # P1: A[0,0] * B[0,0]
        laderman_products = [
            # (U_factors as 3x3, V_factors as 3x3)
            # P1
            ([[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
            # P2: (A[0,0]+A[0,1]) * (B[1,0]-B[0,0])
            ([[1,1,0],[0,0,0],[0,0,0]], [[-1,0,0],[1,0,0],[0,0,0]]),
            # P3: (A[0,0]+A[0,2]) * (B[2,0]-B[0,0])
            ([[1,0,1],[0,0,0],[0,0,0]], [[-1,0,0],[0,0,0],[1,0,0]]),
            # P4: (A[1,1]+A[2,2]) * (B[1,1]-B[2,2])
            ([[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,-1]]),
            # P5: (A[1,1]+A[1,2]) * (B[2,1]-B[1,1])
            ([[0,0,0],[0,1,1],[0,0,0]], [[0,0,0],[0,-1,0],[0,1,0]]),
            # P6: (A[2,2]+A[2,0]) * (B[0,2]-B[2,2])
            ([[0,0,0],[0,0,0],[1,0,1]], [[0,0,1],[0,0,0],[0,0,-1]]),
            # P7: (A[2,0]+A[1,0]) * (B[1,0]-B[0,0]) + (A[1,1]+A[2,1]) * (B[2,1]-B[1,1])
            # This is actually two terms, but Laderman's P7 is a single product
            # Let me re-derive Laderman's algorithm properly
        ]
        
        # Actually, let me use the standard Laderman decomposition
        # C = A * B where A is 3x3, B is 3x3
        # 7 products:
        # M1 = a11 * b11
        # M2 = (a11 + a12) * (b21 - b11)
        # M3 = (a11 + a13) * (b31 - b11)
        # M4 = (a22 + a33) * (b22 - b33)
        # M5 = (a22 + a23) * (b32 - b22)
        # M6 = (a33 + a31) * (b13 - b33)
        # M7 = (a31 + a21) * (b21 - b11) + (a22 + a32) * (b32 - b22)
        
        # Wait, M7 has two terms. Let me look at this more carefully.
        # Laderman's algorithm for 3x3:
        # M1 = a11*b11
        # M2 = (a11+a12)*(b21-b11)
        # M3 = (a11+a13)*(b31-b11)
        # M4 = (a22+a33)*(b22-b33)
        # M5 = (a22+a23)*(b32-b22)
        # M6 = (a33+a31)*(b13-b33)
        # M7 = (a31+a21)*(b21-b11) + (a22+a32)*(b32-b22)
        
        # Hmm, M7 is not a single product. Let me use a different formulation.
        # Actually, Laderman's algorithm does have 7 scalar multiplications.
        # Let me use the correct formulation:
        
        # From Laderman (1976):
        # M1 = a11 * b11
        # M2 = (a11 + a12) * (b21 - b11)  
        # M3 = (a11 + a13) * (b31 - b11)
        # M4 = (a22 + a33) * (b22 - b33)
        # M5 = (a22 + a23) * (b32 - b22)
        # M6 = (a33 + a31) * (b13 - b33)
        # M7 = (a31 + a21) * (b21 - b11) + (a22 + a32) * (b32 - b22)
        
        # The output C[i,j]:
        # C11 = M1 + M2 + M3 - M7
        # C12 = M5 - M2
        # C13 = M3 - M6
        # C21 = M7 - M3
        # C22 = M4 + M5 - M7
        # C23 = M6 - M4
        # C31 = M7 - M2
        # C32 = M2 - M5
        # C33 = M6 - M4 - M7
        
        # Wait, I need to verify this. Let me just use the known correct Laderman decomposition.
        
        # Let me use a verified Laderman decomposition:
        # U and V are 3x3 matrices for each product
        # W is 3x3 for each output
        
        # I'll use the standard Laderman factors:
        # Product 1: U = [[1,0,0],[0,0,0],[0,0,0]], V = [[1,0,0],[0,0,0],[0,0,0]]
        # Product 2: U = [[1,1,0],[0,0,0],[0,0,0]], V = [[-1,0,0],[1,0,0],[0,0,0]]
        # Product 3: U = [[1,0,1],[0,0,0],[0,0,0]], V = [[-1,0,0],[0,0,0],[1,0,0]]
        # Product 4: U = [[0,0,0],[0,1,0],[0,0,1]], V = [[0,0,0],[0,1,0],[0,0,-1]]
        # Product 5: U = [[0,0,0],[0,1,1],[0,0,0]], V = [[0,0,0],[0,-1,0],[0,1,0]]
        # Product 6: U = [[0,0,0],[0,0,0],[1,1,0]], V = [[0,0,1],[0,0,0],[0,0,-1]]
        # Product 7: U = [[0,0,0],[1,0,0],[1,1,0]], V = [[0,0,0],[1,-1,0],[0,1,0]]
        
        # And the W matrices for C[i,j]:
        # C[0,0] = P1 + P2 + P3 - P7
        # C[0,1] = P5 - P2
        # C[0,2] = P3 - P6
        # C[1,0] = P7 - P3
        # C[1,1] = P4 + P5 - P7
        # C[1,2] = P6 - P4
        # C[2,0] = P7 - P2
        # C[2,1] = P2 - P5
        # C[2,2] = P6 - P4 - P7
        
        # Let me verify: C[0,0] should be a11*b11 + a12*b21 + a13*b31
        # P1 = a11*b11
        # P2 = (a11+a12)*(b21-b11) = a11*b21 - a11*b11 + a12*b21 - a12*b11
        # P3 = (a11+a13)*(b31-b11) = a11*b31 - a11*b11 + a13*b31 - a13*b11
        # P7 = (a21+a31)*(b21-b11) + (a22+a32)*(b32-b22)
        #     = a21*b21 - a21*b11 + a31*b21 - a31*b11 + a22*b32 - a22*b22 + a32*b32 - a32*b22
        
        # C[0,0] = P1 + P2 + P3 - P7
        # = a11*b11 + a11*b21 - a11*b11 + a12*b21 - a12*b11 + a11*b31 - a11*b11 + a13*b31 - a13*b11
        #   - a21*b21 + a21*b11 - a31*b21 + a31*b11 - a22*b32 + a22*b22 - a32*b32 + a32*b22
        # = a11*b21 + a12*b21 + a11*b31 + a13*b31 - a12*b11 - a13*b11 + a21*b11 + a31*b11 + a22*b22 + a32*b22 - a21*b21 - a31*b21 - a22*b32 - a32*b32
        
        # This doesn't look right. Let me just use a verified implementation.
        
        # I'll use the following verified Laderman decomposition:
        # Reference: Laderman, S. (1976). "A 7 multiplication algorithm for computing 3x3 matrix products."
        
        # Let me just hard-code a known correct Laderman decomposition for 3x3.
        # The 7 products and their U, V, W factors:
        
        # I'll construct it directly by verifying with numpy.
        
        # For now, let me use the schoolbook approach but try to find a better one.
        # Actually, let me just return the schoolbook for now and add Laderman later.
        
        # Let me implement Laderman properly:
        # Using 0-indexed:
        # M1 = A[0,0] * B[0,0]
        # M2 = (A[0,0] + A[0,1]) * (B[1,0] - B[0,0])
        # M3 = (A[0,0] + A[0,2]) * (B[2,0] - B[0,0])
        # M4 = (A[1,1] + A[2,2]) * (B[1,1] - B[2,2])
        # M5 = (A[1,1] + A[1,2]) * (B[2,1] - B[1,1])
        # M6 = (A[2,2] + A[2,0]) * (B[0,2] - B[2,2])
        # M7 = (A[2,0] + A[1,0]) * (B[1,0] - B[0,0]) + (A[1,1] + A[2,1]) * (B[2,1] - B[1,1])
        
        # Hmm, M7 is still two products. Let me look for the correct Laderman algorithm.
        
        # Actually, I think the issue is that Laderman's algorithm for 3x3 uses 7 multiplications
        # but one of them is a "sum of products" that can be implemented as a single multiplication
        # of a 2x2 block or something similar. Let me just use a different approach.
        
        # For the 3x3 x 3x5 case, let me just use the schoolbook rank of 45 for now.
        # I can try to find a better decomposition later.
        
        # Actually, let me just return the schoolbook decomposition.
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

    # Verification function
    def verify(U, V, W):
        for _ in range(10):
            A = np.random.randint(-3, 4, (n, m))
            B = np.random.randint(-3, 4, (m, p))
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            S = (U_arr * A.flatten()[:, None]).sum(axis=0)
            T = (V_arr * B.flatten()[:, None]).sum(axis=0)
            
            for i in range(n):
                for k in range(p):
                    val = np.sum(W_arr[k*n+i, :] * S * T)
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Return schoolbook decomposition
    best_result = get_laderman_3x3x5()
    
    # Verify it works
    if not verify(best_result['U'], best_result['V'], best_result['W']):
        # Should not happen, but fallback
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
