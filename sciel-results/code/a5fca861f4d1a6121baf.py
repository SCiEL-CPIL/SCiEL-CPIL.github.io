"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses Strassen's rank-7 decomposition for 2x2 matrix multiplication, applied
to the block structure of the 2x4 * 4x8 multiplication.

The 2x4x8 multiplication is decomposed into 7 Strassen products, each of which
is a (1x2)*(2x4) product requiring rank 8. Total rank = 7 * 8 = 56.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Define the 7 Strassen products for the 2x2 block structure
    # A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    #   A00 = A[0,0:2], A01 = A[0,2:4], A10 = A[1,0:2], A11 = A[1,2:4]
    # B is 4x8: B[j,k] where j in {0,1,2,3}, k in {0,...,7}
    #   B00 = B[0:2, 0:4], B01 = B[0:2, 4:8], B10 = B[2:4, 0:4], B11 = B[2:4, 4:8]
    # C is 2x8: C[i,k] where i in {0,1}, k in {0,...,7}
    #   C00 = C[0,0:4], C01 = C[0,4:8], C10 = C[1,0:4], C11 = C[1,4:8]
    
    # Each Strassen product M = L * R where L is 1x2 and R is 2x4:
    #   M[0,k] = L[0]*R[0,k] + L[1]*R[1,k]
    # Schoolbook rank for each: 1*2*4 = 8 terms
    
    products = []
    
    # M1 = (A00+A11)*(B00+B11)
    products.append({
        'left': [
            [(0, 0, 1), (1, 2, 1)],
            [(0, 1, 1), (1, 3, 1)],
        ],
        'right': [
            [(0, 0, 1), (2, 4, 1)],
            [(0, 1, 1), (2, 5, 1)],
            [(0, 2, 1), (2, 6, 1)],
            [(0, 3, 1), (2, 7, 1)],
            [(1, 0, 1), (3, 4, 1)],
            [(1, 1, 1), (3, 5, 1)],
            [(1, 2, 1), (3, 6, 1)],
            [(1, 3, 1), (3, 7, 1)],
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 4, 1)],
            [(0, 1, 1), (1, 5, 1)],
            [(0, 2, 1), (1, 6, 1)],
            [(0, 3, 1), (1, 7, 1)],
        ]
    })
    
    # M2 = (A10+A11)*B00
    products.append({
        'left': [
            [(1, 0, 1), (1, 2, 1)],
            [(1, 1, 1), (1, 3, 1)],
        ],
        'right': [
            [(0, 0, 1)],
            [(0, 1, 1)],
            [(0, 2, 1)],
            [(0, 3, 1)],
            [(1, 0, 1)],
            [(1, 1, 1)],
            [(1, 2, 1)],
            [(1, 3, 1)],
        ],
        'c_contrib': [
            [(1, 0, 1), (1, 4, -1)],
            [(1, 1, 1), (1, 5, -1)],
            [(1, 2, 1), (1, 6, -1)],
            [(1, 3, 1), (1, 7, -1)],
        ]
    })
    
    # M3 = A00*(B01-B11)
    products.append({
        'left': [
            [(0, 0, 1)],
            [(0, 1, 1)],
        ],
        'right': [
            [(0, 4, 1), (2, 4, -1)],
            [(0, 5, 1), (2, 5, -1)],
            [(0, 6, 1), (2, 6, -1)],
            [(0, 7, 1), (2, 7, -1)],
            [(1, 4, 1), (3, 4, -1)],
            [(1, 5, 1), (3, 5, -1)],
            [(1, 6, 1), (3, 6, -1)],
            [(1, 7, 1), (3, 7, -1)],
        ],
        'c_contrib': [
            [(0, 4, 1), (1, 4, 1)],
            [(0, 5, 1), (1, 5, 1)],
            [(0, 6, 1), (1, 6, 1)],
            [(0, 7, 1), (1, 7, 1)],
        ]
    })
    
    # M4 = A11*(B10-B00)
    products.append({
        'left': [
            [(1, 2, 1)],
            [(1, 3, 1)],
        ],
        'right': [
            [(2, 0, 1), (0, 0, -1)],
            [(2, 1, 1), (0, 1, -1)],
            [(2, 2, 1), (0, 2, -1)],
            [(2, 3, 1), (0, 3, -1)],
            [(3, 0, 1), (1, 0, -1)],
            [(3, 1, 1), (1, 1, -1)],
            [(3, 2, 1), (1, 2, -1)],
            [(3, 3, 1), (1, 3, -1)],
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 0, 1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 2, 1), (1, 2, 1)],
            [(0, 3, 1), (1, 3, 1)],
        ]
    })
    
    # M5 = (A00+A01)*B11
    products.append({
        'left': [
            [(0, 0, 1), (0, 2, 1)],
            [(0, 1, 1), (0, 3, 1)],
        ],
        'right': [
            [(2, 4, 1)],
            [(2, 5, 1)],
            [(2, 6, 1)],
            [(2, 7, 1)],
            [(3, 4, 1)],
            [(3, 5, 1)],
            [(3, 6, 1)],
            [(3, 7, 1)],
        ],
        'c_contrib': [
            [(0, 0, -1), (0, 4, 1)],
            [(0, 1, -1), (0, 5, 1)],
            [(0, 2, -1), (0, 6, 1)],
            [(0, 3, -1), (0, 7, 1)],
        ]
    })
    
    # M6 = (A10-A00)*(B00+B01)
    products.append({
        'left': [
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1), (0, 1, -1)],
        ],
        'right': [
            [(0, 0, 1), (0, 4, 1)],
            [(0, 1, 1), (0, 5, 1)],
            [(0, 2, 1), (0, 6, 1)],
            [(0, 3, 1), (0, 7, 1)],
            [(1, 0, 1), (1, 4, 1)],
            [(1, 1, 1), (1, 5, 1)],
            [(1, 2, 1), (1, 6, 1)],
            [(1, 3, 1), (1, 7, 1)],
        ],
        'c_contrib': [
            [(1, 4, 1)],
            [(1, 5, 1)],
            [(1, 6, 1)],
            [(1, 7, 1)],
        ]
    })
    
    # M7 = (A01-A11)*(B10+B11)
    products.append({
        'left': [
            [(0, 2, 1), (1, 2, -1)],
            [(0, 3, 1), (1, 3, -1)],
        ],
        'right': [
            [(2, 0, 1), (2, 4, 1)],
            [(2, 1, 1), (2, 5, 1)],
            [(2, 2, 1), (2, 6, 1)],
            [(2, 3, 1), (2, 7, 1)],
            [(3, 0, 1), (3, 4, 1)],
            [(3, 1, 1), (3, 5, 1)],
            [(3, 2, 1), (3, 6, 1)],
            [(3, 3, 1), (3, 7, 1)],
        ],
        'c_contrib': [
            [(0, 0, 1)],
            [(0, 1, 1)],
            [(0, 2, 1)],
            [(0, 3, 1)],
        ]
    })
    
    # Build U, V, W
    rank = 7 * 8  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        for j in range(2):
            for k in range(4):
                for (a_row, a_col, coeff) in prod['left'][j]:
                    U[a_row * m + a_col][col] += coeff
                for (b_row, b_col, coeff) in prod['right'][j * 4 + k]:
                    V[b_row * p + b_col][col] += coeff
                for (c_row, c_col, coeff) in prod['c_contrib'][k]:
                    W[c_col * n + c_row][col] += coeff
                col += 1
    
    # Deadline guard: return immediately since construction is O(1)
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp[i,k] = total
    
    error = np.max(np.abs(C_true - C_decomp))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
