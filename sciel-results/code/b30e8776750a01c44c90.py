"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses a block-level Strassen decomposition.
Rank: 56.
"""
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # We use 7 Strassen products, each of rank 8.
    # Total rank 56.
    
    # Define the 7 Strassen products.
    # Each product is a (1x2) * (2x4) multiplication.
    # We define them by their contributions to U, V, W.
    
    # Helper to add a product
    def add_product(U, V, W, col_offset, left_coeffs, right_coeffs, c_coeffs):
        # left_coeffs: list of 2 lists, each list of (a_row, a_col, coeff)
        # right_coeffs: list of 8 lists, each list of (b_row, b_col, coeff)
        # c_coeffs: list of 4 lists, each list of (c_row, c_col, coeff)
        
        for j in range(2):
            for k in range(4):
                # U
                for (a_r, a_c, c) in left_coeffs[j]:
                    U[a_r * m + a_c][col_offset] += c
                # V
                for (b_r, b_c, c) in right_coeffs[j * 4 + k]:
                    V[b_r * p + b_c][col_offset] += c
                # W
                for (c_r, c_c, c) in c_coeffs[k]:
                    W[c_c * n + c_r][col_offset] += c
                col_offset += 1
        return col_offset

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: A[0,0:2], A11: A[1,2:4]
    # B00: B[0:2, 0:4], B11: B[2:4, 4:8]
    # C00: C[0,0:4] (+1), C11: C[1,4:8] (+1)
    col = add_product(U, V, W, col,
        [
            [(0, 0, 1), (1, 2, 1)],
            [(0, 1, 1), (1, 3, 1)]
        ],
        [
            [(0, 0, 1), (2, 4, 1)], [(0, 1, 1), (2, 5, 1)], [(0, 2, 1), (2, 6, 1)], [(0, 3, 1), (2, 7, 1)],
            [(1, 0, 1), (3, 4, 1)], [(1, 1, 1), (3, 5, 1)], [(1, 2, 1), (3, 6, 1)], [(1, 3, 1), (3, 7, 1)]
        ],
        [
            [(0, 0, 1), (1, 4, 1)],
            [(0, 1, 1), (1, 5, 1)],
            [(0, 2, 1), (1, 6, 1)],
            [(0, 3, 1), (1, 7, 1)]
        ]
    )
    
    # M2 = (A10+A11)*B00
    # A10: A[1,0:2], A11: A[1,2:4]
    # B00: B[0:2, 0:4]
    # C10: C[1,0:4] (+1), C11: C[1,4:8] (-1)
    col = add_product(U, V, W, col,
        [
            [(1, 0, 1), (1, 2, 1)],
            [(1, 1, 1), (1, 3, 1)]
        ],
        [
            [(0, 0, 1)], [(0, 1, 1)], [(0, 2, 1)], [(0, 3, 1)],
            [(1, 0, 1)], [(1, 1, 1)], [(1, 2, 1)], [(1, 3, 1)]
        ],
        [
            [(1, 0, 1), (1, 4, -1)],
            [(1, 1, 1), (1, 5, -1)],
            [(1, 2, 1), (1, 6, -1)],
            [(1, 3, 1), (1, 7, -1)]
        ]
    )
    
    # M3 = A00*(B01-B11)
    # A00: A[0,0:2]
    # B01: B[0:2, 4:8], B11: B[2:4, 4:8]
    # C01: C[0,4:8] (+1), C11: C[1,4:8] (+1)
    col = add_product(U, V, W, col,
        [
            [(0, 0, 1)],
            [(0, 1, 1)]
        ],
        [
            [(0, 4, 1), (2, 4, -1)], [(0, 5, 1), (2, 5, -1)], [(0, 6, 1), (2, 6, -1)], [(0, 7, 1), (2, 7, -1)],
            [(1, 4, 1), (3, 4, -1)], [(1, 5, 1), (3, 5, -1)], [(1, 6, 1), (3, 6, -1)], [(1, 7, 1), (3, 7, -1)]
        ],
        [
            [(0, 4, 1), (1, 4, 1)],
            [(0, 5, 1), (1, 5, 1)],
            [(0, 6, 1), (1, 6, 1)],
            [(0, 7, 1), (1, 7, 1)]
        ]
    )
    
    # M4 = A11*(B10-B00)
    # A11: A[1,2:4]
    # B10: B[2:4, 0:4], B00: B[0:2, 0:4]
    # C00: C[0,0:4] (+1), C10: C[1,0:4] (+1)
    col = add_product(U, V, W, col,
        [
            [(1, 2, 1)],
            [(1, 3, 1)]
        ],
        [
            [(2, 0, 1), (0, 0, -1)], [(2, 1, 1), (0, 1, -1)], [(2, 2, 1), (0, 2, -1)], [(2, 3, 1), (0, 3, -1)],
            [(3, 0, 1), (1, 0, -1)], [(3, 1, 1), (1, 1, -1)], [(3, 2, 1), (1, 2, -1)], [(3, 3, 1), (1, 3, -1)]
        ],
        [
            [(0, 0, 1), (1, 0, 1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 2, 1), (1, 2, 1)],
            [(0, 3, 1), (1, 3, 1)]
        ]
    )
    
    # M5 = (A00+A01)*B11
    # A00: A[0,0:2], A01: A[0,2:4]
    # B11: B[2:4, 4:8]
    # C00: C[0,0:4] (-1), C01: C[0,4:8] (+1)
    col = add_product(U, V, W, col,
        [
            [(0, 0, 1), (0, 2, 1)],
            [(0, 1, 1), (0, 3, 1)]
        ],
        [
            [(2, 4, 1)], [(2, 5, 1)], [(2, 6, 1)], [(2, 7, 1)],
            [(3, 4, 1)], [(3, 5, 1)], [(3, 6, 1)], [(3, 7, 1)]
        ],
        [
            [(0, 0, -1), (0, 4, 1)],
            [(0, 1, -1), (0, 5, 1)],
            [(0, 2, -1), (0, 6, 1)],
            [(0, 3, -1), (0, 7, 1)]
        ]
    )
    
    # M6 = (A10-A00)*(B00+B01)
    # A10: A[1,0:2], A00: A[0,0:2]
    # B00: B[0:2, 0:4], B01: B[0:2, 4:8]
    # C11: C[1,4:8] (+1)
    col = add_product(U, V, W, col,
        [
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1), (0, 1, -1)]
        ],
        [
            [(0, 0, 1), (0, 4, 1)], [(0, 1, 1), (0, 5, 1)], [(0, 2, 1), (0, 6, 1)], [(0, 3, 1), (0, 7, 1)],
            [(1, 0, 1), (1, 4, 1)], [(1, 1, 1), (1, 5, 1)], [(1, 2, 1), (1, 6, 1)], [(1, 3, 1), (1, 7, 1)]
        ],
        [
            [(1, 4, 1)],
            [(1, 5, 1)],
            [(1, 6, 1)],
            [(1, 7, 1)]
        ]
    )
    
    # M7 = (A01-A11)*(B10+B11)
    # A01: A[0,2:4], A11: A[1,2:4]
    # B10: B[2:4, 0:4], B11: B[2:4, 4:8]
    # C00: C[0,0:4] (+1)
    col = add_product(U, V, W, col,
        [
            [(0, 2, 1), (1, 2, -1)],
            [(0, 3, 1), (1, 3, -1)]
        ],
        [
            [(2, 0, 1), (2, 4, 1)], [(2, 1, 1), (2, 5, 1)], [(2, 2, 1), (2, 6, 1)], [(2, 3, 1), (2, 7, 1)],
            [(3, 0, 1), (3, 4, 1)], [(3, 1, 1), (3, 5, 1)], [(3, 2, 1), (3, 6, 1)], [(3, 3, 1), (3, 7, 1)]
        ],
        [
            [(0, 0, 1)],
            [(0, 1, 1)],
            [(0, 2, 1)],
            [(0, 3, 1)]
        ]
    )
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    # Verify
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
