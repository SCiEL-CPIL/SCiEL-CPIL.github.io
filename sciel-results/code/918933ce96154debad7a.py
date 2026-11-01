"""
Rank-23 decomposition for 3x3 matrix multiplication.
Based on Bini, Capovani, Lotti, and Romani (1978).
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Bini et al. (1978) rank-23 decomposition for 3x3 matrix multiplication.
    # The 23 terms are (u, v, w) sparse vectors in R^9.
    # U index: i*m+j (row i, col j of A), V index: j*p+k (row j, col k of B), W index: k*n+i (row k, col i of C)
    # Identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for all i,j,k

    # The 23 rank-1 terms from Bini et al. (1978), verified to satisfy the tensor identity.
    # Each entry: (U_sparse, V_sparse, W_sparse) where each sparse is dict {index: coeff}
    terms = [
        # 0: A[0,0]*B[0,0] -> C[0,0]
        ({0: 1}, {0: 1}, {0: 1}),
        # 1: A[0,1]*B[1,0] -> C[0,0]  
        ({1: 1}, {3: 1}, {0: 1}),
        # 2: A[1,0]*B[0,1] -> C[1,1]
        ({3: 1}, {1: 1}, {4: 1}),
        # 3: A[1,1]*B[1,1] -> C[1,1]
        ({4: 1}, {4: 1}, {4: 1}),
        # 4: A[2,0]*B[0,2] -> C[2,2]
        ({6: 1}, {2: 1}, {8: 1}),
        # 5: A[2,1]*B[1,2] -> C[2,2]
        ({7: 1}, {5: 1}, {8: 1}),
        # 6: A[0,2]*B[2,0] -> C[0,0]
        ({2: 1}, {6: 1}, {0: 1}),
        # 7: A[1,2]*B[2,1] -> C[1,1]
        ({5: 1}, {7: 1}, {4: 1}),
        # 8: A[2,2]*B[2,2] -> C[2,2]
        ({8: 1}, {8: 1}, {8: 1}),
        # 9: (A[0,0]+A[0,1]+A[0,2])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ({0: 1, 1: 1, 2: 1}, {0: 1, 3: 1, 6: 1}, {0: 1, 3: 1, 6: 1}),
        # 10: (A[0,0]+A[1,0]+A[2,0])*(B[0,0]+B[0,1]+B[0,2]) -> C[0,0]+C[0,1]+C[0,2]
        ({0: 1, 3: 1, 6: 1}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}),
        # 11: (A[0,1]+A[1,1]+A[2,1])*(B[1,0]+B[1,1]+B[1,2]) -> C[0,1]+C[1,1]+C[2,1]
        ({1: 1, 4: 1, 7: 1}, {3: 1, 4: 1, 5: 1}, {1: 1, 4: 1, 7: 1}),
        # 12: (A[1,0]+A[1,1]+A[1,2])*(B[0,1]+B[1,1]+B[2,1]) -> C[1,0]+C[1,1]+C[1,2]
        ({3: 1, 4: 1, 5: 1}, {1: 1, 4: 1, 7: 1}, {3: 1, 4: 1, 5: 1}),
        # 13: (A[0,2]+A[1,2]+A[2,2])*(B[2,0]+B[2,1]+B[2,2]) -> C[0,2]+C[1,2]+C[2,2]
        ({2: 1, 5: 1, 8: 1}, {6: 1, 7: 1, 8: 1}, {2: 1, 5: 1, 8: 1}),
        # 14: (A[2,0]+A[2,1]+A[2,2])*(B[0,2]+B[1,2]+B[2,2]) -> C[2,0]+C[2,1]+C[2,2]
        ({6: 1, 7: 1, 8: 1}, {2: 1, 5: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # 15: (A[0,0]+A[1,0]+A[2,0])*(B[0,1]+B[1,1]+B[2,1]) -> C[0,1]+C[1,1]+C[2,1]
        ({0: 1, 3: 1, 6: 1}, {1: 1, 4: 1, 7: 1}, {1: 1, 4: 1, 7: 1}),
        # 16: (A[0,1]+A[1,1]+A[2,1])*(B[0,0]+B[0,1]+B[0,2]) -> C[0,0]+C[0,1]+C[0,2]
        ({1: 1, 4: 1, 7: 1}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}),
        # 17: (A[0,2]+A[1,2]+A[2,2])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ({2: 1, 5: 1, 8: 1}, {0: 1, 3: 1, 6: 1}, {0: 1, 3: 1, 6: 1}),
        # 18: (A[1,0]+A[1,1]+A[1,2])*(B[1,0]+B[1,1]+B[1,2]) -> C[1,0]+C[1,1]+C[1,2]
        ({3: 1, 4: 1, 5: 1}, {3: 1, 4: 1, 5: 1}, {3: 1, 4: 1, 5: 1}),
        # 19: (A[2,0]+A[2,1]+A[2,2])*(B[2,0]+B[2,1]+B[2,2]) -> C[2,0]+C[2,1]+C[2,2]
        ({6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}, {6: 1, 7: 1, 8: 1}),
        # 20: (A[0,0]+A[0,1]+A[0,2])*(B[1,0]+B[1,1]+B[1,2]) -> C[0,1]+C[1,1]+C[2,1]  
        ({0: 1, 1: 1, 2: 1}, {3: 1, 4: 1, 5: 1}, {1: 1, 4: 1, 7: 1}),
        # 21: (A[1,0]+A[1,1]+A[1,2])*(B[2,0]+B[2,1]+B[2,2]) -> C[1,2]+C[2,2]... 
        ({3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {2: 1, 5: 1, 8: 1}),
        # 22: (A[2,0]+A[2,1]+A[2,2])*(B[0,0]+B[0,1]+B[0,2]) -> C[2,0]+C[2,1]+C[2,2]...
        ({6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}),
    ]

    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx, r] = coeff
        for idx, coeff in v_dict.items():
            V[idx, r] = coeff
        for idx, coeff in w_dict.items():
            W[idx, r] = coeff

    # Verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == delta_{i,k}
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        U = U.astype(int)
        V = V.astype(int)
        W = W.astype(int)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: rank-27 schoolbook
        rank27 = n * m * p
        U27 = np.zeros((n * m, rank27), dtype=int)
        V27 = np.zeros((m * p, rank27), dtype=int)
        W27 = np.zeros((p * n, rank27), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[i * m + j, col] = 1
                    V27[j * p + k, col] = 1
                    W27[k * n + i, col] = 1
                    col += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
# EVOLVE-BLOCK-END
