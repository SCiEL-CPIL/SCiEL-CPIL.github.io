"""Tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]


def construct():
    start_time = time.time()
    deadline = start_time + 130  # leave 20s margin for 150s limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 60

    # Build schoolbook decomposition as baseline
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    # Try to find a lower-rank decomposition using Strassen-style
    # block decomposition for the 2x2x2 sub-block
    # The idea: replace 8 rank-1 terms (i,j,k in {0,1} x {0,1} x {0,1})
    # with 7 rank-1 terms using Strassen's algorithm
    
    # Strassen for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M1 - M2 + M3 + M6

    # For our tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication entry
    # In the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}), all entries are 1
    # We need to verify that Strassen gives the correct result
    
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # So in the 2x2x2 block, we need sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all (i,j,k) in {0,1}^3
    
    # Let's construct the 7 Strassen terms for the 2x2x2 block
    # and verify they sum to 1 for all 8 entries
    
    # Strassen factors (using half-integers where needed):
    # For the 2x2x2 block with indices (i,j,k) in {0,1}^3:
    
    strassen_factors = [
        # (U values for (i,j), V values for (j,k), W values for (k,i))
        # M1: (A00+A11)(B00+B11)
        # U: i=0,j=0: 1; i=1,j=1: 1; others: 0
        # V: j=0,k=0: 1; j=1,k=1: 1; others: 0
        # W: k=0,i=0: 1; k=1,i=1: 1; others: 0
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        # M2: (A10+A11)B00
        # U: i=1,j=0: 1; i=1,j=1: 1; others: 0
        # V: j=0,k=0: 1; others: 0
        # W: k=0,i=0: 1; k=0,i=1: 1; others: 0
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,0): 1, (0,1): 1}
        },
        # M3: A00(B10-B11)
        # U: i=0,j=0: 1; others: 0
        # V: j=1,k=0: 1; j=1,k=1: -1; others: 0
        # W: k=0,i=0: 1; k=1,i=0: 1; others: 0
        {
            'U': {(0,0): 1},
            'V': {(1,0): 1, (1,1): -1},
            'W': {(0,0): 1, (1,0): 1}
        },
        # M4: A11(B01-B10)
        # U: i=1,j=1: 1; others: 0
        # V: j=0,k=1: 1; j=1,k=0: -1; others: 0
        # W: k=1,i=1: 1; k=0,i=1: 1; others: 0
        {
            'U': {(1,1): 1},
            'V': {(0,1): 1, (1,0): -1},
            'W': {(1,1): 1, (0,1): 1}
        },
        # M5: (A00+A10)B11
        # U: i=0,j=0: 1; i=1,j=0: 1; others: 0
        # V: j=1,k=1: 1; others: 0
        # W: k=1,i=0: 1; k=1,i=1: 1; others: 0
        {
            'U': {(0,0): 1, (1,0): 1},
            'V': {(1,1): 1},
            'W': {(1,0): 1, (1,1): 1}
        },
        # M6: (A11-A01)(B00+B10)
        # U: i=1,j=1: 1; i=0,j=1: -1; others: 0
        # V: j=0,k=0: 1; j=1,k=0: 1; others: 0
        # W: k=0,i=0: 1; k=0,i=1: 1; others: 0
        {
            'U': {(1,1): 1, (0,1): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1, (0,1): 1}
        },
        # M7: (A01-A11)(B01+B11)
        # U: i=0,j=1: 1; i=1,j=1: -1; others: 0
        # V: j=0,k=1: 1; j=1,k=1: 1; others: 0
        # W: k=1,i=0: 1; k=1,i=1: 1; others: 0
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(1,0): 1, (1,1): 1}
        },
    ]
    
    # Verify Strassen decomposition for 2x2x2 block
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for f in strassen_factors:
                        u = f['U'].get((i, j), 0)
                        v = f['V'].get((j, k), 0)
                        w = f['W'].get((k, i), 0)
                        total += u * v * w
                    if total != 1:
                        return False, (i, j, k), total
        return True, None, None
    
    verified, fail_idx, fail_val = verify_strassen()
    
    if verified:
        # Build rank 59 decomposition: 52 schoolbook terms (excluding 2x2x2 block) + 7 Strassen terms
        # The 2x2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}
        # These are 8 terms in the schoolbook: indices where i<2, j<2, k<2
        
        rank = base_rank - 8 + 7  # 59
        
        U_new = [[0] * rank for _ in range(n * m)]
        V_new = [[0] * rank for _ in range(m * p)]
        W_new = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Add schoolbook terms EXCEPT those in the 2x2x2 block
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue  # Skip the 2x2x2 block
                    U_new[i * m + j][col] = 1
                    V_new[j * p + k][col] = 1
                    W_new[k * n + i][col] = 1
                    col += 1
        
        # Add Strassen terms
        for f in strassen_factors:
            for (i, j), u in f['U'].items():
                if u != 0:
                    U_new[i * m + j][col] = u
            for (j, k), v in f['V'].items():
                if v != 0:
                    V_new[j * p + k][col] = v
            for (k, i), w in f['W'].items():
                if w != 0:
                    W_new[k * n + i][col] = w
            col += 1
        
        # Verify the full decomposition
        def verify_full(Umat, Vmat, Wmat, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(len(Umat[0])):
                            total += Umat[i * m + j][c] * Vmat[j * p + k][c] * Wmat[k * n + i][c]
                        if total != 1:
                            return False, (i, j, k), total
            return True, None, None
        
        full_verified, _, _ = verify_full(U_new, V_new, W_new, n, m, p)
        
        if full_verified:
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
