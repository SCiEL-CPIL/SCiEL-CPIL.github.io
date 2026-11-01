import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, check over GF(3) whether rank-10 is possible
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1
    # Flatten to a vector in R^{n*m * m*p * p*n} but we can use the matrix form
    # T reshaped: for each output (k,i), the coefficient of A[i,j]*B[j,k] is 1
    # Matrix form: T is a 3-tensor, we can view it as a matrix of size (n*p) x (m*m*p) 
    # Actually, let's use the standard approach: build the constraint matrix
    
    # The tensor decomposition rank-r means:
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
    # where i in [0,n), j in [0,m), k in [0,p), l in [0,n)
    # For matrix multiplication: T[i,j,k,l] = 1 if l==i (output C[k,i] gets A[i,j]*B[j,k])
    
    # Over GF(3), we check: does there exist rank-10?
    # Fix the support pattern from rank-11 and see if we can solve with 10 terms
    
    # Build the tensor as a matrix over GF(3)
    # For each (i,j,k,l) with T=1, we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+l,r] = 1
    # This is nonlinear. Instead, use the linear code approach:
    # Fix W (or U,V) and solve linear system for the others
    
    # Simpler approach: enumerate which 10 of the 11 rank-1 terms to keep
    # and check if the tensor is in their span over GF(3)
    
    def gf3_rank_check(rank_to_check):
        """Check if tensor can be decomposed with given rank over GF(3)
        by trying subsets of the known rank-11 decomposition."""
        if time.time() > deadline - 5:
            return False
        
        # The 11 rank-1 terms from the known solution
        # Each term r has (U_r, V_r, W_r) vectors
        # We need to check if tensor is in span of 10 of these 11 terms
        
        # Build the 11 rank-1 tensors as vectors in GF(3)
        # Tensor dimension: n*m * m*p * p*n = 4*6*6 = 144
        # But many entries are zero; only 24 entries are 1
        
        # For each of the 11 terms, compute its contribution vector
        # term_r[(i,j,k,l)] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+l]
        
        # The target tensor: T[(i,j,k,l)] = 1 if l==i else 0
        # So T has 24 ones (for each i,j,k: T[i,j,k,i]=1)
        
        # Build matrix A where A[:,r] is the r-th term as a vector in GF(3)^{24}
        # We only need the 24 nonzero positions
        
        # Positions where T=1: (i,j,k,l) with l=i, so (i,j,k,i)
        # Index: ((i*m+j) * (m*p) + (j*p+k)) * (p*n) + (k*n+i)
        # But we can just use the 24 positions indexed by (i,j,k)
        
        # For each term r, compute its values at the 24 positions
        # U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # Let me define the 11 terms explicitly over GF(3)
        # U vectors (length 4), V vectors (length 6), W vectors (length 6)
        
        terms_U = [
            [1,0,1,0],  # r=0: A00+A11
            [0,0,1,1],  # r=1: A10+A11
            [1,0,0,0],  # r=2: A00
            [0,0,0,1],  # r=3: A11
            [1,1,0,0],  # r=4: A00+A01
            [2,0,1,0],  # r=5: A10-A00 = -A00+A10, in GF(3): 2,0,1,0
            [0,1,0,2],  # r=6: A01-A11, in GF(3): 0,1,0,2
            [1,0,0,0],  # r=7: A00
            [0,1,0,0],  # r=8: A01
            [0,0,1,0],  # r=9: A10
            [0,0,0,1],  # r=10: A11
        ]
        terms_V = [
            [1,0,0,1,0,0],  # r=0: B00+B11
            [1,0,0,0,0,0],  # r=1: B00
            [0,1,0,2,0,0],  # r=2: B01-B11, GF(3): 0,1,0,2,0,0
            [2,0,0,1,0,0],  # r=3: B10-B00, GF(3): 2,0,0,1,0,0
            [0,0,0,0,1,0],  # r=4: B11
            [1,1,0,0,0,0],  # r=5: B00+B01
            [0,0,0,1,1,0],  # r=6: B10+B11
            [0,0,1,0,0,0],  # r=7: B02
            [0,0,0,0,0,1],  # r=8: B12
            [0,0,1,0,0,0],  # r=9: B02
            [0,0,0,0,0,1],  # r=10: B12
        ]
        terms_W = [
            [1,0,0,0,1,0],  # r=0: C00+C11
            [0,1,0,0,0,2],  # r=1: C10-C11, GF(3): 0,1,0,0,0,2
            [0,0,1,0,1,0],  # r=2: C01+C11
            [1,0,0,1,0,0],  # r=3: C00+C10
            [2,0,1,0,0,0],  # r=4: -C00+C01, GF(3): 2,0,1,0,0,0
            [0,0,0,0,0,1],  # r=5: C11
            [1,0,0,0,0,0],  # r=6: C00
            [0,0,1,0,0,0],  # r=7: C02
            [0,0,1,0,0,0],  # r=8: C02
            [0,0,0,0,1,0],  # r=9: C12
            [0,0,0,0,1,0],  # r=10: C12
        ]
        
        # Target tensor values at 24 positions (i,j,k) -> T[i,j,k,i] = 1
        # For each term r, compute its value at position (i,j,k):
        # val_r(i,j,k) = terms_U[r][i*m+j] * terms_V[r][j*p+k] * terms_W[r][k*n+i]
        
        # Build 24 x 11 matrix M over GF(3)
        M = np.zeros((24, 11), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pos = (i * m + j) * (m * p) + (j * p + k)  # not quite, let me reindex
                    # Actually position index for (i,j,k) is just i*m*p + j*p + k = i*6 + j*3 + k
                    pos = i * m * p + j * p + k
                    for r in range(11):
                        u_val = terms_U[r][i * m + j]
                        v_val = terms_V[r][j * p + k]
                        w_val = terms_W[r][k * n + i]
                        M[pos, r] = (u_val * v_val * w_val) % 3
        
        # Target vector t over GF(3): all ones
        t = np.ones(24, dtype=np.int64)
        
        # Check if t is in the column space of M with at most 10 columns
        # Try all C(11,10) = 11 subsets
        from itertools import combinations
        
        for skip in range(11):
            cols = [c for c in range(11) if c != skip]
            M_sub = M[:, cols]
            # Solve M_sub * x = t over GF(3)
            # Use Gaussian elimination over GF(3)
            A = M_sub.copy()
            b = t.copy()
            
            # Gaussian elimination
            rank = 0
            for col in range(10):
                # Find pivot
                pivot = -1
                for row in range(rank, 24):
                    if A[row, col] % 3 != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap
                A[[rank, pivot]] = A[[pivot, rank]]
                b[[rank, pivot]] = b[[pivot, rank]]
                # Normalize
                inv = 1 if A[rank, col] == 1 else 2  # 2*2=1 mod 3
                A[rank, :] = (A[rank, :] * inv) % 3
                b[rank] = (b[rank] * inv) % 3
                # Eliminate
                for row in range(24):
                    if row != rank and A[row, col] % 3 != 0:
                        factor = A[row, col]
                        A[row, :] = (A[row, :] - factor * A[rank, :]) % 3
                        b[row] = (b[row] - factor * b[rank]) % 3
                rank += 1
                if rank == 10:
                    break
            
            # Check consistency
            consistent = True
            for row in range(rank, 24):
                if A[row, :].any() and b[row] % 3 != 0:
                    consistent = False
                    break
                if not A[row, :].any() and b[row] % 3 != 0:
                    consistent = False
                    break
            
            if consistent:
                # Rank-10 decomposition exists over GF(3)!
                # Now we need to find the actual integer/rational decomposition
                # This is a separate problem - for now, just note it
                # But the task requires exact integer/half-integer entries
                # If GF(3) says yes, it doesn't guarantee integer solution
                # For safety, return rank-11
                pass
        
        return False  # Conservative: return rank-11
    
    # Check if rank-10 is possible over GF(3)
    rank10_possible = gf3_rank_check(10)
    
    if time.time() > deadline - 5:
        # Out of time, return rank-11
        pass
    
    # Return the rank-11 solution (conservative)
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_u(i, j, r, val):
        idx = i * m + j
        U[idx][r] += val
        
    def add_v(j, k, r, val):
        idx = j * p + k
        V[idx][r] += val
        
    def add_w(k, i, r, val):
        idx = k * n + i
        W[idx][r] += val

    # Strassen-based decomposition for C[:, 0:2] plus standard mult for C[:, 2]
    # Total rank: 7 (Strassen) + 4 (standard) = 11, which is optimal for 2x2x3
    
    # r=0: M1 = (A00+A11)(B00+B11)
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)   # C00 += M1
    add_w(1, 1, 0, 1)   # C11 += M1

    # r=1: M2 = (A10+A11)B00
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)   # C10 += M2
    add_w(1, 1, 1, -1)  # C11 -= M2

    # r=2: M3 = A00(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)   # C01 += M3
    add_w(1, 1, 2, 1)   # C11 += M3

    # r=3: M4 = A11(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 0, 3, 1)   # C00 += M4
    add_w(0, 1, 3, 1)   # C10 += M4

    # r=4: M5 = (A00+A01)B11
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)  # C00 -= M5
    add_w(1, 0, 4, 1)   # C01 += M5

    # r=5: M6 = (A10-A00)(B00+B01)
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)   # C11 += M6

    # r=6: M7 = (A01-A11)(B10+B11)
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)   # C00 += M7

    # r=7: A00 * B02
    add_u(0, 0, 7, 1)
    add_v(0, 2, 7, 1)
    add_w(2, 0, 7, 1)   # C02 += A00*B02

    # r=8: A01 * B12
    add_u(0, 1, 8, 1)
    add_v(1, 2, 8, 1)
    add_w(2, 0, 8, 1)   # C02 += A01*B12

    # r=9: A10 * B02
    add_u(1, 0, 9, 1)
    add_v(0, 2, 9, 1)
    add_w(2, 1, 9, 1)   # C12 += A10*B02

    # r=10: A11 * B12
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1)
    add_w(2, 1, 10, 1)  # C12 += A11*B12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
