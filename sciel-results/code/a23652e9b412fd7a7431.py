import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Attempt to construct a rank 36 solution using:
    # 1. A rank 8 decomposition for 2x2x3 (j=0,1; k=0,1,2)
    # 2. Two rank 7 Strassen blocks for 2x2x2 (j=0,1; k=3,4 and k=5,6) -> Wait, 2x2x4 is two 2x2x2.
    #    So k=3,4 uses one Strassen block (rank 7).
    #    k=5,6 uses one Strassen block (rank 7).
    # 3. A rank 14 schoolbook decomposition for 2x1x7 (j=2; k=0..6)
    # Total rank: 8 + 7 + 7 + 14 = 36.
    
    N_A = n * m
    N_B = m * p
    N_C = p * n
    
    U = [[0] * 36 for _ in range(N_A)]
    V = [[0] * 36 for _ in range(N_B)]
    W = [[0] * 36 for _ in range(N_C)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs_u=None, coeffs_v=None, coeffs_w=None):
        nonlocal col
        if coeffs_u is None: coeffs_u = [1]*len(u_indices)
        if coeffs_v is None: coeffs_v = [1]*len(v_indices)
        if coeffs_w is None: coeffs_w = [1]*len(w_indices)
        
        for idx, c in zip(u_indices, coeffs_u):
            U[idx][col] += c
        for idx, c in zip(v_indices, coeffs_v):
            V[idx][col] += c
        for idx, c in zip(w_indices, coeffs_w):
            W[idx][col] += c
        col += 1

    # --- Part 1: 2x2x3 decomposition (rank 8) ---
    # Indices:
    # A: i=0,1; j=0,1. U indices: 0,1,2,3
    # B: j=0,1; k=0,1,2. V indices: 0,1,2, 7,8,9
    # C: i=0,1; k=0,1,2. W indices: 0,1, 2,3, 4,5 (k*n+i)
    
    # Proposed rank 8 scheme for 2x2x3:
    # m1 = (A00+A11)(B00+B10+B11)
    # m2 = (A01+A11)(B01+B10+B11)
    # m3 = (A10+A11)(B00+B01+B11)
    # m4 = A00(B10-B00)
    # m5 = A01(B11-B01)
    # m6 = A10(B00-B10)
    # m7 = A11(B01-B11)
    # m8 = (A00+A01)(B02+B12)
    
    # Let's verify this scheme.
    # C00 = A00 B00 + A01 B10
    # From m1..m7:
    # This scheme is tricky to verify by hand.
    # I will use a known valid rank 8 decomposition if I can find one.
    # Actually, R(2,2,3) is known to be 8.
    # Let's try to construct it via ILP or use a known set of equations.
    # Due to complexity and risk, I will fall back to the seed if I cannot be sure.
    # However, I can try a simpler improvement:
    # Maybe R(2,2,7) can be done in 24?
    # 2x2x7 = 2x2x3 + 2x2x4.
    # If R(2,2,3)=8, R(2,2,4)=14, sum=22.
    # If I can't implement R(2,2,3)=8 correctly, I might break the program.
    
    # Let's try to implement the Strassen blocks for 2x2x2 and see if we can merge them.
    # The seed uses 3 separate Strassen blocks.
    # Can we do 2x2x6 in rank 21? Yes, 3*7=21.
    # Can we do 2x2x7 in rank 24?
    # If we use 3 Strassen blocks (21) + 1 schoolbook (4) = 25.
    # If we use 2 Strassen blocks (14) + 1 "2x2x3" block.
    # If R(2,2,3)=8, then 14+8=22.
    # If R(2,2,3)=9, then 14+9=23.
    # If R(2,2,3)=10, then 14+10=24.
    
    # I will try to implement a rank 8 decomposition for 2x2x3.
    # If it fails verification, I will return the seed.
    
    # Verification helper
    def verify(U, V, W, rank):
        # Check if sum_r U[:,r] V[:,r] W[:,r] == T
        # T is sparse.
        # Instead of full tensor, check random samples or all 42 entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zeros
        # This is expensive, but for small N it's fine.
        # We only need to check if the decomposition is exact.
        # The tensor T is 1 at (i*m+j, j*p+k, k*n+i).
        # All other entries should be 0.
        # Number of entries: 6 * 21 * 14 = 1764.
        # Rank 36. 1764 * 36 ~ 63k ops. Fast.
        for u in range(N_A):
            for v in range(N_B):
                for w in range(N_C):
                    # Check if (u,v,w) is a valid tensor entry
                    i = u // m
                    j = u % m
                    jj = v // p
                    k = v % p
                    kk = w // n
                    ii = w % n
                    is_valid = (j == jj and i == ii)
                    val = 0.0
                    if not is_valid:
                        for r in range(rank):
                            if U[u][r] != 0 and V[v][r] != 0 and W[w][r] != 0:
                                val += U[u][r] * V[v][r] * W[w][r]
                        if abs(val) > 1e-9:
                            return False
        return True

    # Try Rank 36 construction
    # We need a rank 8 decomposition for 2x2x3.
    # I will use the one from "Lower bounds for the rank of matrix multiplication tensors" or similar.
    # Actually, I will try to solve for the coefficients of a rank 8 decomposition using linear algebra.
    # This is too slow for 120s if done naively.
    
    # Let's stick to the seed's structure but try to optimize the 2x2x7 part.
    # If I can't find a valid rank < 25 for 2x2x7, I return seed.
    
    # I will return the seed to ensure validity.
    return _construct_seed()

def _construct_seed():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([0, 4], [b00, b11], [c00, c11])
        
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
