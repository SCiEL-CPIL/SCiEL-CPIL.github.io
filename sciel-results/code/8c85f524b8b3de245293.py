"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decomposition
    # For 2x2 x 2x15, try ranks from 30 upward
    for target_rank in range(30, rank_school + 1):
        if time.time() - _start_time > 140:
            break
        # Try a structured decomposition
        # Idea: use 2 extra terms for Strassen-like savings, plus 2*p-2 for the rest
        # Actually, let's try a specific construction
        
        # Construction: For each column k, use 2 products, plus 2 shared products
        # This gives rank = 2*p + 2 = 32
        if target_rank == 2 * p + 2:
            r = target_rank
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # 2 shared products (indices 0, 1)
            # Product 0: (a00 + a10) * something
            # Product 1: (a01 + a11) * something
            
            # Let me try a different approach:
            # For each k, we need:
            # c0k = a00*b0k + a01*b1k
            # c1k = a10*b0k + a11*b1k
            #
            # Use:
            # p0k = (a00 + a10) * b0k  -> contributes a00*b0k to c0k and a10*b0k to c1k
            # p1k = (a01 + a11) * b1k  -> contributes a01*b1k to c0k and a11*b1k to c1k
            # But we also need cross terms... 
            #
            # Actually:
            # c0k = a00*b0k + a01*b1k
            # c1k = a10*b0k + a11*b1k
            #
            # If I use:
            # s0 = a00 + a10, s1 = a01 + a11
            # t0k = b0k, t1k = b1k
            # 
            # s0 * t0k = a00*b0k + a10*b0k -> gives part of c0k and part of c1k
            # s1 * t1k = a01*b1k + a11*b1k -> gives part of c0k and part of c1k
            #
            # c0k = (s0*t0k + s1*t1k - a10*b0k - a11*b1k) ... no, that's not right
            #
            # Let me try:
            # u0k = (a00 + a01) * (b0k + b1k) = a00*b0k + a00*b1k + a01*b0k + a01*b1k
            # u1k = (a10 + a11) * (b0k + b1k) = a10*b0k + a10*b1k + a11*b0k + a11*b1k
            # u2k = (a00 - a01) * (b0k - b1k) = a00*b0k - a00*b1k - a01*b0k + a01*b1k
            # u3k = (a10 - a11) * (b0k - b1k) = a10*b0k - a10*b1k - a11*b0k + a11*b1k
            #
            # c0k = a00*b0k + a01*b1k = (u0k + u2k)/2
            # c1k = a10*b0k + a11*b1k = (u1k + u3k)/2
            #
            # So 4 products per column, rank = 4*15 = 60. No savings.
            
            # Try another approach: share across columns
            # Define:
            # A0 = a00, A1 = a01, A2 = a10, A3 = a11
            # For column k: B0k = b0k, B1k = b1k
            # c0k = A0*B0k + A1*B1k
            # c1k = A2*B0k + A3*B1k
            #
            # Use products:
            # P0k = (A0 + A2) * B0k = A0*B0k + A2*B0k
            # P1k = (A1 + A3) * B1k = A1*B1k + A3*B1k
            # P2k = (A0 + A1) * B0k = A0*B0k + A1*B0k  
            # P3k = (A2 + A3) * B1k = A2*B1k + A3*B1k
            #
            # c0k = P0k + P1k - A2*B0k - A3*B1k ... still need extra terms
            
            # I think rank 2p+2 might not be achievable with simple constructions.
            # Let me just return the schoolbook for now and try to improve.
            pass
        
        # For now, just use schoolbook
        if time.time() - _start_time > 140:
            break
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
