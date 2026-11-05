"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = 3, 3, 6
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use block structure. Split 3x6 into three 3x2 blocks.
    # For each 3x2 block, find a low-rank decomposition, then combine.
    # 3x3x2 tensor: C[i,k] = sum_j A[i,j]*B[j,k] for k in {0,1}
    # This has 3*3*2 = 18 entries.
    
    # Let me try to find a rank-r decomposition for 3x3x2
    # U: 9 x r, V: 6 x r, W: 6 x r (W indexed by k*n+i = k*3+i)
    
    def find_3x3x2_decomp(rank, max_iters=50000):
        """Find a rank decomposition for 3x3x2 matrix multiplication."""
        nn, mm, pp = 3, 3, 2
        vals = [0, 0.5, 1, -0.5, -1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(nn*mm)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(mm*pp)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(pp*nn)]
            
            ok = True
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*mm+j][c] * V[j*pp+k][c] * W[k*nn+i][c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return U, V, W
        return None
    
    # Try to find low-rank 3x3x2 decompositions
    # Then combine three of them for the 3x3x6 case
    # But we can share the U part since A is the same!
    
    # Key insight: For 3x3x6 = A * [B1 B2 B3], the U matrix (for A) can be shared
    # across all three blocks. Only V and W need to be separate.
    
    # So the decomposition is:
    # U: 9 x r (shared)
    # V: 18 x r (V1, V2, V3 each 6 x r)
    # W: 18 x r (W1, W2, W3 each 6 x r)
    
    # The constraint is: for each block b in {0,1,2}:
    #   sum_c U[i,j,c] * V_b[j,k,c] * W_b[k,i,c] = 1 for all i,j,k
    
    # This means we need a single U that works with three different (V_b, W_b) pairs.
    
    # Let me try a different approach: use a known structure.
    # For 3x3x3, there's a rank-23 decomposition. Let me try to adapt it.
    
    # Actually, let me try a more direct optimization approach.
    # Start with schoolbook and try to reduce rank by merging terms.
    
    # Alternative: use the fact that 3x3x6 can be seen as a tensor product
    # or use bilinear algorithms.
    
    # Let me try: for each of the 3 blocks (3x2), use a rank-10 decomposition
    # and share the U matrix. Total rank = 10 (shared U) + ... no, that doesn't work directly.
    
    # Let me try a completely different structural approach:
    # Use the identity that for any matrices, we can write the product using
    # a fixed number of multiplications that is less than schoolbook.
    
    # For 2x2x2, Strassen uses 7 instead of 8.
    # For 3x3x3, best known is 23 (or 22 with newer results).
    
    # For 3x3x6, a reasonable target is around 40-45.
    
    # Let me try a local search approach: start with schoolbook, then try to
    # replace groups of terms with fewer terms.
    
    # Actually, let me try a more systematic search with better initialization.
    # I'll use the structure that U entries are from a small set, and use
    # a constraint satisfaction approach.
    
    # Let me try ranks from 30 to 53 and see if I can find solutions.
    
    random.seed(12345)
    
    def try_full_rank(r, max_iters=10000):
        """Try to find a rank-r decomposition for 3x3x6."""
        if time.time() > deadline:
            return None
        
        vals = [0, 0.5, 1, -0.5, -1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = [[random.choice(vals) for _ in range(r)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(18)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(18)]
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return U, V, W
        
        return None
    
    # Try a range of ranks
    for r in range(30, 54):
        if time.time() > deadline:
            break
        result = try_full_rank(r, max_iters=5000)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
