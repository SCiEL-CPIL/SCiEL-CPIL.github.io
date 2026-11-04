import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][r] = 1
                V_school[v_idx][r] = 1
                W_school[w_idx][r] = 1
                r += 1
    
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True
    
    def try_strassen_rectangular():
        """
        Decompose 4x5x5 by splitting the 5 dimension:
        A (4x5) = [A_left (4x2) | A_right (4x3)]
        B (5x5) = [[B11 (2x2), B12 (2x3)],
                   [B21 (3x2), B22 (3x3)]]
        
        C = A_left @ B11 + A_left @ B12 + A_right @ B21 + A_right @ B22
        
        Use Strassen on 2x2x2 (rank 7), schoolbook for small blocks,
        and a Strassen-like split for 4x3x3.
        """
        # Strassen 2x2x2 terms
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        def tensor_product_vec(v1, v2):
            """Tensor product of two 2-element vectors -> 4-element vector"""
            result = [0] * 4
            for a in range(2):
                for b in range(2):
                    result[a*2+b] = v1[a] * v2[b]
            return result
        
        # Build 4x2x2 using Strassen: rank 7
        # A_left is 4x2, B11 is 2x2, result is 4x2
        # Split A_left into two 2x2 blocks: A_top (2x2), A_bot (2x2)
        # Split B11 as is (2x2)
        # C_top = A_top @ B11 (2x2), C_bot = A_bot @ B11 (2x2)
        # Each 2x2x2 uses Strassen rank 7
        # Total for A_left@B11: 14 terms
        
        # But we can share: actually A_left@B11 where A_left is 4x2, B11 is 2x2
        # This is (2+2)x2x2 = 2*(2x2x2) = 2*7 = 14 with Strassen
        # Or schoolbook: 4*2*2 = 16. Strassen saves 2.
        
        # Let me use schoolbook for simplicity on small blocks and focus on the big one
        
        # Actually, let me try a different split that's more balanced:
        # Split 5 as 2+3
        # A (4x5) -> A1 (4x2), A2 (4x3)
        # B (5x5) -> B1 (2x5), B2 (3x5)
        # C = A1@B1 + A2@B2
        # A1@B1: 4x2x5, rank schoolbook 40, can we do better?
        # A2@B2: 4x3x5, rank schoolbook 60
        
        # This doesn't help much. Let me try the original block approach but optimized.
        
        # Better approach: Use the known result that M(4,5,5) can be computed
        # by embedding into a larger Strassen structure.
        
        # Let's try: rank = 4*5*5 - savings
        # Known: M(2,2,2) = 7 (Strassen)
        # M(4,4,4) = 49 (Strassen^2)
        # For 4x5x5: embed in 4x4x4 + boundary terms
        
        # The current program's block decomposition gives 85. Let me try to improve.
        
        # Alternative: use the fact that 5 = 4+1
        # A (4x5) = [A4 (4x4) | A1 (4x1)]
        # B (5x5) = [[B44 (4x4), B41 (4x1)],
        #            [B14 (1x4), B11 (1x1)]]
        # C = A4@B44 + A4@B41 + A1@B14 + A1@B11
        # A4@B44: 4x4x4, Strassen rank 49
        # A4@B41: 4x4x1, schoolbook 16
        # A1@B14: 4x1x4, schoolbook 16
        # A1@B11: 4x1x1, schoolbook 4
        # Total: 49+16+16+4 = 85 (same as current)
        
        # Can we do better on A4@B44? No, 49 is optimal for 4x4x4.
        # Can we reduce the boundary terms?
        
        # Try: 5 = 3+2
        # A (4x5) = [A3 (4x3) | A2 (4x2)]
        # B (5x5) = [[B33 (3x3), B32 (3x2)],
        #            [B23 (2x3), B22 (2x2)]]
        # C = A3@B33 + A3@B32 + A2@B23 + A2@B22
        # A3@B33: 4x3x3, schoolbook 36, or Strassen-like?
        #   Split 3=2+1: (2+1)x(2+1)x(2+1)
        #   = 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1 + 1x2x2 + 1x2x1 + 1x1x2 + 1x1x1
        #   = 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26 (using Strassen on 2x2x2)
        #   Or schoolbook: 9, with 4 rows: 36
        #   Actually for 4x3x3: use Strassen on the 3x3 part per row? No.
        #   4x3x3: think of it as 4 copies of 1x3x3, each rank 9 schoolbook = 36
        #   Or: split A3 into two 2x3 blocks, each 2x3x3
        #   2x3x3: split 3=2+1: 2x2x2(7) + 2x2x1(4) + 2x1x2(4) + 2x1x1(2) = 17
        #   Two such: 34. Worse than 36.
        #   So schoolbook 36 is good for 4x3x3.
        
        # A3@B32: 4x3x2, schoolbook 24
        # A2@B23: 4x2x3, schoolbook 24
        # A2@B22: 4x2x2, Strassen: 2*(2x2x2) = 14, or schoolbook 16
        # Total: 36+24+24+14 = 98. Worse than 85.
        
        # So 4+1 split is better. Let's stick with 85 and try to verify it works.
        # The current program already does this. Let me just make sure it's correct.
        
        # Actually, let me try a completely different approach: 
        # Use the tensor rank bound M(4,5,5) <= M(4,4,4) + 4*M(1,5,5) or similar.
        # M(1,5,5) = 25 (schoolbook, since one dimension is 1, it's just outer products)
        # That gives 49 + 100 = 149, worse.
        
        # Let me just return the block decomposition from the current program,
        # but make sure it's verified and correct.
        
        strassen_terms = []
        for ta in strassen_2x2x2:
            for tb in strassen_2x2x2:
                u = tensor_product_vec(ta[0], tb[0])
                v = tensor_product_vec(ta[1], tb[1])
                w = tensor_product_vec(ta[2], tb[2])
                strassen_terms.append((u, v, w))
        
        rank = 49 + 16 + 16 + 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r_idx = 0
        # Strassen 4x4x4 for the 4x4x4 block
        for (u, v, w) in strassen_terms:
            for i in range(4):
                for j in range(4):
                    U[i*5+j][r_idx] = u[i*4+j]
            for j in range(4):
                for k in range(4):
                    V[j*5+k][r_idx] = v[j*4+k]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r_idx] = w[k*4+i]
            r_idx += 1
        
        # 4x1x4 terms
        for i in range(4):
            for k in range(4):
                U[i*5+4][r_idx] = 1
                V[4*5+k][r_idx] = 1
                W[k*4+i][r_idx] = 1
                r_idx += 1
        
        # 4x4x1 terms
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_idx] = 1
                V[j*5+4][r_idx] = 1
                W[4*4+i][r_idx] = 1
                r_idx += 1
        
        # 4x1x1 terms
        for i in range(4):
            U[i*5+4][r_idx] = 1
            V[4*5+4][r_idx] = 1
            W[4*4+i][r_idx] = 1
            r_idx += 1
        
        if verify(U, V, W):
            return {'U': U, 'V': V, 'W': W}
        
        return None
    
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    return {'U': U_school, 'V': V_school, 'W': W_school}
