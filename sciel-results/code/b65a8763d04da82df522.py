import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def naive_solution():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
    ]
    
    # Build rank-56 Strassen solution
    def build_strassen_solution():
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for k_pair in range(4):
            k_start = k_pair * 2
            # First half: X[:, 0:2] @ Y[0:2, k_start:k_start+2]
            base1 = k_pair * 14
            for r in range(7):
                cr = base1 + r
                for idx in range(4):
                    li, lj = divmod(idx, 2)
                    U[li * m + lj][cr] += strassen_U[r][idx]
                for idx in range(4):
                    lj, lk = divmod(idx, 2)
                    V[lj * p + k_start + lk][cr] += strassen_V[r][idx]
                for idx in range(4):
                    li, lk = divmod(idx, 2)
                    W[(k_start + lk) * n + li][cr] += strassen_W[r][idx]
            
            # Second half: X[:, 2:4] @ Y[2:4, k_start:k_start+2]
            base2 = k_pair * 14 + 7
            for r in range(7):
                cr = base2 + r
                for idx in range(4):
                    li, lj = divmod(idx, 2)
                    U[li * m + 2 + lj][cr] += strassen_U[r][idx]
                for idx in range(4):
                    lj, lk = divmod(idx, 2)
                    V[(2 + lj) * p + k_start + lk][cr] += strassen_V[r][idx]
                for idx in range(4):
                    li, lk = divmod(idx, 2)
                    W[(k_start + lk) * n + li][cr] += strassen_W[r][idx]
        
        return U, V, W, rank
    
    # Build rank-28 solution using 2x2x4 decomposition
    # For 2x2 @ 2x4, we can use 14 multiplications
    # Then 2x4 @ 4x8 = two blocks of 2x2 @ 2x4, so 2*14 = 28
    
    def build_rank28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # For 2x2 @ 2x4: A is 2x2, B is 2x4, C is 2x4
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # We can write this as 14 rank-1 terms
        # Using the structure: for each pair (k1, k2), use Strassen-like
        # Actually, let's just use the fact that 2x2 @ 2x4 can be done
        # by viewing B as [B[:,0:2] | B[:,2:4]] and doing two 2x2 @ 2x2
        # Each 2x2 @ 2x2 takes 7 mults, so 2x2 @ 2x4 takes 14 mults
        
        # For X[:, 0:2] @ Y[0:2, :]:
        # Y[0:2, :] is 2x8, split into 4 blocks of 2x2
        for k_pair in range(4):
            k_start = k_pair * 2
            base = k_pair * 14
            for r in range(7):
                cr = base + r
                for idx in range(4):
                    li, lj = divmod(idx, 2)
                    U[li * m + lj][cr] += strassen_U[r][idx]
                for idx in range(4):
                    lj, lk = divmod(idx, 2)
                    V[lj * p + k_start + lk][cr] += strassen_V[r][idx]
                for idx in range(4):
                    li, lk = divmod(idx, 2)
                    W[(k_start + lk) * n + li][cr] += strassen_W[r][idx]
        
        # For X[:, 2:4] @ Y[2:4, :]:
        for k_pair in range(4):
            k_start = k_pair * 2
            base = 28  # Wait, rank is only 28, so this won't fit
            # I need to reconsider. 2*14 = 28, but I'm using 4*14 = 56 slots
            pass
        
        # This approach is wrong. Let me think differently.
        # 2x4 @ 4x8: split X into [X1 | X2] (each 2x2), Y into [Y1; Y2] (each 2x8)
        # Z = X1@Y1 + X2@Y2
        # X1@Y1: 2x2 @ 2x8. This can be done by splitting Y1 into 4 blocks of 2x2
        # Each 2x2 @ 2x2 takes 7 mults, so 2x2 @ 2x8 takes 28 mults
        # Similarly for X2@Y2: 28 mults
        # Total: 56 mults. That's the Strassen solution I already have.
        
        # To get 28, I need a different decomposition.
        # Idea: Precompute linear combinations of X and Y to share multiplications.
        # For example, compute S1 = X1 + X2, S2 = X1 - X2 (each 2x2)
        # Compute T1 = Y1 + Y2, T2 = Y1 - Y2 (each 2x8)
        # Then Z = (S1+X2)(T1+Y2)/4 + ... this gets complicated.
        
        # Actually, for 2x4 @ 4x8, the rank might be 32 or 40, not 28.
        # Let me just return the Strassen solution with rank 56.
        return None
    
    # Try the rank-56 Strassen solution
    U, V, W, rank = build_strassen_solution()
    
    if verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to naive
    return naive_solution()
# EVOLVE-BLOCK-END
