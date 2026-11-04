"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # We will construct a decomposition of rank 46.
    # Structure:
    # Partition j (1..5) into J1={1,2}, J2={3,4}, J3={5}.
    # Term 1: A[:, J1] (2x2) * B[J1, :] (2x5). Rank 18.
    # Term 2: A[:, J2] (2x2) * B[J2, :] (2x5). Rank 18.
    # Term 3: A[:, J3] (2x1) * B[J3, :] (1x5). Rank 10.
    # Total Rank 46.
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add a decomposition block
    # local_n, local_m, local_p
    # i_offset, j_offset, k_offset
    def add_block(local_n, local_m, local_p, i_off, j_off, k_off, block_decomp):
        nonlocal rank_offset
        for r, (u, v, w) in enumerate(block_decomp):
            # u is local_n x local_m
            # v is local_m x local_p
            # w is local_p x local_n
            # Global indices:
            # U index: (i_off + i) * m + (j_off + j)
            # V index: (j_off + j) * p + (k_off + k)
            # W index: (k_off + k) * n + (i_off + i)
            
            for i in range(local_n):
                for j in range(local_m):
                    U[(i_off + i) * m + (j_off + j)][rank_offset + r] = u[i, j]
            for j in range(local_m):
                for k in range(local_p):
                    V[(j_off + j) * p + (k_off + k)][rank_offset + r] = v[j, k]
            for k in range(local_p):
                for i in range(local_n):
                    W[(k_off + k) * n + (i_off + i)][rank_offset + r] = w[k, i]
        rank_offset += len(block_decomp)

    # Decomposition of M_{2,2,2} (Strassen)
    # Returns list of (U, V, W) tuples
    def strassen_2_2_2():
        decomp = []
        # P1 = (a+d)(e+h)
        u1 = np.array([[1, 0], [1, 1]])
        v1 = np.array([[1, 1], [1, 0]])
        w1 = np.array([[1, 0], [1, 0]]) # C00 += P1, C11 += P1? No.
        # Let's use the standard formulas.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # P1 = (a+d)(e+h)
        u = np.array([[1, 0], [1, 1]])
        v = np.array([[1, 1], [1, 0]])
        # Contribution to C:
        # C00 += 1, C11 += 1
        w = np.array([[1, 0], [0, 1]])
        decomp.append((u, v, w))
        
        # P2 = (c+d)e
        u = np.array([[0, 0], [1, 1]])
        v = np.array([[1, 0], [0, 0]])
        # C10 += 1
        w = np.array([[0, 0], [1, 0]])
        decomp.append((u, v, w))
        
        # P3 = a(f-h)
        u = np.array([[1, 0], [0, 0]])
        v = np.array([[0, 1], [0, -1]])
        # C01 += 1, C11 += 1
        w = np.array([[0, 1], [0, 1]])
        decomp.append((u, v, w))
        
        # P4 = d(g-e)
        u = np.array([[0, 0], [0, 1]])
        v = np.array([[-1, 0], [1, 0]])
        # C00 += 1, C10 += 1
        w = np.array([[1, 0], [1, 0]])
        decomp.append((u, v, w))
        
        # P5 = (a+b)h
        u = np.array([[1, 1], [0, 0]])
        v = np.array([[0, 0], [0, 1]])
        # C00 -= 1, C01 += 1
        w = np.array([[-1, 1], [0, 0]])
        decomp.append((u, v, w))
        
        # P6 = (c-a)(e+f)
        u = np.array([[-1, 0], [1, 0]])
        v = np.array([[1, 1], [0, 0]])
        # C11 += 1
        w = np.array([[0, 0], [0, 1]])
        decomp.append((u, v, w))
        
        # P7 = (b-d)(g+h)
        u = np.array([[0, 1], [0, -1]])
        v = np.array([[1, 1], [0, 1]])
        # C00 += 1
        w = np.array([[1, 0], [0, 0]])
        decomp.append((u, v, w))
        
        return decomp

    # Decomposition of M_{2,2,1} (Trivial)
    def trivial_2_2_1():
        decomp = []
        # C00 = ae, C10 = ce, C01 = bg, C11 = dg
        # Wait, B is 2x1. B00=e, B10=g.
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00
        u = np.array([[1, 0], [0, 0]])
        v = np.array([[1, 0], [0, 0]])
        w = np.array([[1, 0]])
        decomp.append((u, v, w))
        
        # Term 2: A01 B10
        u = np.array([[0, 1], [0, 0]])
        v = np.array([[0, 0], [1, 0]])
        w = np.array([[1, 0]])
        decomp.append((u, v, w))
        
        # Term 3: A10 B00
        u = np.array([[0, 0], [1, 0]])
        v = np.array([[1, 0], [0, 0]])
        w = np.array([[0, 1]])
        decomp.append((u, v, w))
        
        # Term 4: A11 B10
        u = np.array([[0, 0], [0, 1]])
        v = np.array([[0, 0], [1, 0]])
        w = np.array([[0, 1]])
        decomp.append((u, v, w))
        
        return decomp

    # Decomposition of M_{2,2,5}
    # Partition k (0..4) into K1={0,1}, K2={2,3}, K3={4}
    # Term 1: M_{2,2,2} on K1
    # Term 2: M_{2,2,2} on K2
    # Term 3: M_{2,2,1} on K3
    def mul_2_2_5():
        decomp = []
        decomp.extend(strassen_2_2_2())
        decomp.extend(strassen_2_2_2())
        decomp.extend(trivial_2_2_1())
        return decomp

    # Decomposition of M_{2,1,5}
    # Trivial
    def mul_2_1_5():
        decomp = []
        # A is 2x1, B is 1x5.
        # C0k = A00 B0k, C1k = A10 B0k
        for k in range(5):
            u = np.zeros((2, 1))
            v = np.zeros((1, 5))
            w = np.zeros((5, 2))
            
            u[0, 0] = 1
            v[0, k] = 1
            w[k, 0] = 1
            decomp.append((u, v, w))
            
            u[1, 0] = 1
            v[0, k] = 1
            w[k, 1] = 1
            decomp.append((u, v, w))
        return decomp

    # Add Term 1: A[:, 0:2] * B[0:2, :]
    # local_n=2, local_m=2, local_p=5
    # i_off=0, j_off=0, k_off=0
    add_block(2, 2, 5, 0, 0, 0, mul_2_2_5())
    
    # Add Term 2: A[:, 2:4] * B[2:4, :]
    # local_n=2, local_m=2, local_p=5
    # i_off=0, j_off=2, k_off=0
    add_block(2, 2, 5, 0, 2, 0, mul_2_2_5())
    
    # Add Term 3: A[:, 4:5] * B[4:5, :]
    # local_n=2, local_m=1, local_p=5
    # i_off=0, j_off=4, k_off=0
    add_block(2, 1, 5, 0, 4, 0, mul_2_1_5())
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
