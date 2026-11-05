import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    bs = 3 # Block size

    # Strassen steps for 2x2 block decomposition
    # (a_terms, b_terms, c_terms)
    # Indices are (block_row, block_col, coeff)
    strassen_steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Laderman steps for 3x3 matrix multiplication
    # (u_terms, v_terms, w_terms)
    # Indices are (row, col, coeff)
    # Note: This is a standard Laderman 19-term algorithm.
    laderman_steps = [
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        ([(1,0,1)], [(0,1,1)], [(1,0,1)]),
        ([(0,2,1)], [(2,0,1)], [(0,2,1)]),
        ([(2,0,1)], [(0,2,1)], [(2,0,1)]),
        ([(1,2,1)], [(2,1,1)], [(1,2,1)]),
        ([(2,1,1)], [(1,2,1)], [(2,1,1)]),
        ([(0,0,1), (0,1,1)], [(0,0,1), (1,0,1)], [(0,1,1)]), # This specific combination needs to be verified or replaced with correct Laderman
        # ...
    ]
    
    # I will use a simpler, verified Laderman implementation from a library if possible, 
    # but since I must output code, I will use a known correct set.
    
    # For the purpose of this response, I will use the 27-rank schoolbook for the inner part 
    # if I cannot guarantee Laderman, but I will try Laderman.
    
    # Actually, I will use the 27-rank schoolbook to ensure correctness, 
    # as Laderman is prone to error if not perfectly recalled.
    # But 189 is the current score.
    
    # I will use Laderman. Here is the correct Laderman 19 steps:
    
    lademan = [
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        ([(1,0,1)], [(0,1,1)], [(1,0,1)]),
        ([(0,2,1)], [(2,0,1)], [(0,2,1)]),
        ([(2,0,1)], [(0,2,1)], [(2,0,1)]),
        ([(1,2,1)], [(2,1,1)], [(1,2,1)]),
        ([(2,1,1)], [(1,2,1)], [(2,1,1)]),
        ([(0,0,1), (0,1,1)], [(0,0,1), (1,0,1)], [(0,1,1)]),
        ([(0,1,1), (1,1,1)], [(1,0,1), (1,1,1)], [(1,0,1)]),
        ([(0,0,1), (1,0,1)], [(0,1,1), (1,1,1)], [(0,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)], [(1,0,1)]),
        ([(0,0,1), (0,2,1)], [(0,0,1), (2,0,1)], [(0,2,1)]),
        ([(0,2,1), (2,2,1)], [(2,0,1), (2,2,1)], [(2,0,1)]),
        ([(0,0,1), (2,0,1)], [(0,2,1), (2,2,1)], [(0,2,1)]),
        ([(2,0,1), (2,2,1)], [(0,0,1), (0,2,1)], [(2,0,1)]),
        ([(1,1,1), (1,2,1)], [(1,1,1), (2,1,1)], [(1,2,1)]),
        ([(1,2,1), (2,2,1)], [(2,1,1), (2,2,1)], [(2,1,1)]),
    ]
    
    # This list is likely incorrect in its specifics.
    # I will use the 27-rank schoolbook for the inner 3x3 to be safe.
    # Rank 189.
    
    # Wait, the prompt says "rank-minimal".
    # I will use the 27-rank schoolbook.
    
    inner_steps = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                inner_steps.append(([(i,j,1)], [(j,k,1)], [(i,k,1)]))
                
    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    for s_idx, (a_terms, b_terms, c_terms) in enumerate(strassen_steps):
        for i_idx, (u_terms, v_terms, w_terms) in enumerate(inner_steps):
            col = s_idx * 27 + i_idx
            
            for I, J, ca in a_terms:
                for ui, uj, cu in u_terms:
                    u_idx = (I * bs + ui) * dim + (J * bs + uj)
                    U[u_idx, col] += ca * cu
                    
            for J, K, cb in b_terms:
                for vj, vk, cv in v_terms:
                    v_idx = (J * bs + vj) * dim + (K * bs + vk)
                    V[v_idx, col] += cb * cv
                    
            for I, K, cc in c_terms:
                for wi, wk, cw in w_terms:
                    w_idx = (K * bs + wk) * dim + (I * bs + wi)
                    W[w_idx, col] += cc * cw

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
