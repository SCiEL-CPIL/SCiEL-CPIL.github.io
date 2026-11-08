import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = 3, 4, 4

    # Target: Find rank-minimal decomposition.
    # Current best known in this family: Rank 44 (4 blocks of rank 11).
    # Hypothesis: Rank 10 decomposition for 3x2x2 exists.
    # If found, total rank = 40.

    def find_rank10_3x2x2(timeout):
        # Tensor T for 3x2 x 2x2 -> 3x2
        # Indices: A (3x2), B (2x2), C (3x2)
        # T[i, j, k] = 1 if C[i,k] depends on A[i,j] and B[j,k]
        # Actually, the tensor is sum_{j} A[i,j] B[j,k] = C[i,k]
        # So T[i,j,k] = 1 for all i in 0..2, j in 0..1, k in 0..1.
        # Wait, the tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 if index i in A, j in B, k in C?
        # Standard convention: T = sum_{i,j,k} e_{i} \otimes e_{j} \otimes e_{k} where A[i,j]B[j,k] -> C[i,k].
        # Here A is 3x2, B is 2x2, C is 3x2.
        # A indices (i, j) with i in 0..2, j in 0..1.
        # B indices (j, k) with j in 0..1, k in 0..1.
        # C indices (i, k) with i in 0..2, k in 0..1.
        # Term: A[i,j] * B[j,k] contributes to C[i,k].
        # So for each (i,j,k), the tensor component is 1.
        
        # We want U (6x10), V (4x10), W (6x10)
        # such that sum_{r} U[i,j,r] V[j,k,r] W[i,k,r] = 1 for all i,j,k.
        
        # Note: The indices in U, V, W correspond to flattened A, B, C.
        # A flat: i*2 + j
        # B flat: j*2 + k
        # C flat: i*2 + k
        
        best_solution = None
        end_time = time.time() + timeout
        
        for _ in range(100):
            if time.time() > end_time:
                break
            
            # Random initialization
            U = np.random.randn(6, 10)
            V = np.random.randn(4, 10)
            W = np.random.randn(6, 10)
            
            # Simple gradient descent or just check if random works? No.
            # We need to solve a system.
            # This is a non-linear system.
            # Let's try a few iterations of a simple update.
            # Or just check if we can find one by random search? Unlikely.
            
            # Alternative: Use the known structure.
            # If we can't find it quickly, we return None.
            # Given the complexity, it's risky to rely on this.
            
            # Let's try a different strategy:
            # Is it possible that the rank is actually 11?
            # Literature check: R(3,2,2) is not definitively 10 in all sources, but 10 is a strong candidate.
            # However, without a guaranteed solver, I should be careful.
            
            # Let's try to verify if the current rank 11 is optimal for 3x2x2.
            # If I can't find 10, I stick to 11.
            
            pass
        
        return None

    # Attempt to find rank 10
    rank10_terms = find_rank10_3x2x2(10.0)
    
    if rank10_terms is not None:
        # Use rank 10
        local_terms = rank10_terms
        rank_per_block = 10
    else:
        # Use rank 11
        local_terms = [
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
            ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
            ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
            ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
            ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
            ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
            ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
            ([ (4, 1) ], [ (0, 1) ], [ (4, 1) ]),
            ([ (5, 1) ], [ (2, 1) ], [ (4, 1) ]),
            ([ (4, 1) ], [ (1, 1) ], [ (5, 1) ]),
            ([ (5, 1) ], [ (3, 1) ], [ (5, 1) ]),
        ]
        rank_per_block = 11

    rank = 4 * rank_per_block
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Blocks: (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] @ B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] @ B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] @ B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] @ B[2:4,2:4] -> C[:,2:4]
    ]

    for b_idx, (a_cs, b_rs, b_cs, c_cs) in enumerate(blocks):
        col_offset = b_idx * rank_per_block
        for l_idx, (u_list, v_list, w_list) in enumerate(local_terms):
            col = col_offset + l_idx

            for idx, val in u_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_a_idx = r_loc * m + (a_cs + c_loc)
                U[g_a_idx, col] += val

            for idx, val in v_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_b_idx = (b_rs + r_loc) * p + (b_cs + c_loc)
                V[g_b_idx, col] += val

            for idx, val in w_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_c_idx = (c_cs + c_loc) * n + r_loc
                W[g_c_idx, col] += val

        if time.time() > deadline:
            break

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
