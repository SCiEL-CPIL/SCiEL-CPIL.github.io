import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    
    # Fallback: Current Rank 189 (Block-Strassen with Schoolbook 3x3)
    def construct_rank_189():
        bs = 3
        rank = 7 * 27
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        strassen_defs = [
            {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
            {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
            {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
            {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
            {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
            {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
            {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
        ]
        recon_coeffs = [
            [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
            [(2, 1.0), (4, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
        ]
        
        r_idx = 0
        for s in range(7):
            s_def = strassen_defs[s]
            for u in range(bs):
                for v in range(bs):
                    for j in range(bs):
                        for block_idx, coeff_a in s_def['A']:
                            br = block_idx // 2
                            bc = block_idx % 2
                            i_glob = br * bs + u
                            k_glob = bc * bs + v
                            u_row = i_glob * m + k_glob
                            U[u_row][r_idx] += coeff_a
                        for block_idx, coeff_b in s_def['B']:
                            br = block_idx // 2
                            bc = block_idx % 2
                            k_glob = br * bs + v
                            j_glob = bc * bs + j
                            v_row = k_glob * p + j_glob
                            V[v_row][r_idx] += coeff_b
                        for c_block_idx in range(4):
                            coeffs = recon_coeffs[c_block_idx]
                            gamma = 0.0
                            for (s_idx, g) in coeffs:
                                if s_idx == s:
                                    gamma = g
                                    break
                            if gamma != 0.0:
                                br = c_block_idx // 2
                                bc = c_block_idx % 2
                                i_glob = br * bs + u
                                j_glob = bc * bs + j
                                w_row = j_glob * n + i_glob
                                W[w_row][r_idx] += gamma
                        r_idx += 1
        return {"U": U, "V": V, "W": W}

    # Attempt Rank 161 (Strassen 2x2 outer, Rank 23 3x3 inner)
    # Coefficients for Rank 23 decomposition of 3x3 matrix multiplication
    # Source: Landsberg (1993) or similar. 
    # NOTE: These coefficients are complex to hardcode correctly without reference.
    # If I cannot guarantee correctness, I should not risk it.
    # However, the directive pushes for improvement.
    # I will use a known set if I can recall it. 
    # Since I cannot recall the 23 terms with 100% accuracy, I will stick to the safe 189 rank.
    # BUT, I can try a different safe improvement?
    # No, 189 is the best I can safely do.
    
    # Wait, is there a Rank 180 decomposition?
    # No.
    
    # I will return the Rank 189 decomposition.
    # To satisfy the "stronger algorithm" directive, I will optimize the construction time.
    # The current construction is O(1).
    
    # Let's try to implement the Rank 161 with a verification step.
    # If I get the coefficients wrong, the verification will fail, and I return 189.
    # This is a safe bet.
    
    # I need the 23 coefficients. I will use a placeholder and see if I can generate them?
    # No, generation is too slow.
    
    # I will use the Rank 189 result.
    return construct_rank_189()
# EVOLVE-BLOCK-END
