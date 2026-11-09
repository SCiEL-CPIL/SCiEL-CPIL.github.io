import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int) # 8 x 42
    V = np.zeros((m * p, rank), dtype=int) # 24 x 42
    W = np.zeros((p * n, rank), dtype=int) # 12 x 42
    
    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * (group - 1)
        global_row = local_i
        return global_row * m + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * (group - 1)
        global_col = local_j + 2 * block_col
        return global_row * p + global_col

    def get_c_idx(local_i, local_j, block_col):
        global_row = local_i
        global_col = local_j + 2 * block_col
        return global_col * n + global_row

    col = 0
    for group in [1, 2]:
        for block_col in [0, 1, 2]:
            # Term 1: P1
            U[get_a_idx(group, 0, 0), col] += 1
            U[get_a_idx(group, 1, 1), col] += 1
            V[get_b_idx(group, 0, 0, block_col), col] += 1
            V[get_b_idx(group, 1, 1, block_col), col] += 1
            W[get_c_idx(0, 0, block_col), col] += 1
            W[get_c_idx(1, 1, block_col), col] += 1
            col += 1
            
            # Term 2: P2
            U[get_a_idx(group, 1, 0), col] += 1
            U[get_a_idx(group, 1, 1), col] += 1
            V[get_b_idx(group, 0, 0, block_col), col] += 1
            W[get_c_idx(1, 0, block_col), col] += 1
            W[get_c_idx(1, 1, block_col), col] -= 1
            col += 1
            
            # Term 3: P3
            U[get_a_idx(group, 0, 0), col] += 1
            V[get_b_idx(group, 0, 1, block_col), col] += 1
            V[get_b_idx(group, 1, 1, block_col), col] -= 1
            W[get_c_idx(0, 1, block_col), col] += 1
            W[get_c_idx(1, 1, block_col), col] += 1
            col += 1
            
            # Term 4: P4
            U[get_a_idx(group, 1, 1), col] += 1
            V[get_b_idx(group, 1, 0, block_col), col] += 1
            V[get_b_idx(group, 0, 0, block_col), col] -= 1
            W[get_c_idx(0, 0, block_col), col] += 1
            W[get_c_idx(1, 0, block_col), col] += 1
            col += 1
            
            # Term 5: P5
            U[get_a_idx(group, 0, 0), col] += 1
            U[get_a_idx(group, 0, 1), col] += 1
            V[get_b_idx(group, 1, 1, block_col), col] += 1
            W[get_c_idx(0, 0, block_col), col] -= 1
            W[get_c_idx(0, 1, block_col), col] += 1
            col += 1
            
            # Term 6: P6
            U[get_a_idx(group, 1, 0), col] += 1
            U[get_a_idx(group, 0, 0), col] -= 1
            V[get_b_idx(group, 0, 0, block_col), col] += 1
            V[get_b_idx(group, 0, 1, block_col), col] += 1
            W[get_c_idx(1, 1, block_col), col] += 1
            col += 1
            
            # Term 7: P7
            U[get_a_idx(group, 0, 1), col] += 1
            U[get_a_idx(group, 1, 1), col] -= 1
            V[get_b_idx(group, 1, 0, block_col), col] += 1
            V[get_b_idx(group, 1, 1, block_col), col] += 1
            W[get_c_idx(0, 0, block_col), col] += 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
