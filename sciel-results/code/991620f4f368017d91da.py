import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 8 blocks * 7 Strassen terms = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 vectors
    # U_local[i*2+j, r], V_local[j*2+k, r], W_local[k*2+i, r]
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # r=0..6
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ]
    W_loc = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ]
    
    # Map local indices to global indices
    # Block indices: b (A col), c (B row), d (B col)
    # b in {0,1}, c in {0,1}, d in {0,1,2,3}
    
    col_offset = 0
    for b in range(2):
        for c in range(2):
            for d in range(4):
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            # U global index
                            u_idx = i * m + (b * 2 + j)
                            val = U_loc[i * 2 + j][r]
                            if val != 0:
                                U[u_idx][col_offset] += val
                            
                            # V global index
                            v_idx = (c * 2 + j) * p + (d * 2 + 0) # k=0? No, k varies
                            # Wait, V_loc index is j*2+k. We need to iterate k.
                            # My loops above only did i,j. I need to handle k for V and W.
                            # Let's restructure the loop to iterate i, j, k for the local tensor.
                            # But U only depends on i,j. V on j,k. W on k,i.
                            # So for a fixed r, we sum over i,j,k where U[i,j] V[j,k] W[k,i] != 0?
                            # No, the rank-1 term is a tensor product u_r \otimes v_r \otimes w_r.
                            # So U[i,j,r] = u_{r,i,j}, V[j,k,r] = v_{r,j,k}, W[k,i,r] = w_{r,k,i}.
                            # We just copy the values.
                            
                            # Let's do it cleanly.
                            # For each r in 0..6:
                            #   For i in 0..1, j in 0..1:
                            #     U[u_idx][col] = U_loc[i*2+j][r]
                            #   For j in 0..1, k in 0..1:
                            #     V[v_idx][col] = V_loc[j*2+k][r]
                            #   For k in 0..1, i in 0..1:
                            #     W[w_idx][col] = W_loc[k*2+i][r]
                            
                            pass # Placeholder for logic in next block
                    
                    # Correct Logic
                    for i in range(2):
                        for j in range(2):
                            u_idx = i * m + (b * 2 + j)
                            U[u_idx][col_offset] = U_loc[i * 2 + j][r]
                            
                    for j in range(2):
                        for k in range(2):
                            v_idx = (c * 2 + j) * p + (d * 2 + k)
                            V[v_idx][col_offset] = V_loc[j * 2 + k][r]
                            
                    for k in range(2):
                        for i in range(2):
                            w_idx = (d * 2 + k) * n + i
                            W[w_idx][col_offset] = W_loc[k * 2 + i][r]
                            
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
