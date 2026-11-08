import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen (7x7)
    rank = 49
    
    # Strassen 2x2 matrices
    # U2[i*2+j, r]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V2[j*2+k, r]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W2[k*2+i, r]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over block indices and local indices
    # I, J are block row/col for A (0..1)
    # i, j are local row/col for A (0..1)
    # K is block col for B (0..1)
    # k is local col for B (0..1)
    
    # We need to fill U, V, W
    # U index: u = (2*I + i)*4 + (2*J + j)
    # V index: v = (2*J + j)*4 + (2*K + k)
    # W index: w = (2*K + k)*4 + (2*I + i)
    
    # For each combination of block indices I, J, K
    # and local indices i, j, k
    # and Strassen indices r, r'
    
    # It is more efficient to iterate over the final indices and compute the coefficients
    # Or iterate over all components.
    
    # Let's iterate over I, J, K (block indices)
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # Iterate over local indices i, j, k
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            u = (2 * I + i) * 4 + (2 * J + j)
                            v = (2 * J + j) * 4 + (2 * K + k)
                            w = (2 * K + k) * 4 + (2 * I + i)
                            
                            # The coefficient for U[u, col] is U2[I*2+J, r] * U2[i*2+j, r']
                            # Wait, the tensor product logic:
                            # U_total[u, (r,r')] = U2_block[I, J, r] * U2_local[i, j, r']
                            # U2_block index is I*2+J? No.
                            # In M2, U index is I*2+J.
                            # So U2[I*2+J, r].
                            
                            u2_block_idx = I * 2 + J
                            u2_local_idx = i * 2 + j
                            
                            v2_block_idx = J * 2 + K
                            v2_local_idx = j * 2 + k
                            
                            w2_block_idx = K * 2 + I
                            w2_local_idx = k * 2 + i
                            
                            for r in range(7):
                                for rp in range(7):
                                    col = r * 7 + rp
                                    
                                    u_val = U2[u2_block_idx][r] * U2[u2_local_idx][rp]
                                    v_val = V2[v2_block_idx][r] * V2[v2_local_idx][rp]
                                    w_val = W2[w2_block_idx][r] * W2[w2_local_idx][rp]
                                    
                                    U[u][col] += u_val
                                    V[v][col] += v_val
                                    W[w][col] += w_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(res['U'][0])
