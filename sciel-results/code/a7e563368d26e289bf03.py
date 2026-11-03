import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 0, 1], # f
        [0, 0, 0, 1, 0, 1, 0], # g
        [1, 0, -1, -1, 1, 0, 1] # h
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [-1, 0, 1, 0, 1, 1, 1], # C01
        [1, 1, 0, 1, 0, -1, 0], # C10
        [1, -1, 0, -1, 0, 0, 0]  # C11
    ]
    
    # Convert to numpy for Kronecker product
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    # Kronecker product
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # Permute rows to match 4x4 layout
    # Index mapping: k (0-3) -> I=k//2, J=k%2
    # l (0-3) -> i=l//2, j=l%2
    # Target row: (2*I + i)*4 + (2*J + j)
    
    def permute_rows(matrix):
        permuted = [[0]*49 for _ in range(16)]
        for k in range(4):
            I = k // 2
            J = k % 2
            for l in range(4):
                i = l // 2
                j = l % 2
                src_idx = k * 4 + l
                dst_idx = (2 * I + i) * 4 + (2 * J + j)
                for c in range(49):
                    permuted[dst_idx][c] = matrix[src_idx][c]
        return permuted

    U4 = permute_rows(U4_np)
    V4 = permute_rows(V4_np)
    W4 = permute_rows(W4_np)
    
    return {"U": U4, "V": V4, "W": W4}
