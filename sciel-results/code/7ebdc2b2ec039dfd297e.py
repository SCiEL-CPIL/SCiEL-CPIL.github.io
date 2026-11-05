import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Partitions
    # I: 3 -> 2, 1
    # J: 5 -> 2, 2, 1
    # K: 6 -> 2, 2, 2
    
    i_splits = [2, 1]
    j_splits = [2, 2, 1]
    k_splits = [2, 2, 2]
    
    # Calculate offsets
    i_offsets = [0]
    for s in i_splits[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in j_splits[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in k_splits[:-1]:
        k_offsets.append(k_offsets[-1] + s)
        
    # We will accumulate U, V, W
    # U shape: (n, rank)
    # V shape: (m, rank)
    # W shape: (p, rank)
    
    # We don't know the total rank yet, so we use lists
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(i_start, j_start, k_start, i_size, j_size, k_size, local_U, local_V, local_W):
        # local_U is (i_size, r), local_V is (j_size, r), local_W is (k_size, r)
        for r in range(local_U.shape[1]):
            u = np.zeros(n)
            v = np.zeros(m)
            w = np.zeros(p)
            
            u[i_start:i_start+i_size] = local_U[:, r]
            v[j_start:j_start+j_size] = local_V[:, r]
            w[k_start:k_start+k_size] = local_W[:, r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Returns U, V, W of shape (2, 7), (2, 7), (2, 7)
        # T_{ijk} = sum U_{ir} V_{jr} W_{kr}
        # The 7 terms:
        # 1: (A11+A22)(B11+B22) -> U: [1, 1], V: [1, 0] for A part? No.
        # The tensor is T_{i, j, k}.
        # Term r contributes U_{i, r} V_{j, r} W_{k, r} to T_{i, j, k}.
        # We need T_{i, j, k} = 1 if it's a valid multiplication index.
        
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # This contributes to C11, C12, C21, C22.
        # The tensor indices for A are (i, j), for B are (j, k).
        # Wait, the tensor is T_{i, j, k}.
        # A_{i, j} B_{j, k}.
        
        # Let's define the 7 vectors for U (i index), V (j index), W (k index).
        # i, j, k in {0, 1}
        
        U = np.zeros((2, 7))
        V = np.zeros((2, 7))
        W = np.zeros((2, 7))
        
        # Term 1: (A11+A22)(B11+B22)
        # A11 is i=0, j=0. A22 is i=1, j=1.
        # So U[:, 0] should be [1, 1]?
        # V[:, 0] should be [1, 1]? (j=0 for A11, j=1 for A22)
        # W[:, 0] should be [1, 1]? (k=0 for B11, k=1 for B22)
        # Let's check T_{0,0,0} = U00 V00 W00 = 1*1*1 = 1. Correct.
        # T_{0,0,1} = U00 V00 W10 = 1*1*1 = 1. Correct (C11 uses B12? No, C11 uses B11. C12 uses B12.
        # Wait. C_{ik} = sum_j A_{ij} B_{jk}.
        # T_{i, j, k} = 1.
        # If we use Strassen, we are replacing the sum with 7 products.
        # The decomposition is an identity for the tensor.
        
        # The standard Strassen decomposition for the tensor:
        # 1. U=[1,1], V=[1,1], W=[1,1]
        # 2. U=[0,1], V=[0,1], W=[1,0]  (A22 * B21? No)
        
        # Let's use the known coefficients for Strassen tensor decomposition.
        # The 7 rank-1 tensors are:
        # 1. (e1+e2) (e1+e2) (e1+e2)
        # 2. e2 (e1+e2) e1
        # 3. e1 (e2-e1) e2
        # 4. e2 (e1-e2) e1  -- Wait, signs matter.
        
        # Correct Strassen vectors (U, V, W):
        # 1: U=[1,1], V=[1,1], W=[1,1]
        # 2: U=[0,1], V=[0,1], W=[1,0]
        # 3: U=[1,0], V=[0,-1], W=[0,1]
        # 4: U=[0,1], V=[1,0], W=[0,-1]
        # 5: U=[1,0], V=[1,0], W=[0,1]
        # 6: U=[-1,1], V=[1,1], W=[0,1]
        # 7: U=[0,-1], V=[0,1], W=[1,0]
        
        # Let's verify one: T_{0,0,0}
        # 1: 1*1*1 = 1
        # 2: 0*0*1 = 0
        # 3: 1*0*0 = 0
        # 4: 0*1*0 = 0
        # 5: 1*1*0 = 0
        # 6: -1*1*0 = 0
        # 7: 0*0*1 = 0
        # Sum = 1. Correct.
        
        # T_{0,0,1}
        # 1: 1*1*1 = 1
        # 2: 0*0*0 = 0
        # 3: 1*0*1 = 0
        # 4: 0*1*(-1) = 0
        # 5: 1*1*1 = 1
        # 6: -1*1*1 = -1
        # 7: 0*0*0 = 0
        # Sum = 1+1-1 = 1. Correct.
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [1, 1, 0, 1, 0, 1, -1]
        ])
        V = np.array([
            [1, 0, 0, 1, 1, 1, 0],
            [1, 1, -1, 0, 0, 1, 1]
        ])
        W = np.array([
            [1, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, -1, 1, 1, 0]
        ])
        return U, V, W

    # 2x1x2 decomposition (Rank 2)
    def block_2x1x2():
        # T_{i, 0, k} = 1
        # Term 1: U=[1,1], V=[1], W=[1,1]
        # Term 2: U=[1,1], V=[0], W=[0,0] -- No, V is size 1.
        # If j_size=1, V is (1, r).
        # T_{i, 0, k} = U_{i, r} V_{0, r} W_{k, r}
        # We need sum_r ... = 1 for all i, k.
        # This is just the outer product of ones.
        # Rank 1?
        # Yes, T_{i, 0, k} = 1 * 1 * 1.
        # So rank 1.
        # Wait, earlier I said 2.
        # If m=1, C_{ik} = A_{i1} B_{1k}.
        # This is a single term.
        # So rank is 1.
        
        U = np.array([[1], [1]])
        V = np.array([[1]])
        W = np.array([[1], [1]])
        return U, V, W

    # 1x2x2 decomposition (Rank 2)
    def block_1x2x2():
        # T_{0, j, k} = 1
        # Rank 1?
        # C_{0k} = sum_j A_{0j} B_{jk}.
        # This is not a single outer product.
        # T_{0, j, k} = 1.
        # Sum_r U_{0, r} V_{j, r} W_{k, r} = 1.
        # If we use rank 1: U=[1], V=[1,1], W=[1,1].
        # Then T_{0, 0, 0} = 1*1*1 = 1.
        # T_{0, 1, 0} = 1*1*1 = 1.
        # So rank 1.
        
        U = np.array([[1]])
        V = np.array([[1], [1]])
        W = np.array([[1], [1]])
        return U, V, W

    # 1x1x2 decomposition (Rank 1)
    def block_1x1x2():
        U = np.array([[1]])
        V = np.array([[1]])
        W = np.array([[1], [1]])
        return U, V, W

    # Iterate through blocks
    for i_idx, i_size in enumerate(i_splits):
        for j_idx, j_size in enumerate(j_splits):
            for k_idx, k_size in enumerate(k_splits):
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U, V, W = strassen_2x2x2()
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    U, V, W = block_2x1x2()
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    U, V, W = block_1x2x2()
                elif i_size == 1 and j_size == 1 and k_size == 2:
                    U, V, W = block_1x1x2()
                else:
                    # Fallback to schoolbook for other combinations (should not happen with this split)
                    r = i_size * j_size * k_size
                    U = np.zeros((i_size, r))
                    V = np.zeros((j_size, r))
                    W = np.zeros((k_size, r))
                    col = 0
                    for a in range(i_size):
                        for b in range(j_size):
                            for c in range(k_size):
                                U[a, col] = 1
                                V[b, col] = 1
                                W[c, col] = 1
                                col += 1

                add_block(i_offsets[i_idx], j_offsets[j_idx], k_offsets[k_idx], 
                          i_size, j_size, k_size, U, V, W)

    U_final = np.column_stack(U_list)
    V_final = np.column_stack(V_list)
    W_final = np.column_stack(W_list)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

