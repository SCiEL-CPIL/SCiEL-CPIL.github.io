import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Base rank 28 decomposition
    # This is the one from the seed, which is known to be valid.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2
    add_strassen(0, 1, 0)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Function to check if we can reduce rank by 1
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=10):
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        current_rank = current_U.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Flatten terms
            terms = []
            for r in range(current_rank):
                t = np.einsum('i,j,k->ijk', current_U[:, r], current_V[:, r], current_W[:, r])
                terms.append(t.flatten())
            M = np.array(terms).T # shape (dim, rank)
            
            # Check rank
            rank_M = np.linalg.matrix_rank(M, tol=1e-9)
            
            if rank_M < current_rank:
                # Find null space
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                c = Vt[-1, :] # Last row of Vt is in null space of M if rank_M < current_rank
                
                # Find pivot
                k_idx = np.argmax(np.abs(c))
                if np.abs(c[k_idx]) < 1e-9:
                    break
                
                # Eliminate
                new_U = []
                new_V = []
                new_W = []
                for r in range(current_rank):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(current_U[:, r] * coeff)
                    new_V.append(current_V[:, r] * coeff)
                    new_W.append(current_W[:, r] * coeff)
                
                current_U = np.array(new_U).T
                current_V = np.array(new_V).T
                current_W = np.array(new_W).T
                current_rank -= 1
            else:
                # No direct reduction. Try random perturbation?
                # If we are at rank 28 and rank_M == 28, we are stuck.
                # We could try to find a different basis.
                # But for now, return what we have.
                break
                
        return current_U, current_V, current_W, current_rank

    U_new, V_new, W_new, final_rank = try_reduce_rank(U_mat, V_mat, W_mat)
    
    if final_rank < rank:
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
