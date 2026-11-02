import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
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

    # Block 1: j=0 (schoolbook)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2 with Strassen
    add_strassen(0, 1, 0)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Check for linear dependencies to reduce rank
    if time.time() < deadline - 10.0:
        terms = []
        for r in range(rank):
            t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
            terms.append(t.flatten())
            
        M = np.array(terms).T  # shape (n*m*p, rank)
        
        # Use SVD to find rank and null vectors
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Find the numerical rank
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are null vectors
            # For each null vector, try to eliminate one term
            best_reduction = None
            best_result = None
            
            for null_idx in range(num_rank, rank):
                c = Vt[null_idx, :]  # null vector
                k_idx = np.argmax(np.abs(c))
                
                if np.abs(c[k_idx]) < 1e-8:
                    continue
                
                # Eliminate term k_idx
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(U_mat[:, r] * coeff)
                    new_V.append(V_mat[:, r] * coeff)
                    new_W.append(W_mat[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                # Verify the reduced decomposition
                if time.time() < deadline - 5.0:
                    terms_new = []
                    for r in range(len(new_U.T)):
                        t = np.einsum('i,j,k->ijk', new_U[:, r], new_V[:, r], new_W[:, r])
                        terms_new.append(t.flatten())
                    M_new = np.array(terms_new).T
                    
                    sum_new = np.sum(M_new, axis=1)
                    sum_old = np.sum(M, axis=1)
                    
                    if np.allclose(sum_new, sum_old, atol=1e-8):
                        reduction = rank - len(new_U.T)
                        if best_reduction is None or reduction > best_reduction:
                            best_reduction = reduction
                            best_result = (new_U, new_V, new_W)
            
            if best_result is not None:
                U_r, V_r, W_r = best_result
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
