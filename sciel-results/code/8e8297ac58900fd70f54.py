import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = 2, 4, 7
    
    # First, build the known 50-rank construction
    U_base = [[0] * 50 for _ in range(n * m)]
    V_base = [[0] * 50 for _ in range(m * p)]
    W_base = [[0] * 50 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        U_base[i0*m + j0][col] += 1
        U_base[(i0+1)*m + (j0+1)][col] += 1
        V_base[j0*p + k0][col] += 1
        V_base[(j0+1)*p + (k0+1)][col] += 1
        W_base[k0*n + i0][col] += 1
        W_base[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U_base[(i0+1)*m + j0][col] += 1
        U_base[(i0+1)*m + (j0+1)][col] += 1
        V_base[j0*p + k0][col] += 1
        W_base[k0*n + (i0+1)][col] += 1
        W_base[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        U_base[i0*m + j0][col] += 1
        V_base[j0*p + (k0+1)][col] += 1
        V_base[(j0+1)*p + (k0+1)][col] -= 1
        W_base[(k0+1)*n + i0][col] += 1
        W_base[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U_base[(i0+1)*m + (j0+1)][col] += 1
        V_base[(j0+1)*p + k0][col] += 1
        V_base[j0*p + k0][col] -= 1
        W_base[k0*n + i0][col] += 1
        W_base[k0*n + (i0+1)][col] += 1
        col += 1
        
        U_base[i0*m + j0][col] += 1
        U_base[i0*m + (j0+1)][col] += 1
        V_base[(j0+1)*p + (k0+1)][col] += 1
        W_base[k0*n + i0][col] -= 1
        W_base[(k0+1)*n + i0][col] += 1
        col += 1
        
        U_base[(i0+1)*m + j0][col] += 1
        U_base[i0*m + j0][col] -= 1
        V_base[j0*p + k0][col] += 1
        V_base[j0*p + (k0+1)][col] += 1
        W_base[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        U_base[i0*m + (j0+1)][col] += 1
        U_base[(i0+1)*m + (j0+1)][col] -= 1
        V_base[(j0+1)*p + k0][col] += 1
        V_base[(j0+1)*p + (k0+1)][col] += 1
        W_base[k0*n + i0][col] += 1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(n):
        for j in range(m):
            U_base[i*m + j][col] = 1
            V_base[j*p + 6][col] = 1
            W_base[6*n + i][col] = 1
            col += 1
    
    assert col == 50, f"Rank mismatch: used {col}, expected 50"
    
    # Precompute index arrays for vectorized verification
    # For each (i,j,k) triple, we need U[i*m+j], V[j*p+k], W[k*n+i]
    ij_list = []
    jk_list = []
    ki_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ij_list.append(i*m + j)
                jk_list.append(j*p + k)
                ki_list.append(k*n + i)
    ij_arr = np.array(ij_list, dtype=int)
    jk_arr = np.array(jk_list, dtype=int)
    ki_arr = np.array(ki_list, dtype=int)
    
    def verify_vec(U, V, W, rank):
        """Vectorized verification: check T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Select rows: U[ij_arr, :], V[jk_arr, :], W[ki_arr, :]
        # Shape: (n*m*p, rank)
        Us = U_arr[ij_arr, :]  # (56, rank)
        Vs = V_arr[jk_arr, :]  # (56, rank)
        Ws = W_arr[ki_arr, :]  # (56, rank)
        # Element-wise product then sum over rank
        vals = np.sum(Us * Vs * Ws, axis=1)  # (56,)
        return np.allclose(vals, 1.0, atol=1e-9)
    
    if not verify_vec(U_base, V_base, W_base, 50):
        return {"U": U_base, "V": V_base, "W": W_base}
    
    best_rank = 50
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Helper to remove a set of columns
    def remove_cols(U, V, W, cols_to_remove, rank):
        keep = [r for r in range(rank) if r not in cols_to_remove]
        new_rank = len(keep)
        U_new = [[0]*new_rank for _ in range(n*m)]
        V_new = [[0]*new_rank for _ in range(m*p)]
        W_new = [[0]*new_rank for _ in range(p*n)]
        for new_idx, old_idx in enumerate(keep):
            for i in range(n*m):
                U_new[i][new_idx] = U[i][old_idx]
            for j in range(m*p):
                V_new[j][new_idx] = V[j][old_idx]
            for k in range(p*n):
                W_new[k][new_idx] = W[k][old_idx]
        return U_new, V_new, W_new, new_rank
    
    # Phase 1: Try removing single terms
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        U_test = [row[:] for row in U_base]
        V_test = [row[:] for row in V_base]
        W_test = [row[:] for row in W_base]
        
        for i in range(n*m):
            U_test[i][trial] = 0
        for j in range(m*p):
            V_test[j][trial] = 0
        for k in range(p*n):
            W_test[k][trial] = 0
        
        if verify_vec(U_test, V_test, W_test, 50):
            best_U, best_V, best_W, best_rank = remove_cols(U_base, V_base, W_base, {trial}, 50)
            break
    
    # Phase 2: Try removing pairs of terms (only if we still have time)
    if time.time() < deadline - 10 and best_rank == 50:
        # Try all pairs
        for a in range(50):
            if time.time() > deadline - 10:
                break
            for b in range(a+1, 50):
                if time.time() > deadline - 10:
                    break
                
                U_test = [row[:] for row in U_base]
                V_test = [row[:] for row in V_base]
                W_test = [row[:] for row in W_base]
                
                for i in range(n*m):
                    U_test[i][a] = 0
                    U_test[i][b] = 0
                for j in range(m*p):
                    V_test[j][a] = 0
                    V_test[j][b] = 0
                for k in range(p*n):
                    W_test[k][a] = 0
                    W_test[k][b] = 0
                
                if verify_vec(U_test, V_test, W_test, 50):
                    best_U, best_V, best_W, best_rank = remove_cols(U_base, V_base, W_base, {a, b}, 50)
                    break
            if best_rank < 50:
                break
    
    # Phase 3: If we reduced to 49, try removing another single term
    if best_rank == 49 and time.time() < deadline - 5:
        for trial in range(best_rank):
            if time.time() > deadline - 5:
                break
            
            U_test = [row[:] for row in best_U]
            V_test = [row[:] for row in best_V]
            W_test = [row[:] for row in best_W]
            
            for i in range(n*m):
                U_test[i][trial] = 0
            for j in range(m*p):
                V_test[j][trial] = 0
            for k in range(p*n):
                W_test[k][trial] = 0
            
            if verify_vec(U_test, V_test, W_test, best_rank):
                best_U, best_V, best_W, best_rank = remove_cols(best_U, best_V, best_W, {trial}, best_rank)
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
