import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_np(U_arr, V_arr, W_arr):
        """Fast vectorized verification using numpy."""
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_rows = U_arr[i_idx*m + j_idx, :]
        v_rows = V_arr[j_idx*p + k_idx, :]
        w_rows = W_arr[k_idx*n + i_idx, :]
        
        sums = np.einsum('ij,ij->i', u_rows * v_rows, w_rows)
        return np.all(np.abs(sums - 1.0) < 1e-9)

    def verify_partial(U_arr, V_arr, W_arr, changed_terms):
        """Verify only terms affected by changes. Returns True if all constraints still met."""
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_rows = U_arr[i_idx*m + j_idx, :]
        v_rows = V_arr[j_idx*p + k_idx, :]
        w_rows = W_arr[k_idx*n + i_idx, :]
        
        # Compute full sums
        sums = np.einsum('ij,ij->i', u_rows * v_rows, w_rows)
        return np.all(np.abs(sums - 1.0) < 1e-9)

    def try_eliminate_terms(U_arr, V_arr, W_arr, rank):
        """Try to eliminate redundant terms using numpy for speed."""
        U_c = U_arr.copy()
        V_c = V_arr.copy()
        W_c = W_arr.copy()
        cur_rank = rank
        
        for _ in range(50):
            if time_left() < 5:
                break
            progress = False
            
            # Compute sparsity scores
            sparsity = np.sum(U_c[:, :cur_rank] == 0, axis=0) + \
                       np.sum(V_c[:, :cur_rank] == 0, axis=0) + \
                       np.sum(W_c[:, :cur_rank] == 0, axis=0)
            order = np.argsort(-sparsity)
            
            for term_idx in order:
                if time_left() < 5:
                    break
                if term_idx >= cur_rank:
                    continue
                    
                # Quick check: if term is all zeros, skip
                if np.all(U_c[:, term_idx] == 0) and np.all(V_c[:, term_idx] == 0) and np.all(W_c[:, term_idx] == 0):
                    mask = np.ones(cur_rank, dtype=bool)
                    mask[term_idx] = False
                    U_c = U_c[:, mask]
                    V_c = V_c[:, mask]
                    W_c = W_c[:, mask]
                    cur_rank -= 1
                    progress = True
                    order = order[order != term_idx]
                    order = np.where(order > term_idx, order - 1, order)
                    continue
                    
                # Save and zero out
                save_U = U_c[:, term_idx].copy()
                save_V = V_c[:, term_idx].copy()
                save_W = W_c[:, term_idx].copy()
                U_c[:, term_idx] = 0
                V_c[:, term_idx] = 0
                W_c[:, term_idx] = 0
                
                if verify_np(U_c, V_c, W_c):
                    # Remove the column
                    mask = np.ones(cur_rank, dtype=bool)
                    mask[term_idx] = False
                    U_c = U_c[:, mask]
                    V_c = V_c[:, mask]
                    W_c = W_c[:, mask]
                    cur_rank -= 1
                    progress = True
                    order = order[order != term_idx]
                    order = np.where(order > term_idx, order - 1, order)
                else:
                    # Restore
                    U_c[:, term_idx] = save_U
                    V_c[:, term_idx] = save_V
                    W_c[:, term_idx] = save_W
            
            if not progress:
                break

        return U_c.tolist(), V_c.tolist(), W_c.tolist(), cur_rank

    def try_pairwise_elimination(U_arr, V_arr, W_arr, rank):
        """Try to eliminate two terms at once using numpy."""
        if time_left() < 10:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank
        
        U_c = U_arr.copy()
        V_c = V_arr.copy()
        W_c = W_arr.copy()
        cur_rank = rank
        max_pairs = min(5000, cur_rank * (cur_rank - 1) // 2)
        pairs_checked = 0
        
        for _ in range(8):
            if time_left() < 8 or pairs_checked >= max_pairs:
                break
            progress = False
            for idx1 in range(cur_rank):
                if time_left() < 8 or pairs_checked >= max_pairs:
                    break
                for idx2 in range(idx1 + 1, cur_rank):
                    if time_left() < 8 or pairs_checked >= max_pairs:
                        break
                    pairs_checked += 1
                    
                    # Quick sparsity check
                    combined_sparsity = np.sum((U_c[:, idx1] == 0) | (U_c[:, idx2] == 0), axis=0) + \
                                       np.sum((V_c[:, idx1] == 0) | (V_c[:, idx2] == 0), axis=0) + \
                                       np.sum((W_c[:, idx1] == 0) | (W_c[:, idx2] == 0), axis=0)
                    if combined_sparsity < 2 * (n*m + m*p + p*n) * 0.7:
                        continue  # Not sparse enough, skip
                    
                    # Save and zero out both
                    save_U1 = U_c[:, idx1].copy()
                    save_U2 = U_c[:, idx2].copy()
                    save_V1 = V_c[:, idx1].copy()
                    save_V2 = V_c[:, idx2].copy()
                    save_W1 = W_c[:, idx1].copy()
                    save_W2 = W_c[:, idx2].copy()
                    U_c[:, idx1] = 0
                    U_c[:, idx2] = 0
                    V_c[:, idx1] = 0
                    V_c[:, idx2] = 0
                    W_c[:, idx1] = 0
                    W_c[:, idx2] = 0
                    
                    if verify_np(U_c, V_c, W_c):
                        # Remove both columns
                        mask = np.ones(cur_rank, dtype=bool)
                        mask[idx1] = False
                        mask[idx2] = False
                        U_c = U_c[:, mask]
                        V_c = V_c[:, mask]
                        W_c = W_c[:, mask]
                        cur_rank -= 2
                        progress = True
                        break
                    else:
                        # Restore
                        U_c[:, idx1] = save_U1
                        U_c[:, idx2] = save_U2
                        V_c[:, idx1] = save_V1
                        V_c[:, idx2] = save_V2
                        W_c[:, idx1] = save_W1
                        W_c[:, idx2] = save_W2
                if progress:
                    break
            
            if not progress:
                break

        return U_c.tolist(), V_c.tolist(), W_c.tolist(), cur_rank

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [(0,2),(1,3),(4,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
        ([(0,1),(2,3)], [(1,2),(0,4),(3,5)]),
        ([(0,1),(2,3)], [(2,3),(0,1),(4,5)]),
        ([(0,1),(2,3)], [(3,4),(0,1),(2,5)]),
        ([(0,1),(2,3)], [(4,5),(0,1),(2,3)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_np(U, V, W):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(
                np.array(U_list, dtype=float), np.array(V_list, dtype=float), 
                np.array(W_list, dtype=float), new_rank)
            
        # Try more elimination passes after pairwise
        if time_left() > 12 and new_rank < best_rank:
            U_list, V_list, W_list, new_rank2 = try_eliminate_terms(
                np.array(U_list, dtype=float), np.array(V_list, dtype=float), 
                np.array(W_list, dtype=float), new_rank)
            if new_rank2 < new_rank:
                new_rank = new_rank2
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
