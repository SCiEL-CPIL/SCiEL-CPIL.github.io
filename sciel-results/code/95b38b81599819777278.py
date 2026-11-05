"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def time_exceeded():
        return time.time() > deadline

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_pattern, v_pattern, w_pattern) where patterns are
        dicts mapping local index tuples to integer values.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def generate_all_set_partitions(dim):
        """Generate all set partitions of {0,...,dim-1} into blocks of size 1 or 2."""
        results = []
        def gen(remaining, current):
            if not remaining:
                results.append(current[:])
                return
            first = min(remaining)
            rem = remaining - {first}
            current.append([first])
            gen(rem, current)
            current.pop()
            for other in rem:
                new_rem = rem - {other}
                current.append([first, other])
                gen(new_rem, current)
                current.pop()
        gen(set(range(dim)), [])
        return results

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute the total rank for a given partition using Strassen for 2x2x2 blocks."""
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        return total_rank

    def build_decomposition_for_partition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = compute_rank_for_partition(i_groups, j_groups, k_groups)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                U[gi * m + gj][col] = val_u
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                V[gj * p + gk][col] = val_v
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                W[gk * n + gi][col] = val_w
                            col += 1
                    else:
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def reduce_rank_greedy(U, V, W, rank):
        """Greedy linear dependency reduction: try removing each term if tensor is preserved."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        def is_valid(U, V, W, r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(r):
                            val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        current_rank = rank
        changed = True
        while changed and current_rank > 1:
            if time_exceeded():
                break
            changed = False
            for t in range(current_rank):
                if time_exceeded():
                    break
                U_new = [row[:t] + row[t+1:] for row in U]
                V_new = [row[:t] + row[t+1:] for row in V]
                W_new = [row[:t] + row[t+1:] for row in W]
                if is_valid(U_new, V_new, W_new, current_rank - 1):
                    U = U_new
                    V = V_new
                    W = W_new
                    current_rank -= 1
                    changed = True
                    break
        
        return U, V, W, current_rank

    def random_search(rank_target, max_iter=5000):
        """
        Randomized search for a rank-rank_target decomposition.
        Start from a random initialization and use coordinate descent / local search.
        """
        np.random.seed(42)
        
        # Build the target tensor T[i,j,k] = 1 for all i,j,k
        # We need U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed over t = 1
        
        # Initialize with random values
        U = np.random.randn(n*m, rank_target) * 0.5
        V = np.random.randn(m*p, rank_target) * 0.5
        W = np.random.randn(p*n, rank_target) * 0.5
        
        # Compute current tensor values
        def compute_tensor(U, V, W):
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
            return T
        
        # Try to minimize ||T - 1||^2
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iter):
            if time_exceeded():
                break
            
            T = compute_tensor(U, V, W)
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
            
            # If we have a good solution, try to verify and return
            if loss < 1e-10:
                U_list = [[U[i*m+j, t] for t in range(rank_target)] for i in range(n) for j in range(m)]
                V_list = [[V[j*p+k, t] for t in range(rank_target)] for j in range(m) for k in range(p)]
                W_list = [[W[k*n+i, t] for t in range(rank_target)] for k in range(p) for i in range(n)]
                return {"U": U_list, "V": V_list, "W": W_list, "rank": rank_target}
            
            # Coordinate descent: update one factor at a time
            # Update U
            for t in range(rank_target):
                if time_exceeded():
                    break
                # For each (i,j), U[i*m+j, t] should satisfy:
                # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all k
                # This is a linear system in U[i*m+j, t] for fixed t
                # Actually, let's do gradient descent style updates
                
            # Simpler: use a small random perturbation (simulated annealing style)
            lr = 0.01 * (1.0 - it/max_iter) + 0.001
            noise_scale = 0.1 * (1.0 - it/max_iter) + 0.01
            
            # Perturb U
            U_pert = U + np.random.randn(n*m, rank_target) * noise_scale
            T_pert = compute_tensor(U_pert, V, W)
            loss_pert = np.sum((T_pert - 1.0)**2)
            if loss_pert < loss:
                U = U_pert
            else:
                # Accept with probability exp(-delta_loss / temperature)
                temperature = 0.1 * (1.0 - it/max_iter) + 0.01
                if np.random.random() < np.exp(-(loss_pert - loss) / temperature):
                    U = U_pert
            
            # Perturb V
            V_pert = V + np.random.randn(m*p, rank_target) * noise_scale
            T_pert = compute_tensor(U, V_pert, W)
            loss_pert = np.sum((T_pert - 1.0)**2)
            if loss_pert < loss:
                V = V_pert
            else:
                if np.random.random() < np.exp(-(loss_pert - loss) / temperature):
                    V = V_pert
            
            # Perturb W
            W_pert = W + np.random.randn(p*n, rank_target) * noise_scale
            T_pert = compute_tensor(U, V, W_pert)
            loss_pert = np.sum((T_pert - 1.0)**2)
            if loss_pert < loss:
                W = W_pert
            else:
                if np.random.random() < np.exp(-(loss_pert - loss) / temperature):
                    W = W_pert
        
        # Return best found
        if best_U is not None and best_loss < 1e-10:
            U_list = [[best_U[i*m+j, t] for t in range(rank_target)] for i in range(n) for j in range(m)]
            V_list = [[best_V[j*p+k, t] for t in range(rank_target)] for j in range(m) for k in range(p)]
            W_list = [[best_W[k*n+i, t] for t in range(rank_target)] for k in range(p) for i in range(n)]
            return {"U": U_list, "V": V_list, "W": W_list, "rank": rank_target}
        
        return None

    def strassen_block_decomposition():
        """
        Use a more sophisticated block decomposition.
        For 3x5x6, we can partition into blocks and apply Strassen where possible.
        
        Key insight: We can partition the dimensions into blocks of sizes that allow
        Strassen (2x2x2) blocks, and use schoolbook for the rest.
        
        For n=3: partitions into {1,1,1}, {1,2}, {3}
        For m=5: partitions into {1,1,1,1,1}, {1,1,1,2}, {1,2,2}, {1,1,3}, {2,3}, {5}
        For p=6: partitions into {1,1,1,1,1,1}, {1,1,1,1,2}, {1,1,1,3}, {1,1,2,2}, {1,2,3}, {1,5}, {2,2,2}, {2,4}, {3,3}, {6}
        
        The best partition will maximize the number of 2x2x2 Strassen blocks.
        """
        i_partitions = generate_all_set_partitions(n)
        j_partitions = generate_all_set_partitions(m)
        k_partitions = generate_all_set_partitions(p)
        
        min_rank = n * m * p
        best_partition = None
        
        for i_g in i_partitions:
            for j_g in j_partitions:
                for k_g in k_partitions:
                    r = compute_rank_for_partition(i_g, j_g, k_g)
                    if r < min_rank:
                        min_rank = r
                        best_partition = (i_g, j_g, k_g)
        
        if best_partition is not None:
            i_g, j_g, k_g = best_partition
            result = build_decomposition_for_partition(i_g, j_g, k_g)
            rank = result.pop("rank")
            return result, rank
        
        return schoolbook(), n * m * p

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try block Strassen decomposition
    if not time_exceeded():
        strassen_result, strassen_rank = strassen_block_decomposition()
        if strassen_rank < best_rank:
            if verify_numpy(strassen_result["U"], strassen_result["V"], strassen_result["W"], strassen_rank):
                # Apply greedy rank reduction
                U_red, V_red, W_red, rank_red = reduce_rank_greedy(
                    strassen_result["U"], strassen_result["V"], strassen_result["W"], strassen_rank
                )
                if rank_red < best_rank:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = rank_red
                else:
                    best_result = strassen_result
                    best_rank = strassen_rank

    # Try random search for lower ranks
    if not time_exceeded():
        # Try ranks from current best down
        for target_rank in range(best_rank - 1, max(best_rank - 20, 1), -1):
            if time_exceeded():
                break
            result = random_search(target_rank, max_iter=3000)
            if result is not None:
                if verify_numpy(result["U"], result["V"], result["W"], target_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = target_rank
                    break

    # Final greedy reduction if time permits
    if not time_exceeded():
        U_s, V_s, W_s = best_result["U"], best_result["V"], best_result["W"]
        U_r, V_r, W_r, rank_r = reduce_rank_greedy(U_s, V_s, W_s, best_rank)
        if rank_r < best_rank:
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = rank_r

    return best_result
# EVOLVE-BLOCK-END

