"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i * m + j
                    v_r = j * p + k
                    w_r = k * n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        return terms

    def build_block_decomp(i_groups, j_groups, k_groups):
        """Build decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere."""
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(nm)]
        V = [[0] * total_rank for _ in range(mp)]
        W = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
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

    def generate_all_partitions(size):
        """Generate all set partitions of {0, 1, ..., size-1}."""
        if size == 0:
            yield []
            return
        if size == 1:
            yield [[0]]
            return
        base_partitions = list(generate_all_partitions(size - 1))
        seen = set()
        for part in base_partitions:
            new_part = part + [[size - 1]]
            key = tuple(tuple(sorted(b)) for b in sorted(new_part, key=lambda x: min(x)))
            if key not in seen:
                seen.add(key)
                yield new_part
            for idx in range(len(part)):
                new_part = [b[:] for b in part]
                new_part[idx].append(size - 1)
                new_part = [sorted(b) for b in new_part]
                new_part.sort(key=lambda x: min(x))
                key = tuple(tuple(b) for b in new_part)
                if key not in seen:
                    seen.add(key)
                    yield new_part

    def prune_columns(U, V, W, rank, max_iters=2000):
        """Iteratively remove redundant columns via random and systematic search."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            if rank == 0:
                break
            
            # Try a few random columns
            found = False
            for _ in range(min(10, rank)):
                col = random.randint(0, rank - 1)
                
                # Build candidate without this column
                cand_U = []
                for row in U:
                    cand_row = row[:col] + row[col+1:]
                    cand_U.append(cand_row)
                
                cand_V = []
                for row in V:
                    cand_row = row[:col] + row[col+1:]
                    cand_V.append(cand_row)
                
                cand_W = []
                for row in W:
                    cand_row = row[:col] + row[col+1:]
                    cand_W.append(cand_row)
                
                if verify_sparse(cand_U, cand_V, cand_W, rank - 1):
                    U, V, W = cand_U, cand_V, cand_W
                    rank -= 1
                    found = True
                    break
            
            if not found and attempt % 100 == 0:
                # Systematic check of all columns
                for col in range(rank):
                    cand_U = []
                    for row in U:
                        cand_row = row[:col] + row[col+1:]
                        cand_U.append(cand_row)
                    cand_V = []
                    for row in V:
                        cand_row = row[:col] + row[col+1:]
                        cand_V.append(cand_row)
                    cand_W = []
                    for row in W:
                        cand_row = row[:col] + row[col+1:]
                        cand_W.append(cand_row)
                    if verify_sparse(cand_U, cand_V, cand_W, rank - 1):
                        U, V, W = cand_U, cand_V, cand_W
                        rank -= 1
                        break
                    if time.time() > deadline - 5:
                        break
        
        return U, V, W, rank

    def svd_based_decomposition(target_rank):
        """
        Use SVD on tensor matricizations to find low-rank approximations,
        then refine to exact rational values.
        """
        # Build the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
        # Actually, the multiplication tensor has T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = delta_{i,k}... no, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in (AB)[i,k]
        # For matrix multiplication, (AB)[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid i,j,k
        
        # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = T[i,j,k] = 1
        
        # Strategy: Try to find rank-r decomposition by solving the system
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Use random initialization and gradient-free local search
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            np.random.seed(42 + trial)
            U_rand = np.random.randn(nm, target_rank) * 0.1
            V_rand = np.random.randn(mp, target_rank) * 0.1
            W_rand = np.random.randn(pn, target_rank) * 0.1
            
            # Simple local search: perturb one entry at a time
            current_err = float('inf')
            U_cur = U_rand.copy()
            V_cur = V_rand.copy()
            W_cur = W_rand.copy()
            
            for step in range(5000):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                            err += (val - 1.0) ** 2
                
                if err < current_err:
                    current_err = err
                    if err < 1e-12:
                        break
                    if err < best_err:
                        best_err = err
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                
                # Pick a random matrix and entry to perturb
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = U_cur[row, col]
                    U_cur[row, col] = old_val + np.random.randn() * 0.05
                elif mat_choice == 'V':
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = V_cur[row, col]
                    V_cur[row, col] = old_val + np.random.randn() * 0.05
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = W_cur[row, col]
                    W_cur[row, col] = old_val + np.random.randn() * 0.05
            
            if best_err < 1e-10:
                break
        
        if best_err > 1e-8:
            return None
        
        # Convert to lists and try to recover rationals
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        return U_list, V_list, W_list

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Phase 1: Block decomposition search
    i_splits = list(generate_all_partitions(n))
    j_splits = list(generate_all_partitions(m))
    k_splits = list(generate_all_partitions(p))
    
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            else:
                                total_rank += si * sj * sk
                
                if total_rank >= best_rank:
                    continue
                
                result = build_block_decomp(i_g, j_g, k_g)
                
                if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Phase 2: Prune columns from best block decomposition
    if best_rank < n * m * p and time.time() < deadline - 15:
        pruned_U, pruned_V, pruned_W, pruned_rank = prune_columns(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    # Phase 3: SVD-based search for lower ranks
    # Try ranks below current best
    for target_rank in range(max(1, best_rank - 5), best_rank):
        if time.time() > deadline - 20:
            break
        
        result = svd_based_decomposition(target_rank)
        if result is None:
            continue
        
        U_list, V_list, W_list = result
        
        # Verify
        if verify_sparse(U_list, V_list, W_list, target_rank):
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break

    # Phase 4: Final aggressive pruning
    if best_rank < n * m * p and time.time() < deadline - 5:
        pruned_U, pruned_V, pruned_W, pruned_rank = prune_columns(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    return best_result
# EVOLVE-BLOCK-END
