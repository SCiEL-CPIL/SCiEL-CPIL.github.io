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

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
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

    def build_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
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
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
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

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    def generate_all_partitions(size):
        """Generate all set partitions of {0, 1, ..., size-1}."""
        if size == 0:
            yield []
            return
        if size == 1:
            yield [[0]]
            return
        
        # Generate all partitions by considering where element (size-1) goes
        # Either it's in a new block, or it joins an existing block
        base_partitions = list(generate_all_partitions(size - 1))
        
        seen = set()
        for part in base_partitions:
            # Option 1: (size-1) starts a new block
            new_part = part + [[size - 1]]
            key = tuple(tuple(sorted(b)) for b in sorted(new_part, key=lambda x: min(x)))
            if key not in seen:
                seen.add(key)
                yield new_part
            
            # Option 2: (size-1) joins each existing block
            for idx in range(len(part)):
                new_part = [b[:] for b in part]
                new_part[idx].append(size - 1)
                # Normalize: sort within blocks and sort blocks
                new_part = [sorted(b) for b in new_part]
                new_part.sort(key=lambda x: min(x))
                key = tuple(tuple(b) for b in new_part)
                if key not in seen:
                    seen.add(key)
                    yield new_part
    
    # Generate all partitions
    i_splits = list(generate_all_partitions(n))
    j_splits = list(generate_all_partitions(m))
    k_splits = list(generate_all_partitions(p))
    
    # Try all combinations with time check
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                # Calculate rank for this grouping
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
                
                # Build the decomposition
                result = build_decomposition(i_g, j_g, k_g)
                
                # Verify
                if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Random local search: try to reduce rank by removing redundant columns
    if best_rank < n * m * p and time.time() < deadline - 10:
        best_U = [row[:] for row in best_result["U"]]
        best_V = [row[:] for row in best_result["V"]]
        best_W = [row[:] for row in best_result["W"]]
        current_rank = best_rank
        
        # Try to remove columns that are redundant
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            
            if current_rank == 0:
                break
            
            # Randomly pick a column to try removing
            col_to_remove = random.randint(0, current_rank - 1)
            
            # Build candidate without this column
            cand_U = []
            for row in best_U:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_U.append(cand_row)
            
            cand_V = []
            for row in best_V:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_V.append(cand_row)
            
            cand_W = []
            for row in best_W:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_W.append(cand_row)
            
            # Verify
            if verify_sparse(cand_U, cand_V, cand_W, current_rank - 1):
                best_U = cand_U
                best_V = cand_V
                best_W = cand_W
                current_rank -= 1
        
        if current_rank < best_rank:
            best_result = {"U": best_U, "V": best_V, "W": best_W}
            best_rank = current_rank
    
    # Random local search: try to reduce rank by removing redundant columns
    if best_rank < n * m * p and time.time() < deadline - 10:
        best_U = [row[:] for row in best_result["U"]]
        best_V = [row[:] for row in best_result["V"]]
        best_W = [row[:] for row in best_result["W"]]
        current_rank = best_rank
        
        # Try to remove columns that are redundant
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            if current_rank == 0:
                break
            
            # Randomly pick a column to try removing
            col_to_remove = random.randint(0, current_rank - 1)
            
            # Build candidate without this column
            cand_U = []
            for row in best_U:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_U.append(cand_row)
            
            cand_V = []
            for row in best_V:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_V.append(cand_row)
            
            cand_W = []
            for row in best_W:
                cand_row = []
                for c in range(current_rank):
                    if c != col_to_remove:
                        cand_row.append(row[c])
                cand_W.append(cand_row)
            
            # Verify
            if verify_sparse(cand_U, cand_V, cand_W, current_rank - 1):
                best_U = cand_U
                best_V = cand_V
                best_W = cand_W
                current_rank -= 1
        
        if current_rank < best_rank:
            best_result = {"U": best_U, "V": best_V, "W": best_W}
            best_rank = current_rank
    
    return best_result
# EVOLVE-BLOCK-END
