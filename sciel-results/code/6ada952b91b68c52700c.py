import time
import random
import numpy as np
import scipy.optimize as so

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses ILP (scipy.optimize.milp) to find the maximum packing of disjoint 2x2x2 Strassen blocks.
    Falls back to heuristic local search if ILP fails or times out.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_params_list = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            block_params_list.append((i1, i2, j1, j2, k1, k2))
                            all_blocks.append(indices)
    
    num_blocks = len(all_blocks)
    
    best_num_blocks = 0
    best_packing = []

    # Try ILP first
    ilp_success = False
    try:
        if time.time() < deadline - 30.0:
            # Formulate ILP
            # x_b is 1 if block b is selected
            # Constraints: for each (i,j,k), sum of x_b for b covering (i,j,k) <= 1
            # Objective: max sum x_b
            
            # Build constraint matrix A
            # Rows: 105 tensor indices
            # Cols: num_blocks
            A = np.zeros((105, num_blocks), dtype=int)
            
            # Map (i,j,k) to row index
            def idx(i, j, k):
                return i * m * p + j * p + k
            
            for b in range(num_blocks):
                for tri in all_blocks[b]:
                    i, j, k = tri
                    A[idx(i, j, k), b] = 1
            
            # Objective: minimize -sum(x)
            c = -np.ones(num_blocks)
            
            # Bounds: 0 <= x <= 1
            bounds = [(0, 1)] * num_blocks
            
            # Integrality: 1 for all
            integrality = np.ones(num_blocks, dtype=int)
            
            res = so.milp(c, constraints=so.LinearConstraint(A, ub=1), bounds=bounds, integrality=integrality)
            
            if res.success:
                selected = res.x > 0.5
                num_selected = int(np.sum(selected))
                if num_selected > best_num_blocks:
                    best_num_blocks = num_selected
                    best_packing = [block_params_list[i] for i in range(num_blocks) if selected[i]]
                    ilp_success = True
                    
    except Exception as e:
        pass

    # If ILP failed or didn't find a good solution, use heuristic search
    # We can also use the ILP result to seed the heuristic or just run heuristic if ILP found 0
    if not ilp_success or best_num_blocks < 5:
        # Heuristic: Greedy with Local Search
        # Precompute block scores (rarity)
        triple_count = {}
        for bidx in range(num_blocks):
            for tri in all_blocks[bidx]:
                triple_count[tri] = triple_count.get(tri, 0) + 1
        
        block_scores = []
        for bidx in range(num_blocks):
            score = sum(1.0 / triple_count[tri] for tri in all_blocks[bidx])
            block_scores.append(score)
        
        sorted_by_score = sorted(range(num_blocks), key=lambda x: -block_scores[x])

        def greedy_pack(current_packing_set, used_indices, max_add=None):
            added = 0
            new_packing_list = list(current_packing_set)
            for bidx in sorted_by_score:
                if max_add is not None and added >= max_add:
                    break
                if bidx in current_packing_set:
                    continue
                idx_set = all_blocks[bidx]
                if not (used_indices & idx_set):
                    new_packing_list.append(bidx)
                    current_packing_set.add(bidx)
                    used_indices = used_indices | idx_set
                    added += 1
            return new_packing_list, used_indices, added

        num_trials = 200
        for t in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            
            random.seed(t + 54321)
            avail = list(range(num_blocks))
            random.shuffle(avail)
            
            current_packing = []
            current_packing_set = set()
            used_indices = set()
            
            for bidx in avail:
                if time.time() > deadline - 10.0:
                    break
                idx_set = all_blocks[bidx]
                if not (used_indices & idx_set):
                    current_packing.append(bidx)
                    current_packing_set.add(bidx)
                    used_indices |= idx_set
            
            current_size = len(current_packing)
            
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for k_remove in [1, 2, 3, 4]:
                    if time.time() > deadline - 10.0:
                        break
                    for _ in range(10):
                        if time.time() > deadline - 10.0:
                            break
                        if len(current_packing) < k_remove:
                            continue
                        remove_set = random.sample(current_packing, k_remove)
                        remove_set_set = set(remove_set)
                        
                        new_used = set()
                        new_packing_set = set()
                        for bp in current_packing:
                            if bp not in remove_set_set:
                                new_used |= all_blocks[bp]
                                new_packing_set.add(bp)
                                
                        new_packing_list = [bp for bp in current_packing if bp not in remove_set_set]
                        
                        # Re-greedy fill
                        new_packing_list, new_used, added = greedy_pack(new_packing_set, new_used, max_add=k_remove + 10)
                        
                        if len(new_packing_list) > current_size:
                            current_packing = new_packing_list
                            current_packing_set = set(current_packing)
                            used_indices = new_used
                            current_size = len(current_packing)
                            improved = True
                            break
                        if improved:
                            break

            if current_size > best_num_blocks:
                best_num_blocks = current_size
                best_packing = [block_params_list[i] for i in current_packing]

    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add blocks
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook for uncovered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    # Verification
    if rank > 0:
        try:
            U_np = np.array(U, dtype=np.int64)
            V_np = np.array(V, dtype=np.int64)
            W_np = np.array(W, dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        prod = U_np[u_idx] * V_np[v_idx]
                        val = np.dot(prod, W_np[w_idx])
                        
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if not valid:
                # Fallback to schoolbook
                U_cols = []
                V_cols = []
                W_cols = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
                            
                rank = len(U_cols)
                U = [[0] * rank for _ in range(U_rows)]
                V = [[0] * rank for _ in range(V_rows)]
                W = [[0] * rank for _ in range(W_rows)]
                for c in range(rank):
                    for r in range(U_rows):
                        U[r][c] = U_cols[c][r]
                    for r in range(V_rows):
                        V[r][c] = V_cols[c][r]
                    for r in range(W_rows):
                        W[r][c] = W_cols[c][r]
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
