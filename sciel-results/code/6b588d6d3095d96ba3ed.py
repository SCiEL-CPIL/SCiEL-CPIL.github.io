import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = 120

    # ---------- Precomputation (once) ----------
    # Global edge index
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1

    # All 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)

    # Local K8 edge indices (0..27)
    local_edge_idx = [[-1] * 8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1

    # Generate Q3 masks for K8 as 28-bit integers
    q3_masks_for_k8 = []
    seen_masks = set()
    for A in itertools.combinations(range(8), 4):
        B = tuple(v for v in range(8) if v not in A)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                m = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != m:
                        mask |= (1 << local_edge_idx[a][b])
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_k8.append(mask)
    q3_masks_for_k8 = tuple(q3_masks_for_k8)

    # Map global edges -> (subset, local_pos)
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1] * total_edges for _ in range(num_subsets)]
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i + 1, 8):
                g_idx = edge_index[s[i]][s[j]]
                l_pos = local_edge_idx[i][j]
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Reverse local_pos -> global edge, per subset (for swap candidate removal)
    local_to_global = [[-1] * 28 for _ in range(num_subsets)]
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i + 1, 8):
                g_idx = edge_index[s[i]][s[j]]
                local_to_global[s_idx][local_edge_idx[i][j]] = g_idx

    # ---------- Fast Q3 membership ----------
    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            new_mask = subset_masks[s_idx] | (1 << local_idx[s_idx][e_idx])
            if contains_q3(new_mask):
                return False
        return True

    # ---------- Tabu Search with Randomized Greedy Initialization ----------
    def run_tabu_search(seed, initial_G=None, initial_subset_masks=None, initial_missing=None):
        rng = random.Random(seed)
        
        if initial_G is not None:
            G = initial_G
            subset_masks = initial_subset_masks[:]
            missing_edges = initial_missing.copy()
        else:
            G = 0
            subset_masks = [0] * num_subsets
            missing_edges = set(range(total_edges))
        
        def add(e_idx):
            nonlocal G
            G |= (1 << e_idx)
            missing_edges.discard(e_idx)
            for s_idx in edge_to_subsets[e_idx]:
                subset_masks[s_idx] |= (1 << local_idx[s_idx][e_idx])
        
        def remove(e_idx):
            nonlocal G
            G &= ~(1 << e_idx)
            missing_edges.add(e_idx)
            for s_idx in edge_to_subsets[e_idx]:
                subset_masks[s_idx] &= ~(1 << local_idx[s_idx][e_idx])
        
        if initial_G is None:
            # Randomized greedy initialization - try multiple orderings
            best_init_G = 0
            best_init_count = 0
            best_init_masks = None
            best_init_missing = None
            
            for attempt in range(3):
                G_try = 0
                masks_try = [0] * num_subsets
                missing_try = set(range(total_edges))
                
                edges = list(range(total_edges))
                rng.shuffle(edges)
                for e_idx in edges:
                    if time.time() > deadline:
                        break
                    if can_add_edge(G_try, e_idx, masks_try):
                        G_try |= (1 << e_idx)
                        missing_try.discard(e_idx)
                        for s_idx in edge_to_subsets[e_idx]:
                            masks_try[s_idx] |= (1 << local_idx[s_idx][e_idx])
                
                cnt_try = G_try.bit_count()
                if cnt_try > best_init_count:
                    best_init_count = cnt_try
                    best_init_G = G_try
                    best_init_masks = masks_try
                    best_init_missing = missing_try
            
            G = best_init_G
            subset_masks = best_init_masks[:]
            missing_edges = best_init_missing.copy()
        
        # Tabu search
        current_count = G.bit_count()
        best_G = G
        best_count = current_count
        best_subset_masks = subset_masks[:]
        
        tabu_list = {}  # edge_idx -> iteration when it expires
        tabu_tenure = 15
        max_iterations = 1500
        no_improve = 0
        max_no_improve = 400
        aspiration_threshold = best_count
        
        iteration = 0
        while iteration < max_iterations and no_improve < max_no_improve and time.time() < deadline - 5.0:
            best_neighbor_count = current_count - 1
            best_neighbor_move = None
            best_neighbor_e_idx = None
            best_neighbor_f_idx = None
            
            present_edges = [i for i in range(total_edges) if (G >> i) & 1]
            
            # Try adding edges
            for e_idx in list(missing_edges):
                if can_add_edge(G, e_idx, subset_masks):
                    if e_idx in tabu_list and tabu_list[e_idx] > iteration:
                        if current_count + 1 >= aspiration_threshold:
                            pass
                        else:
                            continue
                    
                    add(e_idx)
                    new_count = G.bit_count()
                    if new_count > best_neighbor_count:
                        best_neighbor_count = new_count
                        best_neighbor_move = 'add'
                        best_neighbor_e_idx = e_idx
                        best_neighbor_f_idx = -1
                    remove(e_idx)
            
            # Try removing edges
            for f_idx in present_edges:
                if f_idx in tabu_list and tabu_list[f_idx] > iteration:
                    if current_count - 1 >= aspiration_threshold:
                        pass
                    else:
                        continue
                
                remove(f_idx)
                new_count = G.bit_count()
                if new_count > best_neighbor_count:
                    best_neighbor_count = new_count
                    best_neighbor_move = 'remove'
                    best_neighbor_e_idx = -1
                    best_neighbor_f_idx = f_idx
                add(f_idx)
            
            # Try swap moves (remove one, add one) for better exploration
            if best_neighbor_count <= current_count and missing_edges and present_edges:
                for f_idx in present_edges[:10]:  # Limit to first 10 for speed
                    if f_idx in tabu_list and tabu_list[f_idx] > iteration:
                        continue
                    remove(f_idx)
                    for e_idx in list(missing_edges)[:10]:
                        if e_idx in tabu_list and tabu_list[e_idx] > iteration:
                            continue
                        if can_add_edge(G, e_idx, subset_masks):
                            add(e_idx)
                            new_count = G.bit_count()
                            if new_count > best_neighbor_count:
                                best_neighbor_count = new_count
                                best_neighbor_move = 'swap'
                                best_neighbor_e_idx = e_idx
                                best_neighbor_f_idx = f_idx
                            remove(e_idx)
                    add(f_idx)
            
            # Apply best move or random escape
            if best_neighbor_move is not None:
                if best_neighbor_move == 'add':
                    add(best_neighbor_e_idx)
                    tabu_list[best_neighbor_e_idx] = iteration + tabu_tenure
                elif best_neighbor_move == 'remove':
                    remove(best_neighbor_f_idx)
                    tabu_list[best_neighbor_f_idx] = iteration + tabu_tenure
                elif best_neighbor_move == 'swap':
                    remove(best_neighbor_f_idx)
                    add(best_neighbor_e_idx)
                    tabu_list[best_neighbor_f_idx] = iteration + tabu_tenure
                    tabu_list[best_neighbor_e_idx] = iteration + tabu_tenure
                
                current_count = G.bit_count()
            else:
                # Random escape move
                if missing_edges and present_edges:
                    e_idx = rng.choice(list(missing_edges))
                    f_idx = rng.choice(present_edges)
                    remove(f_idx)
                    if can_add_edge(G, e_idx, subset_masks):
                        add(e_idx)
                        current_count = G.bit_count()
                        tabu_list[f_idx] = iteration + tabu_tenure
                        tabu_list[e_idx] = iteration + tabu_tenure
                    else:
                        add(f_idx)
                elif present_edges:
                    f_idx = rng.choice(present_edges)
                    remove(f_idx)
                    current_count = G.bit_count()
                    tabu_list[f_idx] = iteration + tabu_tenure
                elif missing_edges:
                    e_idx = rng.choice(list(missing_edges))
                    if can_add_edge(G, e_idx, subset_masks):
                        add(e_idx)
                        current_count = G.bit_count()
                        tabu_list[e_idx] = iteration + tabu_tenure
                else:
                    break
            
            # Update best
            if current_count > best_count:
                best_count = current_count
                best_G = G
                best_subset_masks = subset_masks[:]
                aspiration_threshold = best_count
                no_improve = 0
            else:
                no_improve += 1
            
            # Clean up expired tabu entries
            if iteration % 10 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
            
            iteration += 1
        
        return best_G, best_count

    # ---------- Multi-restart with tabu search ----------
    best_G = 0
    best_count = 0
    seed = 98765
    while time.time() < deadline - 3.0:
        G, cnt = run_tabu_search(seed)
        seed += 1
        if cnt > best_count:
            best_count = cnt
            best_G = G

    # ---------- Guaranteed-valid fallback (in case nothing found) ----------
    if best_count == 0:
        A = [0, 1, 2]
        B = [3, 4, 5]
        C = [6, 7, 8]
        D = [9, 10, 11, 12, 13, 14, 15]
        fb = []
        for a in A:
            for b in B + C + D:
                fb.append([a, b])
        for b in B:
            for c in C:
                fb.append([b, c])
        for b in B:
            for d in D:
                fb.append([b, d])
        for c in C:
            for d in D:
                fb.append([c, d])
        return fb

    return [[u, v] for (u, v) in edge_list if (best_G >> edge_index[u][v]) & 1]
