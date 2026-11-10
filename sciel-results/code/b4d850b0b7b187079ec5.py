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

    def get_q3_count(G, subset_masks):
        """Count total Q3 subgraphs in the graph."""
        count = 0
        for s_idx in range(num_subsets):
            mask = subset_masks[s_idx]
            if mask.bit_count() < 12:
                continue
            for q in q3_masks_for_k8:
                if (mask & q) == q:
                    count += 1
        return count

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            new_mask = subset_masks[s_idx] | (1 << local_idx[s_idx][e_idx])
            if contains_q3(new_mask):
                return False
        return True

    def run_simulated_annealing(seed):
        rng = random.Random(seed)
        
        # Initialize with a greedy solution
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
        
        # Greedy initialization
        edges = list(range(total_edges))
        rng.shuffle(edges)
        for e_idx in edges:
            if time.time() > deadline:
                break
            if can_add_edge(G, e_idx, subset_masks):
                add(e_idx)
        
        # Simulated annealing
        initial_temp = 1.0
        final_temp = 0.01
        cooling_rate = 0.999
        steps_per_temp = 50
        
        current_count = G.bit_count()
        best_G = G
        best_count = current_count
        best_subset_masks = subset_masks[:]
        
        temp = initial_temp
        while temp > final_temp and time.time() < deadline - 5.0:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5.0:
                    break
                    
                # Choose a move: add, remove, or swap
                move_type = rng.random()
                
                if move_type < 0.3 and missing_edges:
                    # Add an edge
                    e_idx = rng.choice(list(missing_edges))
                    if can_add_edge(G, e_idx, subset_masks):
                        add(e_idx)
                        new_count = G.bit_count()
                        delta = new_count - current_count
                        if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                            current_count = new_count
                            if new_count > best_count:
                                best_count = new_count
                                best_G = G
                                best_subset_masks = subset_masks[:]
                        else:
                            remove(e_idx)
                    # If can't add, try swapping
                    else:
                        # Find a blocking subset
                        s_idx_blocker = None
                        for s_idx in edge_to_subsets[e_idx]:
                            new_mask = subset_masks[s_idx] | (1 << local_idx[s_idx][e_idx])
                            if contains_q3(new_mask):
                                s_idx_blocker = s_idx
                                break
                        
                        if s_idx_blocker is not None:
                            l_to_g = local_to_global[s_idx_blocker]
                            candidates = [l_to_g[l] for l in range(28) if ((G >> l_to_g[l]) & 1)]
                            if candidates:
                                f_idx = rng.choice(candidates)
                                # Remove f_idx
                                remove(f_idx)
                                if can_add_edge(G, e_idx, subset_masks):
                                    add(e_idx)
                                    new_count = G.bit_count()
                                    delta = new_count - current_count
                                    if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                                        current_count = new_count
                                        if new_count > best_count:
                                            best_count = new_count
                                            best_G = G
                                            best_subset_masks = subset_masks[:]
                                    else:
                                        add(f_idx)
                                else:
                                    add(f_idx)
                
                elif move_type < 0.5 and G.bit_count() > 0:
                    # Remove an edge
                    present_edges = [i for i in range(total_edges) if (G >> i) & 1]
                    if present_edges:
                        e_idx = rng.choice(present_edges)
                        remove(e_idx)
                        new_count = G.bit_count()
                        delta = new_count - current_count
                        if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                            current_count = new_count
                            if new_count > best_count:
                                best_count = new_count
                                best_G = G
                                best_subset_masks = subset_masks[:]
                        else:
                            add(e_idx)
                
                else:
                    # Swap: remove one present edge, add one missing edge
                    if missing_edges and G.bit_count() > 0:
                        present_edges = [i for i in range(total_edges) if (G >> i) & 1]
                        if present_edges:
                            f_idx = rng.choice(present_edges)
                            e_idx = rng.choice(list(missing_edges))
                            
                            # Remove f_idx
                            remove(f_idx)
                            if can_add_edge(G, e_idx, subset_masks):
                                add(e_idx)
                                new_count = G.bit_count()
                                delta = new_count - current_count
                                if delta >= 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                                    current_count = new_count
                                    if new_count > best_count:
                                        best_count = new_count
                                        best_G = G
                                        best_subset_masks = subset_masks[:]
                                else:
                                    add(f_idx)
                            else:
                                add(f_idx)
            
            temp *= cooling_rate
        
        return best_G, best_count

    # ---------- Multi-restart with simulated annealing ----------
    best_G = 0
    best_count = 0
    seed = 12345
    while time.time() < deadline - 3.0:
        G, cnt = run_simulated_annealing(seed)
        seed += 1
        if cnt > best_count:
            best_count = cnt
            best_G = G

    # ---------- Guaranteed-valid fallback (in case nothing found) ----------
    if best_count == 0:
        # 3x3 complete bipartite K3,3,3,3 is Q3-free
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
