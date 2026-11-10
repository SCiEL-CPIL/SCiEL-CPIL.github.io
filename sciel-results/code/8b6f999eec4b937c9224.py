"""Dense Q3-free graph on 16 vertices via diverse structured init + targeted local search."""
import time
import random

def construct():
    deadline = time.time() + 112  # Safety margin before 120s kill
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def creates_q3_fast(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def get_missing(adj):
        return [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]

    def get_existing(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append((u, v))
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def prune_to_q3free(adj):
        """Remove edges to make the graph Q3-free. Greedy: iterate edges, remove if it creates a Q3."""
        # We need to check if the current graph contains a Q3.
        # Strategy: try removing each edge and see if it helps.
        # Actually, simpler: iterate through all edges, and for each, check if removing it is necessary.
        # Better: check all 8-subsets for Q3 presence, and if found, remove an edge from it.
        # Most efficient: just iterate edges in random order, and for each edge (u,v),
        # temporarily remove it, check if graph is still Q3-free... no that's expensive.
        
        # Alternative: since we're starting from structured graphs that are close to Q3-free,
        # just do a few passes of: find a Q3, remove an edge from it.
        # To find a Q3: pick random 8-sets and check.
        
        # Actually, the simplest correct approach: 
        # For each edge in the graph, check if the graph WITHOUT that edge is Q3-free.
        # If the graph WITH the edge is not Q3-free but WITHOUT it is, keep it removed.
        # But checking Q3-freedom is expensive.
        
        # Pragmatic approach: start with the dense graph, and repeatedly:
        # 1. Try to find a Q3 by random sampling of 8-vertices
        # 2. If found, remove a random edge from that Q3
        # 3. Repeat until no Q3 found in many trials
        
        # For efficiency, let's use a different approach:
        # The structured initializations (K8,8, Turan) are known to be Q3-free or close.
        # K8,8: bipartite, Q3 is not bipartite? No, Q3 IS bipartite (it's a cube, 2-colorable).
        # So K8,8 could contain Q3. Let's just do greedy edge addition from scratch for these.
        
        # Actually, the cleanest approach: for each structured init, do a greedy ADD pass
        # starting from EMPTY, but using the structured graph as a priority order.
        # This guarantees Q3-freedom by construction.
        
        # Even simpler: just use the structured graph as a HINT for edge ordering in greedy add.
        pass
    
    def structured_greedy_init(structured_edges, rng):
        """Greedy add using structured edge order as priority, then random fill."""
        adj = [0] * n
        # First add structured edges in order
        for u, v in structured_edges:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if not creates_q3_fast(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Then add remaining edges randomly
        remaining = get_missing(adj)
        rng.shuffle(remaining)
        greedy_add(adj, remaining)
        return adj
    
    def local_search(adj, max_time):
        """Targeted swap local search with multiple greedy passes."""
        start_time = time.time()
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        # Phase 1: Multiple greedy passes with different orderings
        for pass_i in range(5):
            if time.time() > deadline - 3 or time.time() > start_time + max_time:
                break
            missing = get_missing(adj)
            if not missing:
                break
            random.shuffle(missing)
            greedy_add(adj, missing)
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]
        
        # Phase 2: Targeted swaps
        for _ in range(50):
            if time.time() > deadline - 3 or time.time() > start_time + max_time:
                break
            
            missing_now = get_missing(adj)
            if not missing_now:
                break
            
            found_swap = False
            for _attempt in range(15):
                if time.time() > deadline - 3 or time.time() > start_time + max_time:
                    break
                if not missing_now:
                    break
                u_miss, v_miss = random.choice(missing_now)
                
                # Find existing edges incident to u_miss or v_miss
                candidates = []
                for neighbor in range(n):
                    if neighbor != u_miss and neighbor != v_miss:
                        if adj[u_miss] & (1 << neighbor):
                            candidates.append((u_miss, neighbor))
                        if adj[v_miss] & (1 << neighbor):
                            candidates.append((v_miss, neighbor))
                
                if not candidates:
                    continue
                
                u_rem, v_rem = random.choice(candidates)
                
                trial = best_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                if not creates_q3_fast(u_miss, v_miss, trial):
                    trial[u_miss] |= (1 << v_miss)
                    trial[v_miss] |= (1 << u_miss)
                
                missing_trial = get_missing(trial)
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
                    found_swap = True
                    missing_now = get_missing(adj)
                    break
            
            if not found_swap:
                # Random perturbation
                existing = get_existing(best_adj)
                if not existing:
                    break
                
                u_rem, v_rem = random.choice(existing)
                trial = best_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                missing_trial = get_missing(trial)
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
        
        return best_adj, best_count
    
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Define structured initializations
    # 1. K8,8 (complete bipartite)
    k88_edges = [(u, v) for u in range(8) for v in range(8, 16)]
    
    # 2. Turan T(16,4): complete 4-partite with parts of size 4
    turan_edges = []
    parts = [[i for i in range(4 * k, 4 * k + 4)] for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    turan_edges.append((u, v))
    
    # 3. Random dense (p=0.6)
    rand_dense_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.6:
                rand_dense_edges.append((u, v))
    
    # 4. Random dense (p=0.7)
    rand_dense2_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.7:
                rand_dense2_edges.append((u, v))
    
    # 5. Pure random
    rand_edges = all_edges[:]
    random.shuffle(rand_edges)
    
    # 6. Another random
    rand_edges2 = all_edges[:]
    random.shuffle(rand_edges2)
    
    structured_inits = [
        k88_edges,
        turan_edges,
        rand_dense_edges,
        rand_dense2_edges,
        rand_edges,
        rand_edges2,
    ]
    
    # For each structured init, do greedy add with that priority, then local search
    for init_edges in structured_inits:
        if time.time() > deadline - 15:
            break
        
        adj = structured_greedy_init(init_edges, random)
        adj, cnt = local_search(adj, max_time=15)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Additional pure random restarts
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        adj, cnt = local_search(adj, max_time=10)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = get_missing(best_adj)
        if remaining:
            random.shuffle(remaining)
            greedy_add(best_adj, remaining)
        
        remaining2 = get_missing(best_adj)
        if remaining2:
            remaining2.sort(key=lambda e: (best_adj[e[0]].bit_count() + best_adj[e[1]].bit_count()))
            greedy_add(best_adj, remaining2)
        
        remaining3 = get_missing(best_adj)
        if remaining3:
            random.shuffle(remaining3)
            greedy_add(best_adj, remaining3)
        
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
