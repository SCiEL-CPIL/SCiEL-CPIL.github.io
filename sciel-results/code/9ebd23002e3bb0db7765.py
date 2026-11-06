"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40
    all_mask = (1 << n) - 1

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Vectorized check: for each b in b_mask, check if adj[b] & c_mask
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def k5_seed():
        """
        Create a K5-clique seed for densification.
        Vertices 0-4 form a K5.
        """
        edges = set()
        for i in range(5):
            for j in range(i + 1, 5):
                edges.add((i, j))
        return edges

    def dense_random_init(seed_val, prob=0.65):
        """
        Create a dense random initialization and prune C6s.
        """
        random.seed(seed_val)
        adj = [0] * n
        edges = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < prob:
                    edges.add((i, j))
        
        adj = build_adj(edges)
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for u, v in list(edges):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if would_create_c6(adj, u, v):
                    to_remove.append((u, v))
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            for u, v in to_remove:
                edges.remove((u, v))
                changed = True
        
        return edges

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 2:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search_sa(edges_set, seed_val, T_start=1.0, T_end=0.1, max_iter=300):
        """
        Simulated Annealing Local Search with more aggressive swaps.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        current_score = len(current_edges)
        
        # Precompute non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        T = T_start
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.4 and non_edges:
                # Add an edge
                idx = random.randint(0, len(non_edges) - 1)
                u, v = non_edges[idx]
                
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    non_edges.pop(idx)
                    current_score += 1
                else:
                    pass
            
            elif move_type < 0.7 and current_edges:
                # Remove an edge
                edge_list = list(current_edges)
                idx = random.randint(0, len(edge_list) - 1)
                e_rem = edge_list[idx]
                u, v = e_rem
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.remove(e_rem)
                
                if random.random() < np.exp(-1.0 / T):
                    current_score -= 1
                    non_edges.append(e_rem)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add(e_rem)
            
            else:
                # Swap: Remove 1, Add 1 (or 2)
                if current_edges and non_edges:
                    edge_list = list(current_edges)
                    idx_rem = random.randint(0, len(edge_list) - 1)
                    e_rem = edge_list[idx_rem]
                    u_rem, v_rem = e_rem
                    
                    # Try 1-for-1 swap
                    idx_add = random.randint(0, len(non_edges) - 1)
                    e_add = non_edges[idx_add]
                    u_add, v_add = e_add
                    
                    adj[u_rem] &= ~(1 << v_rem)
                    adj[v_rem] &= ~(1 << u_rem)
                    current_edges.remove(e_rem)
                    
                    if not would_create_c6(adj, u_add, v_add):
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        current_edges.add(e_add)
                        
                        non_edges.pop(idx_add)
                        non_edges.append(e_rem)
                    else:
                        # Try 2-for-1 swap: add two edges
                        if len(non_edges) >= 2:
                            idx_add2 = random.randint(0, len(non_edges) - 1)
                            e_add2 = non_edges[idx_add2]
                            u_add2, v_add2 = e_add2
                            
                            if not would_create_c6(adj, u_add2, v_add2):
                                adj[u_add2] |= (1 << v_add2)
                                adj[v_add2] |= (1 << u_add2)
                                current_edges.add(e_add2)
                                
                                # Remove e_rem from current, add to non_edges
                                non_edges.append(e_rem)
                                # Remove e_add2 from non_edges
                                non_edges.pop(idx_add2)
                            else:
                                # Revert
                                adj[u_rem] |= (1 << v_rem)
                                adj[v_rem] |= (1 << u_rem)
                                current_edges.add(e_rem)
                        else:
                            # Revert
                            adj[u_rem] |= (1 << v_rem)
                            adj[v_rem] |= (1 << u_rem)
                            current_edges.add(e_rem)
            
            T = max(T_end, T * 0.98)
            
            # Periodic greedy fill
            if it % 30 == 0:
                added_any = False
                for i in range(len(non_edges)):
                    u, v = non_edges[i]
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.add((u, v))
                        non_edges.pop(i)
                        added_any = True
                        break
                if not added_any and it > 50:
                    break
            
        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        adj = build_adj(edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        edges = local_search_sa(edges, seed_val)
        
        return edges

    # Main search loop
    
    best_edges = set()
    best_count = 0
    
    # Try K5 seed + greedy
    if time.time() < deadline - 60:
        k5 = k5_seed()
        greedy_from_k5 = run_greedy(k5, 42)
        if len(greedy_from_k5) > best_count:
            best_edges = greedy_from_k5
            best_count = len(best_edges)
    
    # Try multiple dense random initializations
    if time.time() < deadline - 50:
        for seed_val in [42, 123, 777, 999, 555, 111, 222, 333]:
            if time.time() > deadline - 45:
                break
            dense_edges = dense_random_init(seed_val)
            if len(dense_edges) > best_count:
                best_edges = dense_edges
                best_count = len(best_edges)
    
    # Initial greedy
    if time.time() < deadline - 30:
        best_edges = run_greedy(best_edges, 42)
    
    # SA Local Search
    if time.time() < deadline - 20:
        best_edges = local_search_sa(best_edges, 42)
    
    best_count = len(best_edges)
    
    # Multiple restarts with perturbation
    restart_seed = 100
    for i in range(80):
        if time.time() > deadline - 10:
            break
        
        num_remove = random.randint(2, 10)
        perturbed = perturb_and_refine(best_edges, restart_seed + i, num_remove)
        
        if len(perturbed) > best_count:
            best_edges = perturbed
            best_count = len(perturbed)
        
        restart_seed += 1
    
    # Final polish: try to add any remaining non-edges
    adj_final = build_adj(best_edges)
    final_non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_final[i] & (1 << j)):
                final_non_edges.append((i, j))
    
    random.shuffle(final_non_edges)
    for u, v in final_non_edges:
        if time.time() > deadline - 2:
            break
        if not would_create_c6(adj_final, u, v):
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
            best_edges.add((u, v))
    
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
