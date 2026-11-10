"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 112  # safety margin before 120s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def degree(u):
        return adj_mask[u].bit_count()
    
    def total_edges():
        return sum(adj_mask[u].bit_count() for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        
        Cycle C6: u - a - b - c - d - v - u
        Need path of length 5 from u to v in G - {u,v edge}.
        """
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        # Precompute forbidden masks
        forbidden_u = 1 << u
        forbidden_v = 1 << v
        
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            # b must be neighbor of ai, not u, v, or ai itself
            nb = adj_mask[ai] & ~forbidden_u & ~forbidden_v & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                # c must be neighbor of bi, not u, v, ai, or bi
                nc = adj_mask[bi] & ~forbidden_u & ~forbidden_v & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # d must be in N(ci) ∩ N(v), and distinct from u, ai, bi, ci, v
                    # d is in nv (neighbors of v excluding u), and in N(ci)
                    # d must not be ai, bi, ci
                    nd = adj_mask[ci] & nv & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False
    
    def get_all_edges(mask):
        """Get list of edges from adjacency mask."""
        edges = []
        for u in range(n):
            m = mask[u]
            while m:
                lb = m & (-m)
                m ^= lb
                v = lb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
        return edges
    
    def greedy_add_edges(adj_mask, all_pairs, time_limit, max_iterations=5000):
        """Try to add edges greedily, prioritizing low-degree vertices."""
        added_count = 0
        iterations = 0
        
        while time.time() < time_limit and iterations < max_iterations:
            iterations += 1
            
            # Find candidate edges, prioritizing low-degree pairs
            candidates = []
            for (u, v) in all_pairs:
                if (1 << v) & adj_mask[u]:
                    continue
                du = adj_mask[u].bit_count()
                dv = adj_mask[v].bit_count()
                candidates.append((du + dv, u, v))
            
            if not candidates:
                break
            
            # Sort by combined degree (lower first)
            candidates.sort()
            
            added_this_pass = False
            for (score, u, v) in candidates:
                if time.time() > time_limit:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                if not has_c6_fast(u, v):
                    add_edge(u, v)
                    added_this_pass = True
                    added_count += 1
            
            if not added_this_pass:
                break
        
        return added_count
    
    def local_search(initial_mask, time_limit):
        """Local search with adaptive perturbation."""
        adj_mask[:] = list(initial_mask)
        best_mask = list(adj_mask)
        best_count = total_edges()
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        stall_count = 0
        
        while time.time() < time_limit - 1:
            # Try to add edges
            added = greedy_add_edges(adj_mask, all_pairs, time.time() + min(3, time_limit - time.time() - 0.5))
            
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_mask = list(adj_mask)
                stall_count = 0
            else:
                stall_count += 1
            
            if time.time() > time_limit - 1:
                break
            
            # If no progress, perturb
            if added == 0 or current_count == best_count:
                current_edges = get_all_edges(adj_mask)
                if not current_edges:
                    break
                
                # Adaptive perturbation: larger stalls get larger kicks
                if stall_count <= 1:
                    k = 1
                elif stall_count <= 3:
                    k = 2
                elif stall_count <= 5:
                    k = 3
                else:
                    k = 4
                
                k = min(k, len(current_edges) - 1)  # keep at least 1 edge
                if k <= 0:
                    break
                
                edges_to_remove = random.sample(current_edges, k)
                for (u, v) in edges_to_remove:
                    remove_edge(u, v)
        
        return best_mask, best_count
    
    # Initial state: 8 disjoint K5s
    init_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                init_adj[5 * k + a] |= (1 << (5 * k + b))
                init_adj[5 * k + b] |= (1 << (5 * k + a))
    
    best_overall_mask = list(init_adj)
    best_overall_count = total_edges()
    
    # Phase 1: Initial greedy build
    t1_end = time.time() + 4
    adj_mask[:] = list(init_adj)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    greedy_add_edges(adj_mask, all_pairs, t1_end)
    count1 = total_edges()
    if count1 > best_overall_count:
        best_overall_count = count1
        best_overall_mask = list(adj_mask)
    
    # Phase 2: Multiple local search runs
    remaining_time = deadline - time.time() - 3
    
    # Determine number of restarts based on time
    # Each restart takes roughly 8-15 seconds
    num_restarts = max(2, int(remaining_time / 12))
    
    for r in range(num_restarts):
        if time.time() > deadline - 4:
            break
        
        # Perturb the best state for diversity
        if r == 0:
            start_mask = list(best_overall_mask)
        else:
            start_mask = list(best_overall_mask)
            current_edges = get_all_edges(start_mask)
            if current_edges:
                # Vary perturbation size
                k = min(1 + (r % 3), len(current_edges) - 1)
                if k > 0:
                    edges_to_remove = random.sample(current_edges, k)
                    for (u, v) in edges_to_remove:
                        start_mask[u] &= ~(1 << v)
                        start_mask[v] &= ~(1 << u)
        
        # Time allocation: give more time to later restarts if they're improving
        t_end = time.time() + min(12, remaining_time - (r * 10))
        if t_end < time.time() + 2:
            break
        
        mask, count = local_search(start_mask, t_end)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask
    
    # Final polish: one more greedy pass on the best state
    if time.time() < deadline - 2:
        adj_mask[:] = list(best_overall_mask)
        greedy_add_edges(adj_mask, all_pairs, deadline - 1)
        final_count = total_edges()
        if final_count > best_overall_count:
            best_overall_count = final_count
            best_overall_mask = list(adj_mask)
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_overall_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
