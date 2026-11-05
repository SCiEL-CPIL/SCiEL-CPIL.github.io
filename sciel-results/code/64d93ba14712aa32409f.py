import random
import time
import sys

def construct():
    n = 20
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s limit (evaluator kills at 150s)

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Helper to check if adding (u,v) is valid given current adj_mask and common
    # adj_mask: list of ints, bit i set if i is neighbor
    # common: 2D list of ints, common[u][v] = number of common neighbors
    def is_valid_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        # If u and v already have >= 3 common neighbors, adding edge doesn't change that, 
        # but it's not the constraint. The constraint is that NO pair has >= 3 common neighbors.
        # Adding (u,v) increases common[u][w] for w in N(v) and common[v][w] for w in N(u).
        
        # Check neighbors of v
        m = adj_mask[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                if common[u][w] >= 2:
                    return False
            m -= lsb
            
        # Check neighbors of u
        m = adj_mask[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                if common[v][w] >= 2:
                    return False
            m -= lsb
            
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        m = adj_mask[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
            m -= lsb
            
        m = adj_mask[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        # Remove from edges list
        # Linear search is O(E), E~100, acceptable for n=20
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break
                
        # Update common neighbors
        # Removing (u,v) decreases common[u][w] for w in N(v) (current, which excludes u)
        # And common[v][w] for w in N(u) (current, which excludes v)
        # Note: N(v) currently does not contain u.
        # But before removal, u was in N(v).
        # The common neighbors of u and w are vertices x such that x~u and x~w.
        # v was a common neighbor of u and w if v~u and v~w.
        # v~u was true. v~w is true if w in N(v).
        # So for all w in N(v) (current), v was a common neighbor of u and w.
        # So common[u][w] decreases by 1.
        
        m = adj_mask[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        m = adj_mask[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def fill_edges(adj_mask, common, edges):
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if not ((adj_mask[u] >> v) & 1):
                    if is_valid_add(u, v, adj_mask, common):
                        add_edge(u, v, adj_mask, common, edges)
                        changed = True
        return edges

    def run_local_search(initial_edges, seed):
        random.seed(seed)
        
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []

        for u, v in initial_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])

        for u in range(n):
            for v in range(u + 1, n):
                mask = adj_mask[u] & adj_mask[v]
                cnt = bin(mask).count('1')
                common[u][v] = cnt
                common[v][u] = cnt

        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        max_iterations = 2000
        no_improve_count = 0
        
        for it in range(max_iterations):
            if time.time() - start_time > max_time - 5.0:
                break
                
            improved = False
            
            # 1. Try to add a random valid edge
            valid_edges = []
            for u, v in all_edges:
                if not ((adj_mask[u] >> v) & 1):
                    if is_valid_add(u, v, adj_mask, common):
                        valid_edges.append((u, v))
                        
            if valid_edges:
                u, v = random.choice(valid_edges)
                add_edge(u, v, adj_mask, common, edges)
                improved = True
                no_improve_count = 0
                if len(edges) > best_local_count:
                    best_local_count = len(edges)
                    best_local_edges = list(edges)
            else:
                # 2. Try a swap: remove one, add one
                # Pick a random edge to remove
                if not edges:
                    break
                    
                # We want to find a swap that increases edge count.
                # Since we are at a local optimum for additions, any swap must remove 1 and add 1 (net 0) 
                # or remove 1 and add 2 (net +1, but we couldn't add any single edge, so adding 2 might be possible if the first addition enables the second? 
                # No, if no single edge can be added, adding 2 edges $(x,y)$ and $(a,b)$ might be possible if they don't conflict individually with the rest, but do they conflict with each other?
                # If $(x,y)$ is invalid alone, it means it conflicts with existing edges. Adding $(a,b)$ doesn't remove the conflict.
                # So we can only do swaps that maintain or increase count.
                # Actually, if we are stuck at local optimum for +1, we might be able to do -1/+1 to move to a different region.
                
                # Try a random swap
                idx = random.randint(0, len(edges) - 1)
                u_rem, v_rem = edges[idx]
                
                # Save state for rollback
                old_adj_mask = list(adj_mask)
                old_common = [row[:] for row in common]
                old_edges = list(edges)
                
                remove_edge(u_rem, v_rem, adj_mask, common, edges)
                
                # Now try to add a random missing edge
                valid_adds = []
                for u, v in all_edges:
                    if not ((adj_mask[u] >> v) & 1):
                        if is_valid_add(u, v, adj_mask, common):
                            valid_adds.append((u, v))
                            
                if valid_adds:
                    u_add, v_add = random.choice(valid_adds)
                    add_edge(u_add, v_add, adj_mask, common, edges)
                    # Net change 0. We moved.
                    # Check if this new state allows further additions?
                    # We'll just let the next iteration handle additions.
                    improved = True # We moved
                else:
                    # Rollback
                    adj_mask = old_adj_mask
                    common = old_common
                    edges = old_edges
                    # If we can't even swap, we are very stuck.
                    # Maybe try removing 2 edges?
                    # For now, just count as no improvement.
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break

            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break

        # Final fill
        if time.time() - start_time < max_time - 3.0:
            fill_edges(adj_mask, common, edges)
            if len(edges) > best_local_count:
                best_local_count = len(edges)
                best_local_edges = list(edges)

        return best_local_edges

    # Main loop
    best_edges = []
    best_count = 0
    seed = 42
    
    while time.time() - start_time < max_time - 10.0:
        # Greedy Init
        init_edges = []
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, init_edges)
                
        local_best_edges = run_local_search(init_edges, seed)
        
        if len(local_best_edges) > best_count:
            best_count = len(local_best_edges)
            best_edges = local_best_edges
            
        seed += 1

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
