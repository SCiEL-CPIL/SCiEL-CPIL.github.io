import time
import random
import sys

def construct():
    n = 20
    deadline = time.time() + 110.0
    
    # Initialize adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    # Helper to get neighbors
    def get_neighbors(u):
        return [v for v in range(n) if adj[u][v]]
        
    # Helper to check if adding (u, v) is valid
    def can_add(u, v):
        if adj[u][v]:
            return False
        if u == v:
            return False
            
        # 1. Check common neighbors of u and v
        common_uv = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        # 2. Check neighbors of v (excluding u)
        for w in get_neighbors(v):
            if w == u:
                continue
            # Check pair (u, w)
            # New common neighbors will include v (since v is neighbor of u and w)
            # Current common neighbors of u and w:
            common_uw = 0
            for x in range(n):
                if adj[u][x] and adj[w][x]:
                    common_uw += 1
                    if common_uw > 1: # Because v will be added, so max allowed current is 1
                        return False
                        
        # 3. Check neighbors of u (excluding v)
        for w in get_neighbors(u):
            if w == v:
                continue
            # Check pair (v, w)
            common_vw = 0
            for x in range(n):
                if adj[v][x] and adj[w][x]:
                    common_vw += 1
                    if common_vw > 1:
                        return False
                        
        return True

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True

    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False

    def count_edges():
        return sum(1 for u in range(n) for v in range(u+1, n) if adj[u][v])

    # --- Initialization: Clebsch Graph + 4 vertices ---
    # Clebsch graph construction
    # Vertices 0..15
    # Edges: distance 1 in Q5 folded.
    # Q5 vertices 0..31. Fold x ~ x^31.
    # Reps: 0..15.
    def get_rep(x):
        return min(x, x ^ 31)
        
    clebsch_edges = set()
    for u in range(32):
        for b in range(5):
            v = u ^ (1 << b)
            if u < v:
                cu = get_rep(u)
                cv = get_rep(v)
                if cu != cv:
                    clebsch_edges.add((cu, cv))
                    
    for u, v in clebsch_edges:
        add_edge(u, v)
        
    # Add edges for vertices 16, 17, 18, 19
    # Each can connect to at most 2 adjacent vertices in Clebsch.
    # Let's pick random adjacent pairs.
    # List of edges in Clebsch for random selection
    cleb_edge_list = list(clebsch_edges)
    for i in range(4):
        new_v = 16 + i
        if cleb_edge_list:
            u, v = random.choice(cleb_edge_list)
            add_edge(new_v, u)
            add_edge(new_v, v)
            
    # Local Search
    best_edges = count_edges()
    current_edges = best_edges
    
    # Precompute list of all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Try to add a random edge
        u, v = random.choice(all_edges)
        if not adj[u][v] and can_add(u, v):
            add_edge(u, v)
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                
        # Try 1-for-2 swap
        # Remove a random edge
        existing_edges = [(u, v) for u, v in all_edges if adj[u][v]]
        if existing_edges:
            rem_u, rem_v = random.choice(existing_edges)
            # Save state to rollback if fails? 
            # Better: try to find 2 edges to add, then commit.
            # But adding depends on the removal.
            # So: Remove, try add 1, try add 2. If both fail, rollback?
            # Or: Simulate.
            
            # Simulate removal
            adj[rem_u][rem_v] = False
            adj[rem_v][rem_u] = False
            
            added_count = 0
            added_edges = []
            
            # Try to add 2 random edges
            # We need to pick 2 distinct non-edges
            # To be efficient, pick random non-edges
            non_edges = [(u, v) for u, v in all_edges if not adj[u][v]]
            if len(non_edges) >= 2:
                # Pick 2
                e1 = random.choice(non_edges)
                e2 = random.choice(non_edges)
                if e1 == e2:
                    e2 = random.choice([e for e in non_edges if e != e1])
                    
                if can_add(e1[0], e1[1]):
                    add_edge(e1[0], e1[1])
                    added_count += 1
                    added_edges.append(e1)
                    
                if can_add(e2[0], e2[1]):
                    add_edge(e2[0], e2[1])
                    added_count += 1
                    added_edges.append(e2)
                    
            if added_count > 1:
                # Net gain >= 0 (if 2 added, net +1; if 1 added, net 0)
                # We want net gain > 0, so added_count must be 2.
                if added_count == 2:
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                else:
                    # Rollback
                    for eu, ev in added_edges:
                        remove_edge(eu, ev)
                    add_edge(rem_u, rem_v)
            else:
                # Rollback
                for eu, ev in added_edges:
                    remove_edge(eu, ev)
                add_edge(rem_u, rem_v)
                
        # Occasionally, if stuck, do a 1-for-1 swap to explore
        # This is a bit expensive to check "stuck", so just do it rarely
        if random.random() < 0.05:
            existing_edges = [(u, v) for u, v in all_edges if adj[u][v]]
            if existing_edges:
                rem_u, rem_v = random.choice(existing_edges)
                adj[rem_u][rem_v] = False
                adj[rem_v][rem_u] = False
                
                non_edges = [(u, v) for u, v in all_edges if not adj[u][v]]
                if non_edges:
                    add_u, add_v = random.choice(non_edges)
                    if can_add(add_u, add_v):
                        add_edge(add_u, add_v)
                        # current_edges unchanged
                    else:
                        add_edge(rem_u, rem_v)
                else:
                    add_edge(rem_u, rem_v)

    # Return best edges found
    # Note: The state 'adj' might not correspond to 'best_edges' if we didn't track it.
    # Since we only commit improvements or neutral moves, and rollback bad moves,
    # the current 'adj' state is always valid.
    # However, 'best_edges' might be higher if we found a better config and then
    # performed neutral moves that didn't increase it, or if we rolled back from a peak.
    # To be safe, we should store the best graph.
    
    # Let's refine: store best_adj
    # Re-running the whole logic with best_adj storage is safer.

def construct():
    n = 20
    deadline = time.time() + 110.0
    
    adj = [[False] * n for _ in range(n)]
    best_adj = [[False] * n for _ in range(n)]
    best_edges = 0
    
    def get_neighbors(u, a):
        return [v for v in range(n) if a[u][v]]
        
    def can_add(u, v, a):
        if a[u][v]:
            return False
        if u == v:
            return False
            
        common_uv = 0
        for w in range(n):
            if a[u][w] and a[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        for w in get_neighbors(v, a):
            if w == u:
                continue
            common_uw = 0
            for x in range(n):
                if a[u][x] and a[w][x]:
                    common_uw += 1
                    if common_uw > 1:
                        return False
                        
        for w in get_neighbors(u, a):
            if w == v:
                continue
            common_vw = 0
            for x in range(n):
                if a[v][x] and a[w][x]:
                    common_vw += 1
                    if common_vw > 1:
                        return False
                        
        return True

    def add_edge(u, v, a):
        a[u][v] = True
        a[v][u] = True

    def remove_edge(u, v, a):
        a[u][v] = False
        a[v][u] = False

    def count_edges(a):
        return sum(1 for u in range(n) for v in range(u+1, n) if a[u][v])

    def copy_adj(a):
        return [row[:] for row in a]

    # Init
    def get_rep(x):
        return min(x, x ^ 31)
        
    clebsch_edges = set()
    for u in range(32):
        for b in range(5):
            v = u ^ (1 << b)
            if u < v:
                cu = get_rep(u)
                cv = get_rep(v)
                if cu != cv:
                    clebsch_edges.add((cu, cv))
                    
    for u, v in clebsch_edges:
        add_edge(u, v, adj)
        
    cleb_edge_list = list(clebsch_edges)
    for i in range(4):
        new_v = 16 + i
        if cleb_edge_list:
            u, v = random.choice(cleb_edge_list)
            add_edge(new_v, u, adj)
            add_edge(new_v, v, adj)
            
    best_adj = copy_adj(adj)
    best_edges = count_edges(adj)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Try add
        u, v = random.choice(all_edges)
        if not adj[u][v] and can_add(u, v, adj):
            add_edge(u, v, adj)
            cur_e = count_edges(adj)
            if cur_e > best_edges:
                best_edges = cur_e
                best_adj = copy_adj(adj)
                
        # Try 1-for-2
        existing = [(u, v) for u, v in all_edges if adj[u][v]]
        if existing:
            rem_u, rem_v = random.choice(existing)
            remove_edge(rem_u, rem_v, adj)
            
            added = []
            non_edges = [(u, v) for u, v in all_edges if not adj[u][v]]
            if len(non_edges) >= 2:
                e1 = random.choice(non_edges)
                if can_add(e1[0], e1[1], adj):
                    add_edge(e1[0], e1[1], adj)
                    added.append(e1)
                    
                # Refresh non_edges? No, just pick another distinct one
                e2_candidates = [e for e in non_edges if e != e1]
                if e2_candidates:
                    e2 = random.choice(e2_candidates)
                    if can_add(e2[0], e2[1], adj):
                        add_edge(e2[0], e2[1], adj)
                        added.append(e2)
                        
            if len(added) == 2:
                cur_e = count_edges(adj)
                if cur_e > best_edges:
                    best_edges = cur_e
                    best_adj = copy_adj(adj)
            else:
                # Rollback
                for eu, ev in added:
                    remove_edge(eu, ev, adj)
                add_edge(rem_u, rem_v, adj)
                
        # 1-for-1 swap (exploration)
        if random.random() < 0.1:
            existing = [(u, v) for u, v in all_edges if adj[u][v]]
            if existing:
                rem_u, rem_v = random.choice(existing)
                remove_edge(rem_u, rem_v, adj)
                
                non_edges = [(u, v) for u, v in all_edges if not adj[u][v]]
                if non_edges:
                    add_u, add_v = random.choice(non_edges)
                    if can_add(add_u, add_v, adj):
                        add_edge(add_u, add_v, adj)
                    else:
                        add_edge(rem_u, rem_v, adj)
                else:
                    add_edge(rem_u, rem_v, adj)

    # Return edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
