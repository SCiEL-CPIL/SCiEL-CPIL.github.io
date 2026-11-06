import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    N = 20
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges = 0
    best_adj = [0] * N
    
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            mask = adj[i]
            while mask:
                # Find lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def popcount(x):
        return bin(x).count('1')

    def can_add(adj, u, v):
        # Check if adding (u, v) violates K2,3 condition
        # Condition:
        # For all x in N(v), popcount(N(u) & N(x)) <= 1
        # For all y in N(u), popcount(N(v) & N(y)) <= 1
        
        # Note: u is not in N(u), v is not in N(v)
        
        mask_v = adj[v]
        mask_u = adj[u]
        
        # Check x in N(v)
        m = mask_v
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if popcount(mask_u & adj[x]) > 1:
                return False
            m ^= lsb
            
        # Check y in N(u)
        m = mask_u
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            if popcount(mask_v & adj[y]) > 1:
                return False
            m ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_edge(adj, u, v):
        return (adj[u] & (1 << v)) != 0

    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(N)) // 2

    def local_search(adj, time_left):
        # Greedy fill first
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            random.shuffle(all_edges)
            for u, v in all_edges:
                if not has_edge(adj, u, v) and can_add(adj, u, v):
                    add_edge(adj, u, v)
                    changed = True
        
        # Local search moves
        current_edges = count_edges(adj)
        no_improvement_count = 0
        
        while time.time() - start_time < time_limit:
            # Try to add any missing edge
            added = False
            # Pick a random subset of missing edges to try
            missing = [(u, v) for u, v in all_edges if not has_edge(adj, u, v)]
            if missing:
                random.shuffle(missing)
                # Try a few
                for u, v in missing[:100]:
                    if can_add(adj, u, v):
                        add_edge(adj, u, v)
                        added = True
                        current_edges += 1
                        break
            if added:
                no_improvement_count = 0
                continue
            
            # If no edge could be added, try a kick
            # Remove a random edge
            existing = [(u, v) for u, v in all_edges if has_edge(adj, u, v)]
            if not existing:
                break
            
            u, v = random.choice(existing)
            remove_edge(adj, u, v)
            current_edges -= 1
            
            # Try to add edges again
            added_any = False
            random.shuffle(all_edges)
            for a, b in all_edges:
                if not has_edge(adj, a, b) and can_add(adj, a, b):
                    add_edge(adj, a, b)
                    current_edges += 1
                    added_any = True
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
            
            if current_edges <= best_edges - 5:
                # Too far from best, restart
                break
                
            no_improvement_count += 1
            if no_improvement_count > 100:
                break

        return current_edges, adj

    # Run multiple restarts
    while time.time() - start_time < time_limit:
        adj = [0] * N
        # Random greedy initialization
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        
        # Local search
        edges, adj = local_search(adj, time_limit)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # If we hit a high score, we might want to polish more
        # but for now, just restart

    return get_edge_list(best_adj)

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Edges found: {len(edges)}")
