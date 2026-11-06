import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # State
    adj = [0] * n
    common = [[0] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common counts
        # For w in N(v) \ {u}, v becomes common neighbor of u and w
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            mask_v ^= lsb
            
        # For w in N(u) \ {v}, u becomes common neighbor of v and w
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            mask_u ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Reverse updates
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            mask_v ^= lsb
            
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            mask_u ^= lsb

    def is_valid_addition(u, v):
        if adj[u] & (1 << v):
            return True
        # 1. Common neighbors of u and v must be < 3
        if common[u][v] >= 3:
            return False
        
        # 2. For any w in N(v) \ {u}, v becomes a common neighbor of u and w.
        #    So common[u][w] must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = lsb.bit_length() - 1
            if common[u][w] >= 2:
                return False
            neighbors_v ^= lsb
            
        # 3. Similarly for w in N(u) \ {v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = lsb.bit_length() - 1
            if common[v][w] >= 2:
                return False
            neighbors_u ^= lsb
            
        return True

    def get_edge_count():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def extract_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def try_add_all():
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            # Try adding any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_addition(u, v):
                            add_edge(u, v)
                            improved = True
        return True

    def local_search():
        # Hill climbing with swaps
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            current_count = get_edge_count()
            
            # Try removing each edge and seeing if we can gain more
            edges_to_try = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges_to_try.append((u, v))
            
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                
                # Save state
                adj_backup = adj[:]
                common_backup = [row[:] for row in common]
                
                # Remove edge
                remove_edge(u, v)
                
                # Try to add all possible edges
                try_add_all()
                
                new_count = get_edge_count()
                
                if new_count > current_count:
                    improved = True
                    current_count = new_count
                    # Keep the change (state is already updated)
                else:
                    # Restore state
                    adj[:] = adj_backup
                    for i in range(n):
                        common[i] = common_backup[i]
                        
        try_add_all()

    best_edges = []
    best_count = 0
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Reset state
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        
        # Greedy initialization
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_valid_addition(u, v):
                add_edge(u, v)
                
        # Local search
        local_search()
        
        # Update best
        count = get_edge_count()
        if count > best_count:
            best_count = count
            best_edges = extract_edges()
            
        # If we find a very good solution, we can stop earlier or continue with tighter time
        if best_count >= 125 and time.time() > start_time + 90:
            break

    return best_edges
