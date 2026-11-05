import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        a_mask = adj[u] & ~(1<<v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~(1<<u) & ~(1<<v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~(1<<u) & ~(1<<a) & ~(1<<v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c) and N(v), and not u, a, b, v
                    d_mask = adj[c] & adj[v] & ~(1<<u) & ~(1<<a) & ~(1<<b) & ~(1<<v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (1 << j) & adj[i]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with 8 disjoint K5s
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)
        
        # Phase 1: Greedy Add
        # Try adding all missing edges
        # Randomize order to avoid bias?
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((1 << j) & adj[i]):
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if not path5(u, v, adj):
                add_edge(adj, u, v)
        
        # Phase 2: Local Search (Swap-based)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Get list of current edges to try removing
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
            
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() > deadline:
                    break
                
                # Try removing (u, v)
                remove_edge(adj, u, v)
                old_count = count_edges(adj)
                
                # Try to add as many missing edges as possible
                # We need to check which edges are now addable
                # To be efficient, we can just iterate through all missing edges
                # But we don't have a dynamic list.
                # Let's iterate through all pairs.
                added_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((1 << j) & adj[i]):
                            if not path5(i, j, adj):
                                add_edge(adj, i, j)
                                added_edges.append((i, j))
                
                new_count = count_edges(adj)
                
                if new_count > old_count + 1: # Net gain > 0
                    improved = True
                    # Keep the changes
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)
        
        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
