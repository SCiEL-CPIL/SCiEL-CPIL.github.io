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
        # Checks if adding (u, v) creates a C6
        # Path u-a-b-c-d-v
        # u, v are not connected in adj yet.
        nu = adj[u] & ~(1<<v)
        nv = adj[v] & ~(1<<u)
        
        a_mask = nu
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
                    d_mask = adj[c] & nv & ~(1<<u) & ~(1<<a) & ~(1<<b) & ~(1<<v)
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

    def greedy_add(adj):
        improved = True
        while improved:
            improved = False
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((1 << j) & adj[i]):
                        missing.append((i, j))
            if not missing:
                break
            random.shuffle(missing)
            for i, j in missing:
                if time.time() > deadline:
                    return adj
                if not path5(i, j, adj):
                    add_edge(adj, i, j)
                    improved = True
        return adj

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
        greedy_add(adj)
        
        # Phase 2: Randomized Local Search (Swaps)
        # We run this for a fixed number of steps or until time limit
        steps = 10000
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Build lists
            current_edges = []
            missing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
                    else:
                        missing_edges.append((i, j))
            
            if not current_edges or not missing_edges:
                break
                
            # Pick random edge to remove
            u1, v1 = random.choice(current_edges)
            # Pick random edge to add
            u2, v2 = random.choice(missing_edges)
            
            # Remove u1-v1
            remove_edge(adj, u1, v1)
            
            # Check if adding u2-v2 creates C6
            if not path5(u2, v2, adj):
                add_edge(adj, u2, v2)
                # Keep
            else:
                # Revert
                add_edge(adj, u1, v1)
        
        # Final Greedy Add
        greedy_add(adj)
        
        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
