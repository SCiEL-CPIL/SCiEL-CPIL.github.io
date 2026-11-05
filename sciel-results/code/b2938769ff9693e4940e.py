import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c) and N(v), and not u, a, b, v
                    d_mask = adj[c] & adj[v] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
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

    def greedy_add(adj, deadline):
        # Iteratively add all possible edges until saturation
        # Returns the number of edges added in this pass
        added = 0
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order to break symmetry
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline:
                    return added
                if not ((1 << v) & adj[u]):
                    if not path5(u, v, adj):
                        add_edge(adj, u, v)
                        added += 1
                        changed = True
        return added

    def local_search_swap(adj, deadline):
        # Swap-based local search: remove an edge, try to add others
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
            
            if not current_edges:
                break
                
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() > deadline:
                    break
                
                # Try removing (u, v)
                remove_edge(adj, u, v)
                
                # Try to add as many missing edges as possible
                # Iterate through all missing edges in random order
                missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((1 << j) & adj[i])]
                random.shuffle(missing)
                
                added_edges = []
                for i, j in missing:
                    if time.time() > deadline:
                        break
                    if not path5(i, j, adj):
                        add_edge(adj, i, j)
                        added_edges.append((i, j))
                
                new_count = count_edges(adj)
                old_count = new_count - len(added_edges) + 1  # +1 for the removed edge (u,v)
                
                if len(added_edges) > 1:  # Net gain > 0 (added > 1 to compensate for removed 1)
                    improved = True
                    # Keep the changes
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)

    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with 8 disjoint K5s
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)
        
        # Phase 1: Greedy Add (saturation)
        greedy_add(adj, deadline)
        
        # Phase 2: Local Search (Swap-based)
        local_search_swap(adj, deadline)
        
        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
