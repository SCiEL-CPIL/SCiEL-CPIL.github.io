import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 20
    rng = random.Random(12345)
    
    # Precompute all triples for fast iteration
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    num_triples = len(triples)
    
    def make_pair_counts(adj):
        """Compute for each pair (i,j) the count of common neighbors (excluding i,j)."""
        pc = [[0]*n for _ in range(n)]
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                common = ai & adj[j]
                mask = (1 << i) | (1 << j)
                common &= ~mask
                pc[i][j] = pc[j][i] = bin(common).count('1')
        return pc
    
    def is_valid(adj, pc):
        """Check K3,3-freedom using precomputed pair counts."""
        for (i, j, k) in triples:
            # Common neighbors of all three
            common = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            common &= ~mask
            if bin(common).count('1') >= 3:
                return False
        return True
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    best_edges = 0
    best_adj = None
    
    # Try multiple random starting graphs
    num_starts = 8
    per_start_time = 14.0  # seconds per start
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Create random starting graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.55:
                    add_edge(adj, i, j)
        
        # Clean: remove edges until valid
        while True:
            if time.time() > deadline - 5:
                break
            pc = make_pair_counts(adj)
            if is_valid(adj, pc):
                break
            # Find a violation and remove an edge
            removed = False
            for (i, j, k) in triples:
                common = adj[i] & adj[j] & adj[k]
                mask = (1 << i) | (1 << j) | (1 << k)
                common &= ~mask
                if bin(common).count('1') >= 3:
                    # Remove a random edge between triple and common
                    v_list = [x for x in range(n) if (common >> x) & 1]
                    u_list = [i, j, k]
                    ru = rng.choice(u_list)
                    rv = rng.choice(v_list)
                    remove_edge(adj, ru, rv)
                    removed = True
                    break
            if not removed:
                break
        
        if time.time() > deadline - 5:
            break
        
        # Verify it's valid
        pc = make_pair_counts(adj)
        if not is_valid(adj, pc):
            continue
        
        cur_edges = count_edges(adj)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
        
        # Greedy edge addition
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < start_time + per_start_time * (start_idx + 1) and time.time() < deadline - 5:
            rng.shuffle(all_edges)
            improved = False
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                
                add_edge(adj, u, v)
                pc = make_pair_counts(adj)
                if is_valid(adj, pc):
                    cur_edges += 1
                    improved = True
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = [a for a in adj]
                else:
                    remove_edge(adj, u, v)
            
            if not improved:
                break
        
        # Local search: edge swaps (remove one, add one)
        if time.time() < deadline - 10:
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            swap_rounds = 0
            max_swap_rounds = 50
            
            while swap_rounds < max_swap_rounds and time.time() < deadline - 5:
                swap_rounds += 1
                rng.shuffle(all_edges)
                any_improved = False
                
                # Try removing an edge and adding a different one
                for (u1, v1) in all_edges:
                    if time.time() > deadline - 5:
                        break
                    if not ((adj[u1] >> v1) & 1):
                        continue
                    
                    remove_edge(adj, u1, v1)
                    cur_edges -= 1
                    
                    # Try to add a different edge
                    for (u2, v2) in all_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if (adj[u2] >> v2) & 1:
                            continue
                        
                        add_edge(adj, u2, v2)
                        pc = make_pair_counts(adj)
                        if is_valid(adj, pc):
                            cur_edges += 1
                            any_improved = True
                            if cur_edges > best_edges:
                                best_edges = cur_edges
                                best_adj = [a for a in adj]
                            break
                        else:
                            remove_edge(adj, u2, v2)
                    
                    if not any_improved:
                        # Revert the removal
                        add_edge(adj, u1, v1)
                        cur_edges += 1
                    
                    if any_improved:
                        any_improved = False  # Reset for next iteration
                        break
                
                if not any_improved:
                    break
    
    # If we never found a valid graph, return empty
    if best_adj is None:
        return []
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
