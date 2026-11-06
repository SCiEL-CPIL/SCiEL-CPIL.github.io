import time
import random

def _has_k222(adj, n):
    """
    Check if graph with adjacency bitmasks adj on n vertices contains K_{2,2,2} as a subgraph.
    """
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            
            if cn_mask.bit_count() < 4:
                continue
            
            # Check if induced subgraph on cn_mask contains K_{2,2}
            # This happens if there are two vertices u, v in cn_mask with at least 2 common neighbors in cn_mask
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            for i in range(len(cn_list)):
                u = cn_list[i]
                adj_u = adj[u]
                for j in range(i + 1, len(cn_list)):
                    v = cn_list[j]
                    common = adj_u & adj[v] & cn_mask
                    if common.bit_count() >= 2:
                        return True
    return False

def _count_k222(adj, n):
    """
    Count the number of K_{2,2,2} subgraphs in the graph.
    """
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            
            if cn_mask.bit_count() < 4:
                continue
            
            # Count K_{2,2} in induced subgraph on cn_mask
            # A K_{2,2} is defined by two disjoint pairs of vertices in cn_mask that are fully connected.
            # Or simply, for each pair (u,v) in cn_mask, if they have >=2 common neighbors in cn_mask, 
            # we have found a K_{2,2} structure involving u,v and those neighbors.
            # But we need to be careful not to double count or count non-distinct structures.
            # Actually, we just need to know if ANY K_{2,2} exists for validity.
            # For counting objective in search, we can just count the number of pairs (u,v) in cn_mask 
            # that have >=2 common neighbors in cn_mask. This is a proxy for the number of K_{2,2,2}s.
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            for i in range(len(cn_list)):
                u = cn_list[i]
                adj_u = adj[u]
                for j in range(i + 1, len(cn_list)):
                    v = cn_list[j]
                    common = adj_u & adj[v] & cn_mask
                    if common.bit_count() >= 2:
                        count += common.bit_count() # Weight by number of such structures
    return count

def _get_colors_from_adj(adj, n):
    """Get colors list from adjacency bitmasks."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _paley_adj(q):
    """Generate Paley graph adjacency bitmasks on q vertices."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _search_n30(deadline):
    """
    Attempt to find a valid K_{2,2,2}-free coloring for n=30.
    """
    n = 30
    full_mask = (1 << n) - 1
    
    # Start with a random graph or Paley(29)+1
    # Let's try a few random restarts
    random.seed(42)
    
    best_cost = float('inf')
    best_adj = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize adj
        # Option 1: Random
        # Option 2: Paley(29) + random vertex
        if restart % 2 == 0:
            adj29 = _paley_adj(29)
            adj = [0] * n
            for i in range(29):
                adj[i] = adj29[i]
                # Connect to vertex 29 randomly
                if random.random() < 0.5:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
            # Vertex 29 has no self-loop
            adj[29] &= ~(1 << 29)
        else:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        comp = [0] * n
        for i in range(n):
            comp[i] = full_mask ^ adj[i] ^ (1 << i)
            
        cost = _count_k222(adj, n) + _count_k222(comp, n)
        
        if cost == 0:
            return adj
        
        if cost < best_cost:
            best_cost = cost
            best_adj = adj[:]
            
        # Local Search
        current_adj = adj[:]
        current_comp = comp[:]
        current_cost = cost
        
        no_improve_count = 0
        while time.time() < deadline - 10: # Leave some time for final checks
            if no_improve_count > 100:
                break
                
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            # Calculate new cost if we flip (i,j)
            # Flip in adj
            if (current_adj[i] >> j) & 1:
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
                current_comp[i] |= (1 << j)
                current_comp[j] |= (1 << i)
            else:
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
                current_comp[i] &= ~(1 << j)
                current_comp[j] &= ~(1 << i)
                
            new_cost = _count_k222(current_adj, n) + _count_k222(current_comp, n)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                if current_cost == 0:
                    return current_adj
            else:
                # Revert
                if (current_adj[i] >> j) & 1:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                    current_comp[i] |= (1 << j)
                    current_comp[j] |= (1 << i)
                else:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                    current_comp[i] &= ~(1 << j)
                    current_comp[j] &= ~(1 << i)
                no_improve_count += 1
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = current_adj[:]

    # If we found a best_adj with cost 0, return it
    if best_adj is not None:
        if _count_k222(best_adj, n) + _count_k222([full_mask ^ best_adj[i] ^ (1 << i) for i in range(n)], n) == 0:
            return best_adj
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find n=30
    adj30 = _search_n30(deadline)
    if adj30 is not None:
        colors = _get_colors_from_adj(adj30, 30)
        return {'n': 30, 'colors': colors}
        
    # Fallback to n=29
    q = 29
    adj = _paley_adj(q)
    colors = _get_colors_from_adj(adj, q)
    return {'n': q, 'colors': colors}
