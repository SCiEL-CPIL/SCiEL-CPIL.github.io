import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Clebsch Graph Construction
    def get_clebsch():
        # Vertices are 5-bit vectors with even parity
        vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
        adj = [0] * n
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj

    best_adj = get_clebsch()
    best_count = 0
    for i in range(n):
        best_count += bin(best_adj[i]).count('1')
    best_count //= 2
    
    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def is_triangle_free(adj):
        for i in range(n):
            # Check all pairs of neighbors
            # Or simpler: for each edge (i,j), check common neighbors
            neighbors_i = adj[i]
            while neighbors_i:
                lsb = neighbors_i & -neighbors_i
                j = lsb.bit_length() - 1
                neighbors_i -= lsb
                if adj[i] & (1 << j): # It is an edge
                    if (adj[i] & adj[j]) != 0:
                        return False
        return True

    def is_3_colorable(adj):
        # Backtracking with MRV
        colors = [-1] * n
        # Order vertices by degree descending to prune faster
        vertices = list(range(n))
        vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = vertices[idx]
            # Determine available colors
            # Neighbor colors
            used_colors = 0
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & -neighbors
                u = lsb.bit_length() - 1
                neighbors -= lsb
                if colors[u] != -1:
                    used_colors |= (1 << colors[u])
            
            # Try available colors
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Initial Check
    if not is_3_colorable(best_adj):
        best_count = count_edges(best_adj)
    else:
        # Clebsch is 4-chromatic, so this shouldn't happen, but just in case
        best_count = -1

    # Search
    # We want to maximize edges.
    # Current best is 40.
    # Let's try to find graphs with > 40 edges.
    
    # Simulated Annealing
    # State: adj
    # Energy: -edges * 10 + (1 if 3-colorable else 0) * 100
    
    def energy(adj):
        e = count_edges(adj)
        col = 1 if is_3_colorable(adj) else 0
        return -e * 10 + col * 100

    current_adj = best_adj[:]
    current_energy = energy(current_adj)
    best_energy = current_energy
    
    T = 100.0
    T_min = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        new_adj = current_adj[:]
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick random non-edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if not (current_adj[u] & (1 << v)):
                # Check triangle freeness
                if (current_adj[u] & current_adj[v]) == 0:
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                else:
                    continue # Reject triangle
            else:
                continue
                
        elif move_type == 'remove':
            # Pick random edge
            u = random.randint(0, n-1)
            # Pick a neighbor
            neighbors = current_adj[u]
            if neighbors == 0: continue
            # Randomly pick a bit
            # Simple way: list of neighbors
            nbrs = [i for i in range(n) if current_adj[u] & (1 << i)]
            if not nbrs: continue
            v = random.choice(nbrs)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
        else: # swap
            # Remove edge (u,v) and add (x,y)
            # Find an edge to remove
            edges = []
            for i in range(n):
                nbrs = [j for j in range(i+1, n) if current_adj[i] & (1 << j)]
                for j in nbrs:
                    edges.append((i,j))
            if not edges: continue
            u, v = random.choice(edges)
            
            # Find a non-edge to add that doesn't create triangle
            # Try a few candidates
            added = False
            for _ in range(10):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if current_adj[x] & (1 << y): continue # Already edge
                # Check if adding (x,y) creates triangle in graph WITHOUT (u,v)
                # Temporarily remove (u,v) from a copy?
                # Or just check:
                # If we remove (u,v), does (x,y) create a triangle?
                # Triangles involving x or y.
                # Common neighbors of x and y in G \ {(u,v)}
                # If x or y is u or v, it's complicated.
                # Let's just apply the move to a temp graph and check.
                temp_adj = current_adj[:]
                temp_adj[u] &= ~(1 << v)
                temp_adj[v] &= ~(1 << u)
                
                if (temp_adj[x] & temp_adj[y]) == 0:
                    new_adj = temp_adj[:]
                    new_adj[x] |= (1 << y)
                    new_adj[y] |= (1 << x)
                    added = True
                    break
            if not added:
                continue

        # Calculate new energy
        new_energy = energy(new_adj)
        
        # Acceptance probability
        delta_e = new_energy - current_energy
        if delta_e < 0:
            current_adj = new_adj
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = current_adj[:]
        else:
            prob = random.expovariate(1.0)
            if prob > delta_e / T:
                current_adj = new_adj
                current_energy = new_energy
        
        T *= cooling

    # Final verification
    if not is_triangle_free(best_adj):
        # Fallback to Clebsch
        best_adj = get_clebsch()
    
    # Return edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges

