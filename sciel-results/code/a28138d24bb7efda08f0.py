import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # Target size
    N = 42

    # Initialize with Paley graph on 37 vertices, then random for the rest
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    adj = [0] * N
    # Fill Paley part
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Randomly initialize the rest
    for i in range(q, N):
        for j in range(i + 1, N):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Connect to Paley part randomly
        for j in range(q):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Helper: Count triangles in a set of vertices induced by a graph represented by adj list
    # mask is a bitmask of vertices
    def count_triangles(mask, adj_list):
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        count = 0
        n_v = len(verts)
        for i in range(n_v):
            ui = verts[i]
            for j in range(i + 1, n_v):
                uj = verts[j]
                if adj_list[ui] & (1 << uj):
                    # Check common neighbors in mask
                    common = adj_list[ui] & adj_list[uj] & mask
                    # Count bits in common
                    count += bin(common).count('1')
        return count // 3 # Each triangle counted 3 times? 
        # Wait. For each edge (ui, uj), we count common neighbors.
        # A triangle (u, v, w) is counted for edge (u,v) as w, for (u,w) as v, for (v,w) as u.
        # So yes, divide by 3.

    # Calculate total energy
    def get_energy():
        e = 0
        # Count K5
        # A K5 is 5 vertices all connected.
        # We can count K5 by iterating over edges? No, too slow.
        # Iterate over all 5-sets? C(42, 5) = 850,668.
        # This is too slow to do every SA step.
        # We need delta energy.
        pass

    # We will use Delta Energy approach.
    # Initial Energy Calculation
    # We need to count all K5 and I5 in the initial graph.
    # This is expensive but done once.
    
    def count_k5s():
        count = 0
        # Optimized: Iterate over 4-cliques?
        # Or just brute force 5-sets if N is small?
        # C(42, 5) ~ 850k. 
        # Check each 5-set: 10 edges. 8.5M ops. ~1-2s in Python.
        # Acceptable for initialization.
        for quad in combinations(range(N), 5):
            is_k5 = True
            for u, v in combinations(quad, 2):
                if not (adj[u] & (1 << v)):
                    is_k5 = False
                    break
            if is_k5:
                count += 1
        return count

    def count_i5s():
        count = 0
        for quad in combinations(range(N), 5):
            is_i5 = True
            for u, v in combinations(quad, 2):
                if (adj[u] & (1 << v)):
                    is_i5 = False
                    break
            if is_i5:
                count += 1
        return count

    current_energy = count_k5s() + count_i5s()
    
    def delta_k5(u, v, adding):
        # Change in number of K5s when flipping (u,v)
        if adding:
            # New K5s must contain (u,v).
            # Other 3 vertices must be common neighbors of u and v, and form a triangle.
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            return count_triangles(common, adj)
        else:
            # Removing (u,v) destroys K5s containing (u,v).
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            return -count_triangles(common, adj)

    def delta_i5(u, v, adding):
        # Change in number of I5s (independent sets of size 5) in G.
        # I5 in G is K5 in complement G_bar.
        # Edge (u,v) in G is Non-Edge in G_bar.
        # If adding (u,v) to G, we remove (u,v) from G_bar.
        # This destroys K5s in G_bar that contain (u,v).
        # In G_bar, neighbors of u are non-neighbors in G.
        # common_bar = non_common in G.
        # non_common = ~(adj[u] | adj[v]) & full_mask
        full_mask = (1 << N) - 1
        non_common = ~(adj[u] | adj[v]) & full_mask
        non_common &= ~(1 << u)
        non_common &= ~(1 << v)
        
        # We need to count triangles in G_bar induced by non_common.
        # This is equivalent to counting independent triangles in G induced by non_common.
        # An independent triangle in G is a set of 3 vertices with no edges between them.
        # This is hard to count directly with adj.
        # But we can use the same logic:
        # In G_bar, a triangle is a set of 3 vertices where each pair is connected in G_bar.
        # Connected in G_bar means NOT connected in G.
        # So we need to count triples {x,y,z} in non_common such that:
        # (x,y) not in G, (y,z) not in G, (x,z) not in G.
        
        # We can write a helper count_independent_triangles(mask, adj)
        def count_ind_triangles(mask, adj_list):
            verts = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            
            count = 0
            n_v = len(verts)
            for i in range(n_v):
                ui = verts[i]
                for j in range(i + 1, n_v):
                    uj = verts[j]
                    if not (adj_list[ui] & (1 << uj)):
                        # ui, uj are non-adjacent in G.
                        # Need w such that w is non-adjacent to ui and uj.
                        # w in mask, w != u, w != v.
                        # common_non = ~(adj_list[ui] | adj_list[uj]) & mask
                        common_non = ~(adj_list[ui] | adj_list[uj]) & mask
                        common_non &= ~(1 << ui)
                        common_non &= ~(1 << uj)
                        count += bin(common_non).count('1')
            return count // 3

        if adding:
            # Adding edge to G removes edge from G_bar.
            # Destroys K5s in G_bar containing (u,v).
            # So we subtract the number of such K5s.
            return -count_ind_triangles(non_common, adj)
        else:
            # Removing edge from G adds edge to G_bar.
            # Creates K5s in G_bar containing (u,v).
            return count_ind_triangles(non_common, adj)

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000
    it = 0
    
    best_energy = current_energy
    best_adj = adj[:]
    
    while time.time() < deadline and it < max_iter:
        it += 1
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_edge = bool(adj[u] & (1 << v))
        adding = not is_edge
        
        d_k5 = delta_k5(u, v, adding)
        d_i5 = delta_i5(u, v, adding)
        delta = d_k5 + d_i5
        
        new_energy = current_energy + delta
        
        if new_energy < current_energy or random.random() < pow(2.718281828, -delta / T):
            # Accept
            if adding:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj[:]
                
                if current_energy == 0:
                    break
        else:
            pass # Reject
            
        T = max(T_min, T * cooling)

    # Reconstruct edges from best_adj
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
                
    # Verify validity (optional but good for safety)
    # If best_energy > 0, the graph is invalid.
    # The task says "Every such graph... proves R(5,5) > n".
    # If we return an invalid graph, score is 0.
    # So we must ensure we return a valid graph.
    # If we didn't reach 0 energy, we should fallback to a known valid graph, e.g., Paley(37).
    
    if best_energy > 0:
        # Fallback to Paley(37)
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        n = q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return n, edges
        
    return N, edges

# EVOLVE-BLOCK-END
