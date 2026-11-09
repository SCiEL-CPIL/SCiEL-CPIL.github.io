import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Generate all candidate nodes
    # Node id = index
    # Node data = (residue, modulus)
    nodes = []
    node_to_id = {}
    for m in moduli:
        for r in range(m):
            node_to_id[(r, m)] = len(nodes)
            nodes.append((r, m))
            
    N = len(nodes)
    
    # Precompute gcds for moduli
    # We need gcd(m_i, m_j) for all pairs.
    # But we need it for nodes.
    
    # Build Disjointness Graph adjacency bitmasks
    # adj[i] is a bitmask of j such that node i and node j are disjoint.
    adj = [0] * N
    
    # To speed up, we can group by modulus.
    # Two nodes with same modulus are NOT disjoint (they intersect).
    # So no edges within same modulus group in Disjointness Graph.
    
    # For nodes with different moduli m_i, m_j:
    # Disjoint if (r_i - r_j) % gcd(m_i, m_j) != 0
    
    # Optimization:
    # Iterate over pairs of moduli? Or pairs of nodes?
    # Pairs of nodes is N^2/2 ~ 32M operations. Might be slow in Python if not careful.
    # Let's try to optimize.
    
    # We can precompute gcd for moduli.
    gcd_cache = {}
    def get_gcd(m1, m2):
        key = (min(m1, m2), max(m1, m2))
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(m1, m2)
        return gcd_cache[key]
    
    # We will build adj bitmasks.
    # To make it faster, we can process modulus by modulus.
    # But bitmask operations are on the whole N.
    
    # Let's just do the double loop but with some optimization.
    # Actually, 8000 nodes. 8000*8000 = 64M checks.
    # In Python, 64M iterations might take ~5-10 seconds.
    # This is acceptable within 85s.
    
    # We can reduce checks:
    # If m1 == m2, no edge.
    # If gcd(m1, m2) == 1, they are ALWAYS disjoint?
    # No. Disjoint if (r1 - r2) % 1 != 0. 0 % 1 == 0. So they are NEVER disjoint?
    # Wait.
    # Condition for INTERSECTION: (r1 - r2) % gcd(m1, m2) == 0.
    # If gcd=1, (r1-r2) % 1 is always 0.
    # So if gcd(m1, m2) == 1, the classes ALWAYS intersect.
    # Therefore, there is NO edge in the Disjointness Graph if gcd(m1, m2) == 1.
    
    # This is a huge simplification!
    # Nodes with coprime moduli are NEVER disjoint.
    # So we only need to check pairs where gcd(m1, m2) > 1.
    
    # Let's filter pairs.
    
    # Group nodes by modulus
    nodes_by_mod = {}
    for idx, (r, m) in enumerate(nodes):
        if m not in nodes_by_mod:
            nodes_by_mod[m] = []
        nodes_by_mod[m].append(idx)
        
    # List of moduli that have gcd > 1 with other moduli
    # Actually, just iterate over all pairs of moduli (m1, m2) with m1 < m2.
    # If gcd(m1, m2) > 1, check all pairs of nodes (n1, n2) from these groups.
    
    # Number of such pairs is much smaller than N^2.
    
    # Let's do it.
    
    # Initialize adj as list of ints (bitmasks)
    # Python ints are arbitrary precision, so bit i is (1 << i)
    
    for i in range(len(moduli)):
        m1 = moduli[i]
        group1 = nodes_by_mod[m1]
        for j in range(i + 1, len(moduli)):
            m2 = moduli[j]
            g = math.gcd(m1, m2)
            if g == 1:
                continue
            
            group2 = nodes_by_mod[m2]
            
            # For each node in group1 and group2, check disjointness
            # (r1 - r2) % g != 0
            
            # We can optimize this inner loop.
            # For a fixed r1, we want r2 such that r2 % g != r1 % g.
            # Let's precompute residues mod g for group2?
            # Or just loop.
            # Size of groups is small (max 128).
            # Total pairs of moduli with gcd>1 is small.
            
            for idx1 in group1:
                r1 = nodes[idx1][0]
                r1_mod_g = r1 % g
                mask1 = 0
                for idx2 in group2:
                    r2 = nodes[idx2][0]
                    if (r1_mod_g - r2) % g != 0:
                        # Disjoint
                        mask1 |= (1 << idx2)
                adj[idx1] |= mask1
                # Symmetric
                # We can just build the full mask for idx1 from idx2's perspective later?
                # Or just set adj[idx2] |= (1 << idx1) here?
                # It's better to build full rows.
                
            # The above loop builds adj[idx1] partially.
            # We need to ensure symmetry.
            # Let's just set both.
            # But the loop above only sets adj[idx1].
            # Let's rewrite to set both efficiently or just accept O(N^2) logic but filtered.
            
    # The loop above is a bit messy. Let's restart the graph building logic cleanly.
    
    adj = [0] * N
    # We will iterate over pairs of moduli
    for i in range(len(moduli)):
        m1 = moduli[i]
        g1 = nodes_by_mod[m1]
        for j in range(i + 1, len(moduli)):
            m2 = moduli[j]
            g = math.gcd(m1, m2)
            if g == 1:
                continue
            
            g2 = nodes_by_mod[m2]
            
            # For all pairs (idx1, idx2)
            for idx1 in g1:
                r1 = nodes[idx1][0]
                r1_rem = r1 % g
                # Check all idx2 in g2
                for idx2 in g2:
                    r2 = nodes[idx2][0]
                    if (r1_rem - r2) % g != 0:
                        adj[idx1] |= (1 << idx2)
                        adj[idx2] |= (1 << idx1)
                        
    # Now we have the Disjointness Graph.
    # We want Max Clique.
    
    # Tomita's Algorithm for Max Clique
    # We use a recursive branch and bound.
    
    best_clique = []
    best_size = 0
    
    # We can use a greedy heuristic to find a good initial solution
    # Greedy: pick node with highest degree, add to clique, remove neighbors, repeat.
    # Or just use the result from the old program if we had it? No, we are replacing it.
    # Let's run a quick greedy.
    
    # Greedy Max Clique
    def greedy_max_clique():
        used = (1 << N) - 1
        clique = []
        while used:
            # Pick node with max degree in remaining graph
            # To speed up, just pick first one? Or max degree?
            # Max degree is better.
            # Calculating degree for all nodes is expensive.
            # Let's just pick a random one or the first one for simplicity, 
            # or we can pre-sort by degree.
            
            # Let's pre-sort nodes by degree (number of bits in adj[i])
            # degree[i] = adj[i].bit_count()
            
            # We'll do a simple greedy:
            # While used:
            #   Pick u in used with max (adj[u] & used).bit_count()
            #   Add u to clique
            #   used &= adj[u]  # Remove u and non-neighbors? 
            #   Wait, if we add u, we can only add neighbors of u.
            #   So new used = used & adj[u].
            
            # This is a standard greedy heuristic.
            
            # To avoid O(N^2) in the greedy, we can just iterate.
            # N is 8000. 8000 * 8000 is too slow for greedy if done naively.
            # But we can just pick the first available node?
            # Let's try picking the node with the highest index? No.
            
            # Let's just use a simple random greedy or first-available.
            # Given the structure, maybe just picking any is okay for a lower bound.
            
            # Let's try to pick the node with the highest degree in the subgraph.
            # We can approximate by pre-computed global degree.
            
            # For now, let's just run the B&B with a lower bound of 0 or a simple greedy.
            # A simple greedy:
            # Pick u = lowest set bit in used.
            # This might be weak.
            
            # Let's do a slightly better greedy:
            # Sort nodes by global degree descending.
            # Iterate through sorted nodes.
            # If node is in used and disjoint from all in current clique (which is ensured by used &= adj[u]), add it.
            
            # Actually, the B&B will find the optimum. The lower bound just helps pruning.
            # If we start with 0, it might take longer.
            # Let's run a quick greedy:
            
            used = (1 << N) - 1
            c = []
            # Sort nodes by degree
            degrees = [adj[i].bit_count() for i in range(N)]
            sorted_nodes = sorted(range(N), key=lambda x: degrees[x], reverse=True)
            
            for u in sorted_nodes:
                if (used >> u) & 1:
                    c.append(u)
                    used &= adj[u]
                    if not used:
                        break
            return c
            
    initial_clique = greedy_max_clique()
    best_size = len(initial_clique)
    best_clique = initial_clique
    
    # B&B
    # We need to order vertices for the B&B to be efficient.
    # Usually, ordering by degree helps.
    # We will use the sorted_nodes order for the B&B recursion.
    
    # Map original index to new index in sorted order?
    # Or just use the original indices but pick the "best" candidate from the set?
    # Tomita's algorithm usually processes vertices in a specific order.
    
    # Let's implement a standard Max Clique B&B with bitsets.
    
    # To make it fast, we should process vertices in an order that allows good coloring.
    # We'll stick to the original indices but use the `used` bitmask.
    
    # Function to color the subgraph induced by `used`
    # Returns list of colors and max color
    def color_bound(used):
        # Greedy coloring
        # We want to color the vertices in `used` such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the clique size.
        
        colors = []
        color_class = used
        color_idx = 0
        
        while color_class:
            color_idx += 1
            # Pick a vertex from color_class
            # To make coloring tighter, we should pick vertices carefully.
            # Standard approach: pick a vertex, remove it and its neighbors from color_class.
            
            # Which vertex to pick?
            # Often, picking the vertex with the highest degree in the subgraph helps?
            # Or just the first one.
            # Let's pick the lowest set bit for speed.
            
            # Find lowest set bit
            lsb = color_class & -color_class
            v = lsb.bit_length() - 1
            
            # Remove v and its neighbors from color_class
            # Neighbors in the subgraph are adj[v] & color_class
            # We want to remove v and its neighbors.
            # So we keep non-neighbors?
            # No, in a color class, no two vertices can be adjacent.
            # So if we pick v, we cannot pick any neighbor of v.
            # So we remove v and adj[v] from the pool of vertices that can be in this color class.
            
            color_class &= ~(lsb | (adj[v] & color_class))
            
        return color_idx
        
    # This coloring is O(N^2) in worst case per call, which is too slow if called many times.
    # We need a faster coloring or a different bound.
    # However, N=8000. If the depth is ~50, and we branch, the number of calls could be huge.
    # But bitset operations are fast.
    # The bottleneck is the coloring loop.
    
    # Alternative: Use a pre-computed order and simple greedy coloring.
    
    # Let's try to run the B&B and see if it fits in time.
    # If not, we might need to reduce N or use a better heuristic.
    
    # One optimization: If `used` has few bits, coloring is fast.
    # If `used` has many bits, it's slow.
    
    # We can limit the search if time is running out.
    
    def max_clique_bnb(used, current_clique, depth):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if not used:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
            
        # Upper bound: current size + color count
        # If we don't do coloring, we can use popcount(used) as a weak bound.
        # Let's try coloring.
        
        # To speed up coloring, we can stop if current_size + max_possible <= best_size
        # Max possible is popcount(used).
        if len(current_clique) + used.bit_count() <= best_size:
            return
            
        # Try coloring
        # If coloring is too slow, we might skip it or use a simpler bound.
        # Let's try a simplified coloring: just count bits? No, that's weak.
        
        # Let's implement a faster greedy coloring.
        # We'll just pick vertices and remove neighbors.
        
        # Optimization: If the number of vertices is small, exact coloring is fine.
        
        c_count = 0
        temp_used = used
        while temp_used:
            c_count += 1
            if len(current_clique) + c_count + temp_used.bit_count() <= best_size:
                 # Even if we color all remaining, we won't beat best
                 # Wait, c_count is just current color.
                 # The bound is current_clique + max_color.
                 # We don't know max_color yet.
                 # But if c_count + (temp_used.bit_count() - 1) <= best_size - len(current_clique) ... no.
                 # Let's just finish coloring.
                 pass
                 
            # Pick vertex
            lsb = temp_used & -temp_used
            v = lsb.bit_length() - 1
            temp_used &= ~(lsb | (adj[v] & temp_used))
            
        if len(current_clique) + c_count <= best_size:
            return
            
        # Branch
        # Pick a vertex to branch on.
        # Usually, we branch on the vertices in the order of the coloring or degree.
        # Let's pick the vertex with the highest degree in `used`?
        # Or just iterate through vertices in `used`.
        
        # To ensure we find the max clique, we need to try adding each vertex.
        # But we can prune.
        
        # Let's pick the vertex with the highest index?
        # Or use the order from `sorted_nodes`?
        
        # Let's iterate through vertices in `used`.
        # To make it efficient, we can process them in a specific order.
        
        # We'll pick the vertex that appears first in our sorted order (by degree).
        # We need a mapping from bit index to priority.
        
        # For simplicity, let's just pick the lowest set bit for branching.
        # This might be less efficient but is simple.
        
        # Actually, Tomita's algorithm branches on the last vertex in the color class?
        # No, it branches on vertices one by one.
        
        # Let's try branching on all vertices in `used`.
        # This is exponential.
        
        # A better approach:
        # Pick a vertex u.
        # Branch 1: Include u. New used = used & adj[u].
        # Branch 2: Exclude u. New used = used & ~ (1 << u).
        
        # We can do this recursively.
        
        # To reduce branching, we can use the coloring bound to skip some.
        
        # Let's just do a simple DFS.
        
        # Pick a vertex
        # We can pick the one with the highest degree in `used` to maximize pruning?
        
        # Let's pick the lowest set bit for now.
        lsb = used & -used
        u = lsb.bit_length() - 1
        
        # Branch 1: Include u
        new_used = used & adj[u]
        max_clique_bnb(new_used, current_clique + [u], depth + 1)
        
        # Branch 2: Exclude u
        # We must ensure we don't loop forever.
        # Remove u from used.
        used_without_u = used & ~lsb
        
        # We can also prune if the bound is low.
        max_clique_bnb(used_without_u, current_clique, depth)
        
    # This simple B&B might be too slow without good vertex ordering and pruning.
    # Given the time limit, we might not find the global optimum if the graph is hard.
    # However, the problem structure (coprime moduli are disconnected) might make it easier.
    
    # Let's run it.
    
    # Initial used: all nodes
    initial_used = (1 << N) - 1
    max_clique_bnb(initial_used, [], 0)
    
    # Convert best_clique to result format
    result = []
    for idx in best_clique:
        r, m = nodes[idx]
        result.append([r, m])
        
    if not result:
        return [[0, 2]]
        
    return result

