import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    masks = [1 << i for i in range(n)]
    all_vertices = (1 << n) - 1
    
    # Adjacency list as bitsets
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count()
    
    def has_c6_with_edge(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        This is true if there is a path of length 5 between u and v in the current graph.
        Path: u - a - b - c - d - v
        Vertices a,b,c,d must be distinct and distinct from u,v.
        """
        # We can iterate over neighbors of u and v
        # To minimize work, iterate over the smaller neighborhood
        nu = adj[u] & ~masks[v]
        nv = adj[v] & ~masks[u]
        
        # If either is empty, no C6
        if not nu or not nv:
            return False
            
        # We need a path a-b-c-d where a in N(u), d in N(v)
        # Let's iterate a in N(u)
        # Optimization: Pre-compute "reachable in 2 steps" from a?
        # Path u-a-b-c-d-v.
        # b is neighbor of a (not u)
        # c is neighbor of b (not a, u)
        # d is neighbor of c (not b, a, u) AND neighbor of v (not c, b, a, u)
        
        # To speed up, we can iterate over a and d?
        # For a fixed a, we want to know if any d in N(v) is reachable from a in 3 steps (a-b-c-d)
        # excluding u.
        
        # Let's try:
        # For each a in N(u):
        #   Let B = N(a) & ~mask(u)
        #   For each b in B:
        #     Let C = N(b) & ~mask(u) & ~mask(a)
        #     For each c in C:
        #       Let D = N(c) & ~mask(u) & ~mask(a) & ~mask(b)
        #       If D & N(v) is not empty, return True.
        
        # This is worst case O(k^4) but k is small.
        
        # Heuristic: Iterate over a with smallest degree? Or just all a?
        # Let's swap u and v if N(v) is smaller?
        # Actually, the structure is symmetric.
        
        a_mask = nu
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b must be in N(a), not u
            b_mask = adj[a] & ~masks[u]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c must be in N(b), not u, a
                c_mask = adj[b] & ~masks[u] & ~masks[a]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c), not u, a, b
                    # AND d must be in N(v), not c, b, a, u
                    d_mask = adj[c] & ~masks[u] & ~masks[a] & ~masks[b]
                    # Check intersection with N(v)
                    # N(v) is adj[v]
                    # We need d in adj[v] and d in d_mask
                    # Also d != v (implied by adj[v] not having self loops)
                    # d != u, a, b (implied by d_mask)
                    # d != c (implied by simple graph, no self loops in N(c)? 
                    # Wait, adj[c] does not contain c. So d != c is guaranteed if we just check intersection?
                    # Yes, adj[c] & adj[v] will not contain c if c is not adjacent to v? 
                    # Actually, if c is in N(v), then c is a candidate for d?
                    # If d=c, then path is a-b-c-c? No, d is a vertex. 
                    # The path is u-a-b-c-d-v. 
                    # If d=c, then c is adjacent to v. 
                    # The path is u-a-b-c-v. Length 4. 
                    # Cycle: u-a-b-c-v-u. Length 5. C5. 
                    # We are looking for C6. 
                    # So we must exclude c from d?
                    # The path vertices are u, a, b, c, d, v. They must be distinct.
                    # So d != c.
                    # Since d is chosen from N(c), d != c is always true in a simple graph?
                    # No, d is a neighbor of c. So d != c.
                    # So we just need to check if (adj[c] & adj[v] & ~(masks[u] | masks[a] | masks[b])) is non-empty.
                    
                    if adj[c] & adj[v] & ~(masks[u] | masks[a] | masks[b]):
                        return True
                        
        return False

    def add_edge(u, v):
        adj[u] |= masks[v]
        adj[v] |= masks[u]
        
    def remove_edge(u, v):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                bit = m & -m
                j = bit.bit_length() - 1
                if i < j:
                    edges.append((i, j))
                m ^= bit
        return edges

    def get_non_edges():
        non_edges = []
        for i in range(n):
            # Non-neighbors of i greater than i
            m = all_vertices & ~adj[i] & ~masks[i]
            # Only consider j > i to avoid duplicates? 
            # For local search, we might want all? 
            # Usually we work on undirected edges.
            for j in range(i+1, n):
                if not (adj[i] & masks[j]):
                    non_edges.append((i, j))
        return non_edges

    def greedy_fill():
        """Try to add all possible edges in a smart order."""
        # Get all non-edges
        non_edges = get_non_edges()
        # Sort by degree sum? Lower degree sum is "easier" to add?
        # Or higher? 
        # Adding an edge between high-degree vertices is more likely to create C6.
        # So adding low-degree edges first is safer and might open up more space.
        non_edges.sort(key=lambda e: adj[e[0]].bit_count() + adj[e[1]].bit_count())
        
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if not (adj[u] & masks[v]): # Check if still non-edge
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    def local_search():
        """Iteratively try to add edges or swap edges to increase count."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Try adding a non-edge
            non_edges = get_non_edges()
            if not non_edges:
                break
                
            # Shuffle to handle ties randomly? Or sort by heuristic?
            # Sorting by degree sum (ascending) was good for greedy.
            # For local search, maybe try random ones first?
            random.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if not (adj[u] & masks[v]):
                    if not has_c6_with_edge(u, v):
                        add_edge(u, v)
                        improved = True
                        added = True
                        break # Restart loop with new state
            
            if not added:
                # 2. Try 1-remove / 1-add swap (Neutral move to escape local opt)
                # Pick a random edge to remove
                edges = get_edges_list()
                if not edges:
                    break
                random.shuffle(edges)
                
                for eu, ev in edges[:20]: # Try few swaps
                    if time.time() > deadline:
                        break
                    remove_edge(eu, ev)
                    
                    # Now try to add a non-edge
                    # This is a "move" that keeps edge count same, but might allow future adds
                    # Actually, we want to increase edge count. 
                    # If we are stuck, we are at a local max for 1-add.
                    # We need to find a sequence that increases count.
                    # A 1-remove/1-add doesn't increase count.
                    # We need to find a "2-opt" or similar.
                    
                    # Let's try: Remove (eu, ev). Then try to add as many non-edges as possible greedily.
                    # If we add more than 1, net gain.
                    # If we add 1, net 0 (but structure changed).
                    # If we add 0, net -1 (bad, revert).
                    
                    initial_count = count_edges()
                    
                    greedy_fill()
                    
                    final_count = count_edges()
                    
                    if final_count > initial_count:
                        # We made progress! Keep the change
                        improved = True
                        break
                    else:
                        # Revert the removal of (eu, ev) and any added edges?
                        # This is complex. 
                        # Simpler: Just revert the removal of (eu, ev) and hope the greedy fill 
                        # didn't mess things up? No, greedy fill added edges.
                        # We need to store state.
                        # Let's implement a simpler "Perturbation" strategy instead of complex revert.
                        pass
                        
        # If we didn't improve in the above loop, we are stuck.
        # The above loop didn't handle reversion well. Let's rewrite the main search loop.

    def run_search():
        nonlocal adj
        
        best_adj = [0] * n
        best_edges = 0
        
        # Initializations
        # 1. Empty graph
        adj = [0] * n
        greedy_fill()
        if count_edges() > best_edges:
            best_edges = count_edges()
            best_adj = adj[:]
            
        # 2. 8 K5s
        adj = [0] * n
        for k in range(8):
            start = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    add_edge(start + i, start + j)
        greedy_fill()
        if count_edges() > best_edges:
            best_edges = count_edges()
            best_adj = adj[:]

        # 3. Random sparse start
        adj = [0] * n
        for _ in range(50):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not (adj[u] & masks[v]):
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)
        greedy_fill()
        if count_edges() > best_edges:
            best_edges = count_edges()
            best_adj = adj[:]

        # ILS Loop
        # Current state is best_adj
        adj = best_adj[:]
        
        while time.time() < deadline - 5:
            # Perturb: Remove 2-5 random edges
            edges = get_edges_list()
            if not edges:
                break
            random.shuffle(edges)
            k = random.randint(2, 5)
            for eu, ev in edges[:k]:
                remove_edge(eu, ev)
                
            # Optimize: Greedy Fill
            greedy_fill()
            
            # Check if better
            current_edges = count_edges()
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
            else:
                # Revert to best state? 
                # Or keep current state for next perturbation?
                # Keeping current state allows exploring "downhill" moves.
                # But if we go too far down, we might waste time.
                # Let's keep current state.
                pass

        return best_adj

    best_adj = run_search()
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & masks[j]:
                edges.append([i, j])
                
    return edges

if __name__ == "__main__":
    # Test
    e = construct()
    print(f"Edges: {len(e)}")
