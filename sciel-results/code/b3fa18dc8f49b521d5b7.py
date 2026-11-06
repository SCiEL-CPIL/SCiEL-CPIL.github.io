import time
import random

def construct():
    deadline = time.time() + 145
    best_n = 32
    # Initial best: Complement of 8 disjoint K4s
    # Vertices 0..31
    # Edges: all pairs except those within the same K4 block of 4
    # Wait, the complement of 8 K4s is 8 disjoint K_{1,3}? No.
    # G = complement of (K4 x 8).
    # In G, vertices in the same K4 block are NOT connected.
    # Vertices in different K4 blocks ARE connected.
    # This graph is C5-free?
    # Let's check.
    # If we pick 5 vertices from 5 different blocks, they form a K5, which contains C5.
    # So the complement of 8 K4s is NOT C5-free.
    # The task says: "best known has 32 vertices".
    # The graph with 32 vertices that works is likely the **disjoint union of 8 K4s**?
    # If G is 8 disjoint K4s:
    # 1. C5-free? Yes, components are K4 (no cycles > 3).
    # 2. Independent set? An independent set can have at most 1 vertex from each K4.
    #    So alpha = 8.
    #    We need alpha < 9. 8 < 9 is True.
    # So, 8 disjoint K4s is a valid graph with n=32.
    
    best_edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
                
    def is_c5_free(n, adj_masks):
        # adj_masks: list of ints
        for u in range(n):
            for v in range(u+1, n):
                if (adj_masks[u] >> v) & 1:
                    # Check for path u-a-b-c-v
                    # a in N(u)\{v}, c in N(v)\{u}
                    mask_u = adj_masks[u] & ~(1 << v)
                    mask_v = adj_masks[v] & ~(1 << u)
                    
                    # Iterate a
                    ma = mask_u
                    while ma:
                        a = (ma & -ma).bit_length() - 1
                        ma &= ma - 1
                        if a == v: continue # Should be removed by mask
                        # Iterate c
                        mc = mask_v
                        while mc:
                            c = (mc & -mc).bit_length() - 1
                            mc &= mc - 1
                            if a == c: continue
                            # Check if N(a) and N(c) have common neighbor b
                            # b != u, v, a, c
                            common = adj_masks[a] & adj_masks[c]
                            common &= ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << c)
                            if common:
                                return False
        return True

    def has_independent_set_9(n, adj_masks):
        # Greedy
        # Try to find independent set of size 9
        # Use randomized greedy
        for _ in range(20):
            rem = list(range(n))
            random.shuffle(rem)
            iset = []
            while rem:
                # Pick vertex with min degree in remaining
                best_v = -1
                min_deg = 100
                for v in rem:
                    # Count neighbors in rem
                    deg = 0
                    mask_v = adj_masks[v]
                    for u in rem:
                        if (mask_v >> u) & 1:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                iset.append(best_v)
                rem.remove(best_v)
                # Remove neighbors of best_v from rem
                mask_bv = adj_masks[best_v]
                new_rem = []
                for u in rem:
                    if not (mask_bv >> u) & 1:
                        new_rem.append(u)
                rem = new_rem
            if len(iset) >= 9:
                return True
        return False

    def validate(n, adj_masks):
        if not is_c5_free(n, adj_masks):
            return False
        if has_independent_set_9(n, adj_masks):
            return False
        return True

    def adj_to_edges(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj_masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Local Search
    def local_search(n, initial_adj, time_limit):
        adj = initial_adj[:]
        start_time = time.time()
        while time.time() < start_time + time_limit:
            if validate(n, adj):
                return adj
            # Pick random edge to toggle
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            # Toggle
            mask = 1 << v
            adj[u] ^= mask
            mask = 1 << u
            adj[v] ^= mask
            # Check if valid
            if validate(n, adj):
                return adj
            # Accept with some probability? Or just keep if better?
            # Let's use a simple hill climbing with restarts if stuck.
            # But we don't have a score.
            # Let's try to minimize "badness".
            # Badness = (1 if C5 else 0) * 100 + (1 if alpha>=9 else 0) * 10
            # This is too expensive to compute every step.
            pass
        return None

    # Strategy:
    # 1. Try n=35, 34, 33.
    # Start with 8 K4s + random vertices.
    # For n=33: 8 K4s + 1 vertex.
    # The new vertex is isolated initially. Alpha becomes 9 (8 from K4s + 1 new).
    # We need to connect the new vertex to at least 1 vertex in each K4?
    # If we connect new vertex v to one vertex in each K4, does it create C5?
    # v is connected to 8 vertices.
    # If v-a-b-c-v is a cycle?
    # a is in K4_1. b, c must be neighbors of a.
    # In K4, a is connected to 3 others.
    # If v is connected to a in K4_1 and b in K4_2.
    # Path v-a-...-b-v?
    # a and b are in different components (originally).
    # So no path between a and b unless we add edges.
    # If we only add edges from v to the K4s, the graph remains C5-free?
    # The K4s are K4. Adding a vertex v connected to some vertices in K4s.
    # Can we form a C5?
    # v-a-b-c-d-v?
    # a,b,c,d must be in the K4s.
    # If a,b in K4_1. c in K4_2.
    # v-a (yes), a-b (yes, in K4), b-c (no, different K4s), c-d (yes, in K4_2), d-v (yes).
    # So if we connect v to a in K4_1 and d in K4_2, and there is an edge b-c? No.
    # So just connecting v to one vertex in each K4 might be safe?
    # Let's check alpha.
    # If v is connected to one vertex in each K4, can we pick an independent set of size 9?
    # We pick v. We cannot pick the neighbor of v in each K4.
    # From each K4, we can pick at most 1 vertex (since it's a clique).
    # We have 8 K4s. We can pick 1 from each, avoiding the neighbor of v.
    # So we can pick 8 vertices from the K4s + v = 9 vertices.
    # So alpha >= 9. We need to reduce alpha.
    # We need to add more edges to prevent picking 1 from each K4.
    # This suggests we need to mix the K4s or add edges between them.
    
    # Let's run a generic local search for n=33, 34, 35.
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 10: break
        # Initialize: 8 K4s + (target_n - 32) random vertices
        # Connect random vertices randomly to existing ones
        adj = [0] * target_n
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i+1, 4):
                    adj[base+i] |= (1 << (base+j))
                    adj[base+j] |= (1 << (base+i))
        
        # Add random edges for new vertices
        for v in range(32, target_n):
            # Connect to some random existing vertices
            for u in range(v):
                if random.random() < 0.3:
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
        
        # Local search
        # We will try to toggle edges.
        # To speed up, we only check validity if we change an edge that "looks" important?
        # No, just random toggle and validate.
        # Validation is the bottleneck.
        # is_c5_free is O(n^5) worst case but fast in practice with bitmasks?
        # Let's optimize is_c5_free.
        
        # We'll do 1000 steps of local search per restart.
        found = False
        for restart in range(10):
            if time.time() > deadline - 5: break
            # Random restart or continue?
            # Let's try to repair the current adj.
            for step in range(200):
                if time.time() > deadline - 5: break
                if validate(target_n, adj):
                    best_n = target_n
                    best_edges = adj_to_edges(target_n, adj)
                    found = True
                    break
                # Toggle random edge
                u = random.randint(0, target_n-1)
                v = random.randint(0, target_n-1)
                if u == v: continue
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            if found: break
        
        if found: break

    if best_n == 0:
        best_n = 32
        best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]

    return {'n': best_n, 'edges': best_edges}
