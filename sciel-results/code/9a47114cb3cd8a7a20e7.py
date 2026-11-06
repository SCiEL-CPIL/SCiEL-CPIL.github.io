import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    
    # Build Q3 adjacency patterns
    # Q3 is the 3-cube: vertices 0-7, edges between vertices differing in exactly 1 bit
    q3_vertices = list(range(8))
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Generate all 8-vertex subsets and check if they can host a Q3
    # Precompute: for each 8-subset S, does there exist a bijection phi: Q3 -> S
    # such that (u,v) in Q3 edges => (phi(u), phi(v)) is an edge in G?
    # That's the full Q3 check. Instead, we use a faster approach.
    
    # Key insight: A Q3 has a very specific structure. We can enumerate all 
    # possible Q3 embeddings into 16 vertices. But 16 choose 8 = 12870, and 
    # 8! = 40320 bijections per subset... that's too many.
    
    # Better approach: Use the fact that Q3 is a 3-regular bipartite graph with 
    # girth 4. We can use a smarter enumeration.
    
    # Actually, let's use a different strategy: 
    # 1. Start with the complete graph K16 (120 edges)
    # 2. Remove edges to destroy all Q3s
    # 3. Use a greedy/local search approach
    
    # First, let's implement a fast Q3 detector using bitmask adjacency
    
    def build_adj_bitmask(edge_set):
        """Build adjacency as bitmask array."""
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_q3_fast(adj):
        """
        Check if graph with adjacency bitmask has a Q3 subgraph.
        A Q3 can be identified by its bipartition: 4 vertices in one part, 4 in other.
        Each vertex in one part connects to exactly 3 in the other part.
        
        We enumerate all 4-4 bipartitions and check.
        """
        # Enumerate all 4-subsets A of {0..15}
        from itertools import combinations
        vertices = list(range(n))
        
        for A in combinations(vertices, 4):
            A_set = set(A)
            # The complement in a candidate Q3 would be a 4-subset B from remaining 12
            remaining = [v for v in vertices if v not in A_set]
            
            for B in combinations(remaining, 4):
                # Check if A and B form a Q3
                # Each vertex in A must connect to exactly 3 vertices in B
                # Each vertex in B must connect to exactly 3 vertices in A
                # And the bipartite graph must be a 3-regular bipartite graph on 4+4 vertices
                # There are exactly 2 non-isomorphic 3-regular bipartite graphs on 4+4:
                # The 3-cube Q3 and K4,4 minus a perfect matching.
                # Actually, Q3 is the only 3-regular bipartite graph on 8 vertices that is a cube.
                
                # Check: for each a in A, it must have exactly 3 neighbors in B
                valid = True
                for a in A:
                    deg_in_B = sum(1 for b in B if adj[a] & (1 << b))
                    if deg_in_B != 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for b in B:
                    deg_in_A = sum(1 for a in A if adj[b] & (1 << a))
                    if deg_in_A != 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Now we have a 3-regular bipartite graph on A ∪ B.
                # We need to check if it's isomorphic to Q3.
                # Q3 has girth 4 and is the unique 3-regular bipartite graph on 8 vertices
                # with exactly 6 4-cycles. Let's count 4-cycles.
                
                # Count 4-cycles in the bipartite graph A-B-A-B
                # A 4-cycle is a1-b1-a2-b2-a1 where a1,a2 in A, b1,b2 in B
                # and all edges exist.
                
                count_4_cycles = 0
                A_list = list(A)
                B_list = list(B)
                for i in range(4):
                    for j in range(i + 1, 4):
                        a1, a2 = A_list[i], A_list[j]
                        # Common neighbors in B
                        common = []
                        for b in B_list:
                            if (adj[a1] & (1 << b)) and (adj[a2] & (1 << b)):
                                common.append(b)
                        # Each pair of common neighbors gives a 4-cycle
                        count_4_cycles += len(common) * (len(common) - 1) // 2
                
                # Q3 has exactly 6 4-cycles
                if count_4_cycles == 6:
                    return True, (A, B)
        
        return False, None
    
    def find_q3_edges(adj, A, B):
        """Return the edges of the Q3 formed by bipartition (A, B)."""
        edges = []
        for a in A:
            for b in B:
                if adj[a] & (1 << b):
                    edges.append((a, b) if a < b else (b, a))
        return edges
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start from complete graph, remove edges to destroy Q3s
    # Strategy 2: Start from random sparse graph, add edges
    
    # Let's use multiple approaches
    
    # Approach A: Greedy edge removal from K16
    def approach_greedy_remove():
        """Start with K16, greedily remove edges to destroy Q3s."""
        current_edges = set(all_edges)
        adj = build_adj_bitmask(current_edges)
        
        steps = 0
        max_steps = 500
        while steps < max_steps:
            if time.time() > deadline - 10:
                break
            found, (A, B) = has_q3_fast(adj)
            if not found:
                break
            # Remove a random edge from the Q3
            q3_e = find_q3_edges(adj, A, B)
            e_to_remove = random.choice(q3_e)
            if e_to_remove[0] > e_to_remove[1]:
                e_to_remove = (e_to_remove[1], e_to_remove[0])
            if e_to_remove in current_edges:
                current_edges.remove(e_to_remove)
                u, v = e_to_remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            steps += 1
        
        return list(current_edges)
    
    # Approach B: Random initialization + hill climbing
    def approach_hill_climb(p_init, n_restarts=20):
        """Start with random graph, remove Q3s, then hill climb by adding edges."""
        best_local = []
        best_local_count = 0
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Generate random graph
            current_edges = set()
            for e in all_edges:
                if random.random() < p_init:
                    current_edges.add(e)
            
            adj = build_adj_bitmask(current_edges)
            
            # Remove Q3s
            steps = 0
            while steps < 300:
                if time.time() > deadline - 10:
                    break
                found, (A, B) = has_q3_fast(adj)
                if not found:
                    break
                q3_e = find_q3_edges(adj, A, B)
                e_to_remove = random.choice(q3_e)
                if e_to_remove[0] > e_to_remove[1]:
                    e_to_remove = (e_to_remove[1], e_to_remove[0])
                if e_to_remove in current_edges:
                    current_edges.remove(e_to_remove)
                    u, v = e_to_remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                steps += 1
            
            if len(current_edges) > best_local_count:
                best_local_count = len(current_edges)
                best_local = list(current_edges)
            
            # Hill climbing: try to add missing edges
            missing = [e for e in all_edges if e not in current_edges]
            random.shuffle(missing)
            
            improved = True
            while improved:
                if time.time() > deadline - 10:
                    break
                improved = False
                random.shuffle(missing)
                for e in missing:
                    if time.time() > deadline - 10:
                        break
                    u, v = e
                    # Try adding e
                    old_adj_u = adj[u]
                    old_adj_v = adj[v]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    found, _ = has_q3_fast(adj)
                    if found:
                        # Rollback
                        adj[u] = old_adj_u
                        adj[v] = old_adj_v
                    else:
                        # Keep the edge
                        current_edges.add(e)
                        missing.remove(e)
                        improved = True
                        break  # Restart the scan
        
        return best_local
    
    # Run approach A first
    if time.time() < deadline - 30:
        edges_a = approach_greedy_remove()
        if len(edges_a) > best_count:
            best_count = len(edges_a)
            best_edges = edges_a
    
    # Run approach B with different p_init values
    for p in [0.3, 0.4, 0.5]:
        if time.time() > deadline - 20:
            break
        edges_b = approach_hill_climb(p, n_restarts=10)
        if len(edges_b) > best_count:
            best_count = len(edges_b)
            best_edges = edges_b
    
    # Final verification and return
    # Ensure we have a valid Q3-free graph
    if best_edges:
        adj = build_adj_bitmask(set(best_edges))
        found, _ = has_q3_fast(adj)
        if found:
            # This shouldn't happen, but just in case, fall back to K3,13
            best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    if not best_edges:
        # Fallback: K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
