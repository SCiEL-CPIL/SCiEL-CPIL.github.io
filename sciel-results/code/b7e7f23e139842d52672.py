import time
import random
from itertools import combinations

# Q3 edges
Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    n = 16
    best_edges = []
    
    # Precompute Q3 adjacency for the pattern
    q3_adj = [set() for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_adj[u].add(v)
        q3_adj[v].add(u)

    def find_q3(adj_mask):
        """
        adj_mask: list of 16 ints, each a bitmask of neighbors.
        Returns a list of edges (u, v) in the original graph that form a Q3, or None.
        """
        # Iterate through all 8-subsets
        # Optimization: Skip subsets where any vertex has degree < 3 in the induced subgraph
        # We can precompute degrees in the induced subgraph? 
        # It's easier to just check during the isomorphism or beforehand.
        
        for S in combinations(range(n), 8):
            S_set = set(S)
            # Check min degree in induced subgraph
            # For each v in S, count neighbors in S
            min_deg = 8
            for v in S:
                deg = 0
                for u in S:
                    if u == v: continue
                    if adj_mask[v] & (1 << u):
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
            if min_deg < 3:
                continue
                
            # Try to map Q3 to S
            # We will map q3 vertices 0..7 to vertices in S
            # Use backtracking
            mapping = [-1] * 8
            # To reduce symmetry, we can fix mapping[0] to S[0]? 
            # No, Q3 is vertex transitive, but S is not ordered.
            # However, we can just try all 8 choices for mapping[0].
            
            # We'll use a recursive helper or a stack.
            # Since it's small, recursion is fine.
            
            def backtrack(idx):
                if idx == 8:
                    return True
                v = S[idx] # This is not quite right, we are mapping Q3 vertices to S vertices.
                # Let's map Q3 vertex i to some vertex in S.
                # We need to choose which vertex in S gets Q3 vertex i.
                # This is a permutation problem.
                
            # Better approach:
            # We have 8 Q3 vertices. We assign them to 8 S vertices.
            # Let's iterate through permutations of S? 8! = 40320.
            # For each permutation, check if it's a valid mapping.
            # 40320 * 12 edge checks = ~500k ops per 8-set.
            # 12870 8-sets * 500k = 6.4 billion ops. Too slow.
            
            # We need backtracking with pruning.
            
            # Let's map Q3 vertices in order 0, 1, ..., 7.
            # available = list(S)
            
            def dfs(q3_idx, available):
                if q3_idx == 8:
                    return True
                
                # Get neighbors of q3_idx in Q3 that are already assigned
                assigned_neighbors = []
                for nb in q3_adj[q3_idx]:
                    if nb < q3_idx: # Only look at previously assigned to check constraints
                         # Wait, we need to check ALL assigned neighbors
                         pass
                
                # Actually, standard backtracking:
                # We assign Q3 vertex i to an available vertex v in S.
                # Constraints:
                # For every neighbor nb of i in Q3:
                #   If nb is already assigned (mapping[nb] != -1), then (mapping[nb], v) must be an edge in G.
                #   If nb is not assigned, no constraint yet.
                
                # We also need to ensure that if we assign v to i, 
                # we don't violate constraints for already assigned vertices?
                # The check "for every assigned neighbor nb, (mapping[nb], v) is edge" covers this.
                
                for j, v in enumerate(available):
                    # Check constraints
                    valid = True
                    for nb in q3_adj[q3_idx]:
                        if nb < q3_idx: # nb is already assigned
                             mapped_nb = mapping[nb]
                             if not (adj_mask[mapped_nb] & (1 << v)):
                                 valid = False
                                 break
                    if valid:
                        mapping[q3_idx] = v
                        if dfs(q3_idx + 1, available[:j] + available[j+1:]):
                            return True
                        mapping[q3_idx] = -1
                return False

            # To speed up, we can sort S by degree or something?
            # Also, we can fix mapping[0] to S[0] to reduce search space by 8?
            # No, because S[0] is arbitrary. But we can iterate 0..7 for the first mapping.
            # Actually, the loop above does that implicitly if we don't fix it.
            # But we can optimize by checking if the vertex chosen for Q3 vertex 0 has enough neighbors in S for its Q3 neighbors.
            
            # Let's just run dfs.
            # To avoid global variable issues in nested function, use closure or pass state.
            
            # We need to be careful with performance.
            # If this is too slow, we might time out.
            # Let's add a quick check: 
            # Does S contain a 4-cycle? Q3 has 6 4-cycles.
            # If no 4-cycle, no Q3.
            # Checking 4-cycle is faster.
            
            # For now, let's rely on the degree filter and backtracking.
            
            # Reset mapping
            for i in range(8): mapping[i] = -1
            if dfs(0, list(S)):
                # Construct the edges
                edges = []
                for u, v in Q3_EDGES:
                    edges.append([mapping[u], mapping[v]])
                return edges
        return None

    # Main loop
    # Start with K16
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    current_edges = n*(n-1)//2
    
    while time.time() < deadline:
        q3 = find_q3(adj_mask)
        if q3 is None:
            # We found a Q3-free graph
            if current_edges > len(best_edges):
                best_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj_mask[u] & (1 << v):
                            best_edges.append([u, v])
            break
        else:
            # Remove a random edge from q3
            edge_idx = random.randint(0, len(q3)-1)
            u, v = q3[edge_idx]
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            current_edges -= 1
            
    # If we didn't find a Q3-free graph, best_edges might be empty or from a previous iteration?
    # The loop breaks when q3 is None.
    # If we run out of time, we should return the best graph we had?
    # But the graph is only Q3-free when we break.
    # If we time out, the current graph still has Q3s.
    # We must return a VALID graph.
    # So we should only update best_edges when we find a valid graph.
    # If we never find one, we should return the seed?
    # Or we can keep the "least bad" graph? No, it must be valid.
    # So if we time out, we return the best valid graph found.
    # If none found, return seed?
    
    if not best_edges:
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    return best_edges
