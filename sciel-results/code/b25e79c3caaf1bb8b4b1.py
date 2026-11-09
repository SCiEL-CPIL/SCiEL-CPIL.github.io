"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n = 16
    
    # Q3 has 8 vertices and 12 edges
    # Vertices are 0..7 (binary 000..111)
    # Edges connect vertices differing in exactly 1 bit
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Precompute for each edge in Q3, the set of vertices at each "distance pattern"
    # For edge (0,1): 
    #   0's other neighbors: 2, 4
    #   1's other neighbors: 3, 5
    #   2's other neighbor (besides 0): 6
    #   4's other neighbor (besides 0): 6
    #   3's other neighbor (besides 1): 7
    #   5's other neighbor (besides 1): 7
    #   6's neighbors: 2, 4, 7
    #   7's neighbors: 3, 5, 6
    
    # Build Q3 adjacency
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def count_q3_subgraphs(adj_sets):
        """Count Q3 subgraphs in graph with adjacency sets."""
        count = 0
        # Try all 8-vertex subsets
        for subset in itertools.combinations(range(n), 8):
            subset_list = list(subset)
            # Backtracking: assign Q3 vertices to subset vertices
            assignment = [-1] * 8
            used = [False] * n
            
            def backtrack(q3_idx):
                nonlocal count
                if q3_idx == 8:
                    count += 1
                    return
                for sv in subset_list:
                    if used[sv]:
                        continue
                    # Check compatibility with already assigned neighbors
                    compatible = True
                    for prev_q3 in range(q3_idx):
                        if prev_q3 in q3_adj[q3_idx]:
                            prev_sv = assignment[prev_q3]
                            if prev_sv not in adj_sets[sv]:
                                compatible = False
                                break
                    if compatible:
                        used[sv] = True
                        assignment[q3_idx] = sv
                        backtrack(q3_idx + 1)
                        used[sv] = False
                        assignment[q3_idx] = -1
            
            backtrack(0)
        return count
    
    def is_q3_free(adj_sets):
        """Check if graph is Q3-free."""
        # For each 8-vertex subset, check if it contains a Q3
        for subset in itertools.combinations(range(n), 8):
            subset_list = list(subset)
            subset_set = set(subset)
            
            # Try to embed Q3 into this subset using backtracking
            # Q3 vertices: 0-7, edges as above
            assignment = [-1] * 8
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    return True
                for sv in subset_list:
                    # Check if sv is compatible with all already-assigned Q3 neighbors
                    compatible = True
                    for prev_q3 in range(q3_idx):
                        if prev_q3 in q3_adj[q3_idx]:
                            prev_sv = assignment[prev_q3]
                            if prev_sv not in adj_sets[sv]:
                                compatible = False
                                break
                        else:
                            # Non-neighbor in Q3 should NOT be adjacent in graph? 
                            # No! Subgraph isomorphism: we only need the Q3 edges to map to edges.
                            # Non-edges in Q3 can map to anything in G.
                            pass
                    if compatible:
                        assignment[q3_idx] = sv
                        if backtrack(q3_idx + 1):
                            return True
                        assignment[q3_idx] = -1
                return False
            
            if backtrack(0):
                return False
        
        return True
    
    def has_q3(adj_sets, new_edge):
        """Fast check: does adding new_edge create a Q3?
        We check all Q3 embeddings where new_edge is one of the Q3 edges."""
        u, v = new_edge
        
        # For each Q3 edge (a,b), check if we can embed Q3 with a->u, b->v or a->v, b->u
        for (a, b) in q3_edges:
            # Case 1: a->u, b->v
            if _check_embedding(adj_sets, u, v, a, b):
                return True
            # Case 2: a->v, b->u
            if _check_embedding(adj_sets, v, u, a, b):
                return True
        
        return False
    
    def _check_embedding(adj_sets, u, v, a, b):
        """Check if Q3 can be embedded with vertex a mapped to u and vertex b mapped to v.
        a and b are adjacent in Q3."""
        # We need to find a bijection from Q3 vertices {0..7} to some 8 vertices in G
        # such that a->u, b->v, and all Q3 edges map to G edges.
        
        # Build partial assignment
        assignment = [None] * 8
        assignment[a] = u
        assignment[b] = v
        
        # Determine which Q3 vertices still need to be assigned
        remaining = [i for i in range(8) if i not in (a, b)]
        
        # For each remaining Q3 vertex, find candidate G vertices
        # A G vertex g is a candidate for Q3 vertex q if:
        #   For every already-assigned Q3 vertex p:
        #     if p is adjacent to q in Q3, then g must be adjacent to assignment[p] in G
        #     if p is NOT adjacent to q in Q3, no constraint (subgraph, not induced)
        
        # Use backtracking
        def backtrack(rem_idx):
            if rem_idx == len(remaining):
                return True
            q = remaining[rem_idx]
            # Find all G vertices that are compatible with q
            for g in range(n):
                # g must not already be used
                if g in (u, v) and g not in (assignment[a], assignment[b]):
                    # Actually u and v are already used for a and b
                    # Let's check if g is already assigned to some Q3 vertex
                    if assignment.count(g) > 0:
                        continue
                else:
                    if assignment.count(g) > 0:
                        continue
                
                # Check compatibility
                compatible = True
                for p in range(8):
                    if assignment[p] is not None:
                        if p in q3_adj[q]:
                            if assignment[p] not in adj_sets[g]:
                                compatible = False
                                break
                if compatible:
                    assignment[q] = g
                    if backtrack(rem_idx + 1):
                        return True
                    assignment[q] = None
            return False
        
        return backtrack(0)
    
    # Strategy: Start with a random graph, then use simulated annealing / local search
    # to maximize edges while maintaining Q3-free property.
    
    # Better: Start with K_{16} and remove edges to make it Q3-free, minimizing removals.
    # Or: Start with a sparse graph and add edges greedily.
    
    # Let's use a smarter approach:
    # 1. Start with K_{3,13} (39 edges) as seed
    # 2. Use local search to add edges one at a time, checking Q3-freeness
    
    # Actually, let's try: start with complete graph, then iteratively remove edges
    # that are "most involved" in Q3 subgraphs until Q3-free.
    
    # For speed, let's use the fast has_q3 check.
    
    # Phase 1: Greedy construction
    adj_sets = [set() for _ in range(n)]
    edges = []
    
    # Start with K_{3,13}
    part_a = [0, 1, 2]
    part_b = list(range(3, n))
    
    for a in part_a:
        for b in part_b:
            adj_sets[a].add(b)
            adj_sets[b].add(a)
            edges.append([a, b])
    
    # Now try to add all remaining edges in a good order
    all_remaining = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj_sets[i]:
                all_remaining.append((i, j))
    
    # Sort by a heuristic: edges that share more structure with existing graph first?
    # Or: edges within part_a first (3 edges), then within part_b
    
    def edge_score(edge):
        i, j = edge
        # Prefer edges within the smaller part
        if i in part_a and j in part_a:
            return 0
        elif i in part_b and j in part_b:
            return 1
        else:
            return 2
    
    all_remaining.sort(key=edge_score)
    
    best_edges = list(edges)
    best_adj = [set(s) for s in adj_sets]
    best_count = len(edges)
    
    for edge in all_remaining:
        if time.time() > deadline - 30:
            break
        
        u, v = edge
        # Try adding
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edges.append([u, v])
        
        if not has_q3(adj_sets, (u, v)):
            best_edges = list(edges)
            best_adj = [set(s) for s in adj_sets]
            best_count = len(edges)
        else:
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            edges.pop()
    
    # Phase 2: Local search / simulated annealing to improve
    # Try swapping: remove one edge, add two others
    current_adj = [set(s) for s in best_adj]
    current_edges = list(best_edges)
    current_count = len(current_edges)
    
    # Build list of non-edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated annealing-like local search
    temperature = 1.0
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
        
        # Try a random move: 
        # 1. Add a random non-edge
        # 2. If it creates Q3, remove it and try to add a different edge
        # 3. Or: remove a random edge, then try to add two others
        
        move_type = random.choice(['add', 'swap'])
        
        if move_type == 'add':
            # Find a random non-edge
            non_edges = []
            for (i, j) in all_possible:
                if j not in current_adj[i]:
                    non_edges.append((i, j))
            if not non_edges:
                continue
            
            u, v = random.choice(non_edges)
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            if not has_q3(current_adj, (u, v)):
                current_edges.append([u, v])
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(current_edges)
                    best_adj = [set(s) for s in current_adj]
            else:
                current_adj[u].remove(v)
                current_adj[v].remove(u)
        
        elif move_type == 'swap':
            # Remove a random edge, then try to add two others
            if not current_edges:
                continue
            
            idx = random.randint(0, len(current_edges) - 1)
            e1 = current_edges[idx]
            u1, v1 = e1[0], e1[1]
            
            # Remove e1
            current_adj[u1].remove(v1)
            current_adj[v1].remove(u1)
            current_edges.pop(idx)
            current_count -= 1
            
            # Now try to add two edges from the remaining non-edges
            non_edges = []
            for (i, j) in all_possible:
                if j not in current_adj[i] and (i, j) != (u1, v1) and (i, j) != (v1, u1):
                    non_edges.append((i, j))
            
            # Try to add up to 2 edges
            added = 0
            random.shuffle(non_edges)
            for (u, v) in non_edges[:20]:
                if added >= 2:
                    break
                if time.time() > deadline - 10:
                    break
                
                current_adj[u].add(v)
                current_adj[v].add(u)
                
                if not has_q3(current_adj, (u, v)):
                    current_edges.append([u, v])
                    current_count += 1
                    added += 1
                else:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
            
            if current_count > best_count:
                best_count = current_count
                best_edges = list(current_edges)
                best_adj = [set(s) for s in current_adj]
    
    # Phase 3: Final greedy pass
    adj_sets = [set(s) for s in best_adj]
    edges = list(best_edges)
    
    for i in range(n):
        for j in range(i+1, n):
            if time.time() > deadline - 5:
                break
            if j in adj_sets[i]:
                continue
            
            u, v = i, j
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
            if not has_q3(adj_sets, (u, v)):
                edges.append([u, v])
            else:
                adj_sets[u].remove(v)
                adj_sets[v].remove(u)
    
    return edges
# EVOLVE-BLOCK-END
