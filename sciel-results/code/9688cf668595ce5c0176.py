"""Construct a K4-free graph with alpha <= 5, maximizing n within time limit."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Start with Paley graph on 17 vertices (known valid)
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i].add(j)
                adj[j].add(i)

    best_n = n
    best_adj = [set(s) for s in adj]

    def has_k4(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    continue
                for k in range(j + 1, n):
                    if k not in adj[i] or k not in adj[j]:
                        continue
                    for l in range(k + 1, n):
                        if l not in adj[i] or l not in adj[j] or l not in adj[k]:
                            continue
                        return True
        return False

    def has_independent_6(adj, n):
        def backtrack(remaining, current):
            if len(current) == 6:
                return True
            if len(current) + len(remaining) < 6:
                return False
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            new_remaining = remaining - {v} - adj[v]
            if backtrack(new_remaining, current + [v]):
                return True
            if backtrack(remaining - {v}, current):
                return True
            return False
        return backtrack(set(range(n)), [])

    def is_valid(adj, n):
        if has_k4(adj, n):
            return False
        if has_independent_6(adj, n):
            return False
        return True

    if not is_valid(adj, n):
        n = 8
        adj = [set() for _ in range(n)]
        for i in range(4):
            for j in range(4, 8):
                adj[i].add(j)
                adj[j].add(i)
        best_n = n
        best_adj = [set(s) for s in adj]

    # Constraint propagation approach: build graph vertex by vertex
    # For each new vertex, determine valid neighbor sets using bitmask operations
    
    def get_valid_neighbor_masks(adj, n):
        """
        For a graph on n vertices, return for each vertex v the set of valid
        neighbor masks (as frozensets) that a new vertex (v) could have.
        
        A neighbor set S for new vertex v is valid iff:
        1. No K4: For any two vertices a,b in S that are adjacent, 
           there must be no common neighbor c of both a and b in S∪{v} 
           that is also adjacent to v. More precisely: if a~b and a~v and b~v,
           then for any c~a, c~b, c~v, we'd have K4 {v,a,b,c}.
           So: for all a,b in S with a~b, N(a) ∩ N(b) ∩ S = ∅
        2. No independent 6: Adding v with non-neighbors T = V\S must not
           create an independent set of size 6.
           
        We'll check these constraints directly.
        """
        # This function is expensive; we'll use it selectively
        pass

    def check_new_vertex_valid(adj, n, new_v_neighbors):
        """
        Check if adding a new vertex (index n) with the given neighbor set
        keeps the graph K4-free and alpha <= 5.
        """
        # Check K4 constraint:
        # For any pair (a,b) in new_v_neighbors that are adjacent,
        # no vertex c in new_v_neighbors can be adjacent to both a and b
        # (otherwise {new_v, a, b, c} is a K4)
        for a, b in combinations(new_v_neighbors, 2):
            if b in adj[a]:  # a and b are adjacent
                # Check if any c in new_v_neighbors is adjacent to both a and b
                for c in new_v_neighbors:
                    if c == a or c == b:
                        continue
                    if c in adj[a] and c in adj[b]:
                        return False  # K4 found: {new_v, a, b, c}
        
        # Check independent set constraint:
        # Non-neighbors of new vertex: T = {0,...,n-1} - new_v_neighbors
        # We need to ensure no independent set of size 6 exists in the new graph.
        # An independent set in the new graph either:
        # (a) doesn't include new_v: already checked (current graph is valid)
        # (b) includes new_v: then the other 5 vertices must be in T and
        #     form an independent set in the original graph
        # So we need: alpha of T in the original graph <= 5
        # More precisely: no independent set of size 5 in T (since adding new_v gives size 6)
        
        T = set(range(n)) - new_v_neighbors
        if len(T) < 5:
            return True  # Can't form independent set of size 6 including new_v
        
        # Check if T contains an independent set of size 5
        # Use backtracking
        def has_independent_k_in_subset(vertices, k, adj):
            if k == 0:
                return True
            if len(vertices) < k:
                return False
            if k == 1:
                return len(vertices) >= 1
            
            # Pick vertex with highest degree within vertices for pruning
            v = max(vertices, key=lambda x: len(adj[x] & vertices))
            
            # Try including v: then we need k-1 more from vertices - {v} - N(v)
            remaining_after_include = vertices - {v} - adj[v]
            if has_independent_k_in_subset(remaining_after_include, k - 1, adj):
                return True
            
            # Try excluding v: need k more from vertices - {v}
            if has_independent_k_in_subset(vertices - {v}, k, adj):
                return True
            
            return False
        
        if has_independent_k_in_subset(T, 5, adj):
            return False
        
        return True

    # Main search loop: try to add vertices using constraint-aware randomization
    while time.time() < deadline:
        new_n = n + 1
        if new_n > 40:
            break

        success = False
        for attempt in range(500):
            if time.time() > deadline:
                break

            # Generate candidate neighbor set for new vertex
            # Bias: prefer moderate density
            density = random.uniform(0.3, 0.7)
            new_v_neighbors = set()
            for i in range(n):
                if random.random() < density:
                    new_v_neighbors.add(i)

            if check_new_vertex_valid(adj, n, new_v_neighbors):
                # Add the new vertex
                new_adj = [set(s) for s in adj]
                new_adj.append(set())
                for i in new_v_neighbors:
                    new_adj[i].add(n)
                    new_adj[n].add(i)
                
                adj = new_adj
                n = new_n
                success = True
                best_n = n
                best_adj = [set(s) for s in adj]
                break

        if not success:
            # Try local search: modify edges to enable growth
            improved = False
            for _ in range(200):
                if time.time() > deadline:
                    break

                if n < 2:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue

                had_edge = v in adj[u]
                if had_edge:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)

                if is_valid(adj, n):
                    # Try to add a vertex
                    new_n = n + 1
                    if new_n <= 40:
                        for attempt2 in range(100):
                            if time.time() > deadline:
                                break
                            density = random.uniform(0.3, 0.7)
                            new_v_neighbors = set()
                            for i in range(n):
                                if random.random() < density:
                                    new_v_neighbors.add(i)
                            if check_new_vertex_valid(adj, n, new_v_neighbors):
                                trial_adj = [set(s) for s in adj]
                                trial_adj.append(set())
                                for i in new_v_neighbors:
                                    trial_adj[i].add(n)
                                    trial_adj[n].add(i)
                                adj = trial_adj
                                n = new_n
                                best_n = n
                                best_adj = [set(s) for s in adj]
                                improved = True
                                break
                    if improved:
                        break
                else:
                    # Revert
                    if had_edge:
                        adj[u].add(v)
                        adj[v].add(u)
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)

            if improved:
                continue
            else:
                # Revert to best known
                adj = [set(s) for s in best_adj]
                n = best_n
                if time.time() > deadline - 30:
                    break

    # Return best result
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
