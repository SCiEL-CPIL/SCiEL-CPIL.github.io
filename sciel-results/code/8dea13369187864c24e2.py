"""
Pentagon-free graphs without independent 9-sets (Ramsey number R(C5,K9)).

Strategy:
  1. Start from 8 disjoint K4s (32 vertices). This is known to be valid:
     - Each K4 is a clique, so no C5 within a component.
     - The graph is a disjoint union, so any cycle must lie within a single K4.
       K4 has no C5. So no C5 in the whole graph.
     - Independence number of disjoint union = sum of independence numbers.
       alpha(K4) = 1, so alpha(total) = 8 < 9. Valid.

  2. Try to extend to 33+ vertices using local search / simulated annealing:
     - Add vertices and try to connect them to existing ones to keep alpha < 9.
     - Or start from the 32-vertex graph and try to add vertices with appropriate edges.
     - The challenge: adding a vertex to a C5-free graph while keeping it C5-free
       and reducing or maintaining independence number is tricky.

  3. Alternative structural approach: use Turan-type constructions or other
     known C5-free graphs with small independence number.

Given the time limit and the fact that 32 is the known lower bound, I'll implement:
  - A robust 32-vertex solution (8 disjoint K4s) as fallback.
  - A heuristic search that tries to find a 33+ vertex graph within the time budget.

The search will:
  - Start from 8 disjoint K4s.
  - Try to add one vertex at a time, connecting it to existing vertices in a way that:
    a) Does not create a C5.
    b) Keeps the independence number < 9.
  - Use a greedy + local improvement approach.
"""

import time
import random
import sys

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj (list of lists of bool) on n vertices has a 5-cycle.
    Uses a simple O(n^5) check - too slow for n=33+. Instead use a smarter approach."""
    pass

def _has_c5_smart(adj, n, deadline=None):
    """
    Check for C5 in graph. A C5 is a cycle of length 5.
    We can check by trying all 5-tuples (v0,v1,v2,v3,v4) with distinct vertices
    such that edges exist between consecutive and v4-v0.
    To speed up: for each vertex v0, for each neighbor v1 of v0 (v1 != v0),
    for each neighbor v2 of v1 (v2 != v0, v1), for each neighbor v3 of v2 (v3 != v0,v1,v2),
    for each neighbor v4 of v3 (v4 != v0,v1,v2,v3), check if v4 is neighbor of v0 and v4 != v1.
    Also need v0,v1,v2,v3,v4 all distinct.
    
    This is still O(n * d^4) which for n=33, d~10 is 33*10000 = 330000, acceptable.
    But we can prune more.
    """
    if deadline is not None and time.time() > deadline:
        return True  # Assume C5 exists to be safe if we run out of time
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if not adj[v3][v4]:
                            continue
                        if not adj[v4][v0]:
                            continue
                        # Check that v1 is not neighbor of v4 (to ensure it's a simple 5-cycle, not shorter)
                        # Actually for C5 we just need the 5 vertices to form a cycle. The edges are:
                        # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0. We need these 5 edges and the vertices distinct.
                        # We also need to ensure no "chord" that would make it a shorter cycle? 
                        # No, a C5 as a subgraph just means there exists 5 vertices forming a cycle of length 5.
                        # The edges between non-consecutive vertices don't matter for the subgraph being a C5.
                        # So we just need the 5 edges above and 5 distinct vertices.
                        return True
    return False

def _independence_number(adj, n):
    """Compute the independence number of the graph. Use a simple O(2^n) for small n, or heuristic for larger.
    For n up to 35, we can use a branch and bound or just try all subsets if n is small enough.
    2^35 is too big. We need a better approach.
    
    Use a greedy + local search, or use the fact that we can check if alpha >= k by trying to find an independent set of size k.
    For our purposes, we need to check if alpha < 9, i.e., no independent set of size 9.
    """
    # Check if there exists an independent set of size 9
    # Use backtracking
    # This is NP-hard but n <= 35 and we're checking for size 9, which is small.
    # We can use a simple recursive search with pruning.
    
    # Sort vertices by degree (ascending) to prune more effectively
    # Actually, let's use a simple approach: try all combinations of 9 vertices? C(35,9) is about 70 million, too many.
    
    # Better: backtracking with pruning
    # At each step, try to add a vertex that is independent of all previously chosen vertices.
    # Prune if remaining vertices + current size < 9.
    
    def exists_independent_set_of_size_k(k):
        # Backtracking
        # Order vertices by degree (higher degree first might prune better? or lower?)
        # Let's try: at each step, the set of candidates is those not adjacent to any chosen vertex.
        # If len(candidates) < k - len(chosen), prune.
        
        # Use a recursive function
        # To avoid deep recursion, we can use iterative or set recursion limit
        sys.setrecursionlimit(10000)
        
        # Precompute adjacency as sets for fast lookup
        adj_sets = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_sets[i].add(j)
        
        # Candidates that can be added to an independent set containing 'chosen'
        # We'll use a DFS that at each level picks a vertex from the current candidate set.
        
        def dfs(chosen, candidates):
            if len(chosen) == k:
                return True
            remaining = k - len(chosen)
            if len(candidates) < remaining:
                return False
            # Try each vertex in candidates
            # To speed up, we can try in some order
            cands_list = list(candidates)
            random.shuffle(cands_list)
            for v in cands_list:
                # New candidates: those in candidates that are not adjacent to v and not v itself
                new_cands = set()
                for u in cands_list:
                    if u != v and u not in adj_sets[v]:
                        new_cands.add(u)
                if dfs(chosen + [v], new_cands):
                    return True
            return False
        
        # Initial candidates: all vertices
        return dfs([], set(range(n)))
    
    return exists_independent_set_of_size_k(9)

def _is_valid(adj, n, deadline=None):
    """Check if the graph is C5-free and has no independent set of size 9."""
    if deadline is not None and time.time() > deadline:
        return False
    if _has_c5_smart(adj, n, deadline):
        return False
    if _independence_number(adj, n):
        return False
    return True

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with 8 disjoint K4s: 32 vertices
    n = 32
    adj = [[False] * n for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i][base + j] = True
                adj[base + j][base + i] = True
    
    # Verify base case
    if not _is_valid(adj, n, deadline):
        # Should not happen, but fallback to 7 K4s
        n = 28
        adj = [[False] * n for _ in range(n)]
        for b in range(7):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[base + i][base + j] = True
                    adj[base + j][base + i] = True
        best_n = n
        best_adj = [row[:] for row in adj]
    else:
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Try to extend to 33+ vertices
    # Strategy: try to add a vertex and connect it to some existing vertices
    # The new vertex must not create a C5 and must not allow an independent set of size 9.
    
    # Alternative: use a different base construction
    # For now, try adding vertices one by one
    
    current_n = best_n
    current_adj = [row[:] for row in best_adj]
    
    # Try to add vertices from 33 up to 35
    for target_n in range(current_n + 1, 36):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations for the new vertex connections
        found = False
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Start from current graph, add one vertex
            new_n = current_n + 1
            if new_n > target_n:
                continue
            if new_n != target_n:
                continue  # Only try to add one at a time
            
            new_adj = [row[:] + [False] for row in current_adj]
            new_adj.append([False] * new_n)
            
            # Randomly connect the new vertex to some existing vertices
            # The new vertex (index new_n-1) should be connected to a subset of existing vertices
            # To avoid C5, the new vertex's neighbors should form an independent set in the subgraph induced by them?
            # Actually, if the new vertex is connected to two vertices a and b that are at distance 2 in the existing graph,
            # we might create a 4-cycle, but not necessarily a 5-cycle.
            # A 5-cycle involving the new vertex would be: new - v1 - v2 - v3 - v4 - new
            # where v1 and v4 are neighbors of new, and v1-v2-v3-v4 is a path of length 3 in the existing graph.
            
            # For now, just try random subsets and check validity
            subset_size = random.randint(0, min(10, current_n))
            neighbors = random.sample(range(current_n), subset_size)
            for v in neighbors:
                new_adj[new_n - 1][v] = True
                new_adj[v][new_n - 1] = True
            
            if _is_valid(new_adj, new_n, deadline):
                current_n = new_n
                current_adj = new_adj
                best_n = new_n
                best_adj = [row[:] for row in new_adj]
                found = True
                break
        
        if not found:
            # Try a different approach: modify the existing graph
            # Or just give up on this target_n
            continue
    
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with 8 disjoint K4s: 32 vertices
    n = 32
    adj = [[False] * n for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i][base + j] = True
                adj[base + j][base + i] = True
    
    # Verify base case
    if not _is_valid(adj, n, deadline):
        # Fallback to 7 K4s
        n = 28
        adj = [[False] * n for _ in range(n)]
        for b in range(7):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[base + i][base + j] = True
                    adj[base + j][base + i] = True
        best_n = n
        best_adj = [row[:] for row in adj]
    else:
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Try to extend to 33+ vertices
    current_n = best_n
    current_adj = [row[:] for row in best_adj]
    
    # Try to add vertices from 33 up to 35
    for target_n in range(current_n + 1, 36):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            new_n = current_n + 1
            if new_n != target_n:
                continue
            
            new_adj = [row[:] + [False] for row in current_adj]
            new_adj.append([False] * new_n)
            
            subset_size = random.randint(0, min(10, current_n))
            neighbors = random.sample(range(current_n), subset_size)
            for v in neighbors:
                new_adj[new_n - 1][v] = True
                new_adj[v][new_n - 1] = True
            
            if _is_valid(new_adj, new_n, deadline):
                current_n = new_n
                current_adj = new_adj
                best_n = new_n
                best_adj = [row[:] for row in new_adj]
                found = True
                break
        
        if not found:
            continue
    
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END

