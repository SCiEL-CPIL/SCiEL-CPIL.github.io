"""Construct a K4-free graph with alpha <= 5, maximizing n within time limit."""
import time
import random
import itertools
import sys

# We'll use a repair-based local search approach:
# 1. Start with a known valid graph (e.g., Paley-17 or a small valid graph)
# 2. Try to add vertices one at a time, choosing edges to maintain K4-free and alpha<=5
# 3. If stuck, use local search to repair/adjust the graph

def has_k4(adj, n):
    """Check if graph has a K4. adj is a list of sets."""
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                continue
            for k in range(j+1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                for l in range(k+1, n):
                    if l not in adj[i] or l not in adj[j] or l not in adj[k]:
                        continue
                    return True
    return False

def max_independent_set_size(adj, n, limit=6):
    """Check if there's an independent set of size >= limit. Returns size if found, else -1."""
    # Use a simple branching approach with pruning
    # For efficiency, we'll use a greedy + backtracking hybrid
    
    # First try greedy to find a large independent set
    def greedy_is():
        remaining = set(range(n))
        is_set = []
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = n
            best_v = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            is_set.append(best_v)
            remaining.discard(best_v)
            remaining -= adj[best_v]
        return is_set
    
    # Check if greedy finds one of size >= limit
    g_is = greedy_is()
    if len(g_is) >= limit:
        return len(g_is)
    
    # Try to find an independent set of size exactly limit via backtracking
    # with strong pruning
    found = [False]
    
    def backtrack(remaining, current):
        if found[0]:
            return
        if len(current) + len(remaining) < limit:
            return
        if len(current) == limit:
            found[0] = True
            return
        # Branching: pick a vertex to include or exclude
        # Use the vertex with highest degree in remaining for pruning
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        # Include v
        new_remaining = remaining - {v} - adj[v]
        backtrack(new_remaining, current + [v])
        if found[0]:
            return
        # Exclude v
        backtrack(remaining - {v}, current)
    
    backtrack(set(range(n)), [])
    if found[0]:
        return limit
    return len(g_is)

def is_valid(adj, n):
    """Check if graph is K4-free and alpha < 6."""
    if has_k4(adj, n):
        return False
    if max_independent_set_size(adj, n, 6) >= 6:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with Paley graph on 17 vertices (known valid)
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 17 in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Verify initial graph is valid
    if not is_valid(adj, n):
        # Fallback: use a small valid graph
        n = 8
        adj = [set() for _ in range(n)]
        # Complete bipartite K4,4 is K4-free? No, K4,4 has K4? 
        # Actually K4,4 contains no K4 since it's bipartite. Alpha of K4,4 is 4 (one side).
        # So K4,4 works for n=8.
        for i in range(4):
            for j in range(4, 8):
                adj[i].add(j)
                adj[j].add(i)
        best_n = n
        best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices
    while time.time() < deadline:
        new_n = n + 1
        if new_n > 40:
            break
        
        # Try to add a new vertex with various edge patterns
        success = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Randomly decide edges from new vertex to existing vertices
            # with bias toward creating a valid graph
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            # For each existing vertex, decide whether to connect
            # Use a strategy: connect to vertices that are "sparse" to avoid K4
            # and to break up large independent sets
            
            # Strategy: for each vertex i, connect if:
            # 1. It doesn't create a K4 with any two neighbors of i
            # 2. It helps break independent sets
            
            candidate_edges = set()
            for i in range(n):
                # Check if adding edge (i, new_n-1) would create a K4
                # A K4 would be {i, new, j, k} where j,k are common neighbors of i and new
                # Since new only connects to i so far, we need to check incrementally
                # Simpler: just add randomly and check
                if random.random() < 0.5:
                    candidate_edges.add(i)
            
            # Apply edges
            for i in candidate_edges:
                new_adj[i].add(new_n - 1)
                new_adj[new_n - 1].add(i)
            
            # Check validity
            if is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                success = True
                best_n = n
                best_adj = [set(s) for s in adj]
                break
        
        if not success:
            # Try local search: modify some edges in current graph to enable growth
            # Or try a different random seed for edge selection
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Try flipping a random edge
                if n < 2:
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                
                # Toggle edge
                if v in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                
                if is_valid(adj, n):
                    # Try to add a vertex again
                    new_n = n + 1
                    if new_n <= 40:
                        for attempt2 in range(50):
                            if time.time() > deadline:
                                break
                            trial_adj = [set(s) for s in adj]
                            trial_adj.append(set())
                            for i in range(n):
                                if random.random() < 0.5:
                                    trial_adj[i].add(new_n - 1)
                                    trial_adj[new_n - 1].add(i)
                            if is_valid(trial_adj, new_n):
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
                    if v in adj[u]:
                        adj[u].discard(v)
                        adj[v].discard(u)
                    else:
                        adj[u].add(v)
                        adj[v].add(u)
            
            if improved:
                continue
            else:
                # Revert to best known
                adj = [set(s) for s in best_adj]
                n = best_n
                # If we can't grow, break
                if time.time() > deadline - 30:
                    break
    
    # Return best result
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
