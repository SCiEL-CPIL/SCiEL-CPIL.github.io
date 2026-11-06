import random
import time
import sys
from itertools import combinations

def _has_triangle(adj, n):
    """Check if graph has a triangle using adjacency matrix."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                # Check common neighbors
                common = [k for k in range(j + 1, n) if adj[i][k] and adj[j][k]]
                if common:
                    return True
    return False

def _max_independent_set_size(adj, n, timeout=5.0):
    """Find maximum independent set size using branch and bound with time limit."""
    start_time = time.time()
    best = 0
    best_set = []
    
    # Use a greedy approach with backtracking
    def greedy_independent(adj, n):
        remaining = set(range(n))
        result = []
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and adj[v][u])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            result.append(min_v)
            remaining.discard(min_v)
            # Remove neighbors of min_v
            to_remove = set()
            for u in remaining:
                if adj[min_v][u]:
                    to_remove.add(u)
            remaining -= to_remove
        return result
    
    # Get a good initial solution
    initial = greedy_independent(adj, n)
    best = len(initial)
    best_set = initial[:]
    
    # Try to improve with local search
    def can_add_to_independent(current_set, v, adj, n):
        for u in current_set:
            if adj[v][u]:
                return False
        return True
    
    # Branch and bound with time check
    def branch_bound(remaining, current_set, depth):
        nonlocal best, best_set
        if time.time() - start_time > timeout:
            return
        
        if not remaining:
            if len(current_set) > best:
                best = len(current_set)
                best_set = current_set[:]
            return
        
        # Upper bound: current size + remaining count
        if len(current_set) + len(remaining) <= best:
            return
        
        # Pick a vertex to branch on
        v = list(remaining)[0]
        remaining_list = list(remaining)
        remaining_list.remove(v)
        
        # Option 1: include v
        if can_add_to_independent(current_set, v, adj, n):
            new_remaining = set()
            for u in remaining_list:
                if not adj[v][u]:
                    new_remaining.add(u)
            branch_bound(new_remaining, current_set + [v], depth + 1)
        
        # Option 2: exclude v
        branch_bound(set(remaining_list), current_set, depth + 1)
    
    branch_bound(set(range(n)), [], 0)
    return best

def _is_valid(adj, n):
    """Check if graph is triangle-free and has no independent set of size 10."""
    if _has_triangle(adj, n):
        return False
    alpha = _max_independent_set_size(adj, n, timeout=3.0)
    return alpha < 10

def _build_adj_from_edges(edges, n):
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    best_n = 26
    best_adj = None
    
    # Start with Andrasfai graph And(9) as seed
    n = 26
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i][j] = 1
                adj[j][i] = 1
    
    if _is_valid(adj, n):
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Try to grow the graph by adding vertices
    # For each new vertex, try all possible connections to existing vertices
    # that maintain triangle-free property
    
    def try_add_vertex(adj, n):
        """Try to add a new vertex connected to a subset of existing vertices."""
        # The new vertex v can be connected to a set S of existing vertices
        # such that S is an independent set in G (no edges within S)
        # and no triangle is formed (which is guaranteed if S is independent)
        
        # Find all maximal independent sets in current graph
        # For each, try adding a vertex connected to that independent set
        
        # Greedy approach: find a good independent set to connect to
        remaining = set(range(n))
        independent_sets = []
        
        # Generate some independent sets
        def find_independent_sets(adj, n, max_sets=50):
            sets = []
            # Try different starting vertices
            for start in range(min(n, 20)):
                current_set = [start]
                candidates = set(range(start + 1, n))
                
                while candidates:
                    # Pick vertex with min degree among candidates
                    best_v = -1
                    min_deg = float('inf')
                    for v in candidates:
                        deg = sum(1 for u in current_set if adj[v][u])
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                    
                    if best_v == -1 or min_deg > 0:
                        break
                    
                    current_set.append(best_v)
                    candidates.discard(best_v)
                    # Remove neighbors of best_v from candidates
                    to_remove = set()
                    for u in candidates:
                        if adj[best_v][u]:
                            to_remove.add(u)
                    candidates -= to_remove
                
                if len(current_set) >= 3:
                    sets.append(current_set)
            
            # Also try random independent sets
            for _ in range(20):
                random.shuffle(range(n))
                current_set = []
                for v in range(n):
                    if all(not adj[v][u] for u in current_set):
                        current_set.append(v)
                if len(current_set) >= 3:
                    sets.append(current_set)
            
            return sets[:max_sets]
        
        ind_sets = find_independent_sets(adj, n)
        
        for ind_set in ind_sets:
            if time.time() > deadline:
                return None
            
            # Create new graph with vertex n connected to ind_set
            new_n = n + 1
            new_adj = [row[:] + [0] for row in adj]
            new_adj.append([0] * new_n)
            
            for v in ind_set:
                new_adj[n][v] = 1
                new_adj[v][n] = 1
            
            # Check if valid
            if _is_valid(new_adj, new_n):
                return new_adj
        
        return None
    
    # Try to grow from current best
    current_adj = [row[:] for row in best_adj]
    current_n = best_n
    
    # Multiple attempts at growing
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Try different randomizations
        temp_adj = [row[:] for row in current_adj]
        temp_n = current_n
        
        # Try adding one vertex at a time
        added = False
        for _ in range(5):
            if time.time() > deadline:
                break
            
            new_adj = try_add_vertex(temp_adj, temp_n)
            if new_adj is not None:
                temp_adj = new_adj
                temp_n += 1
                added = True
        
        if added and temp_n > best_n:
            best_n = temp_n
            best_adj = [row[:] for row in temp_adj]
            current_adj = [row[:] for row in temp_adj]
            current_n = temp_n
    
    # If we couldn't grow, try simulated annealing on edge flips to find
    # a graph with smaller independence number, which might allow growth
    if best_n <= 26 and time.time() < deadline - 20:
        # Try random triangle-free graphs with various edge densities
        for target_density in [0.15, 0.2, 0.25, 0.3, 0.35]:
            if time.time() > deadline - 10:
                break
            
            for trial in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Generate random triangle-free graph
                # Use Turan-like construction: partition into parts
                # Random partition of 30 vertices into 3 parts
                vertices = list(range(30))
                random.shuffle(vertices)
                
                # Try different part sizes
                part_sizes = [10, 10, 10]
                parts = []
                idx = 0
                for size in part_sizes:
                    parts.append(vertices[idx:idx+size])
                    idx += size
                
                # Complete multipartite graph (Turan graph T(30,3))
                # This is triangle-free
                adj_try = [[0] * 30 for _ in range(30)]
                for i in range(3):
                    for j in range(i + 1, 3):
                        for u in parts[i]:
                            for v in parts[j]:
                                adj_try[u][v] = 1
                                adj_try[v][u] = 1
                
                # Remove some random edges to reduce independence number
                edges_to_consider = []
                for i in range(30):
                    for j in range(i + 1, 30):
                        if adj_try[i][j]:
                            edges_to_consider.append((i, j))
                
                # Remove 30% of edges randomly
                num_remove = int(len(edges_to_consider) * 0.3)
                to_remove = set(random.sample(range(len(edges_to_consider)), num_remove))
                for idx in to_remove:
                    i, j = edges_to_consider[idx]
                    adj_try[i][j] = 0
                    adj_try[j][i] = 0
                
                # Check validity
                if _is_valid(adj_try, 30):
                    if 30 > best_n:
                        best_n = 30
                        best_adj = [row[:] for row in adj_try]
    
    # Final check and return
    if best_adj is not None and best_n >= 26:
        edges = _edges_from_adj(best_adj, best_n)
        return {'n': best_n, 'edges': edges}
    else:
        # Fallback to And(9)
        n = 26
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i][j] = 1
                    adj[j][i] = 1
        edges = _edges_from_adj(adj, n)
        return {'n': n, 'edges': edges}
