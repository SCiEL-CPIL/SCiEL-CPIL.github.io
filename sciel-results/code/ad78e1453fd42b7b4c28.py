import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s safety margin (90s limit)
    
    n = 15
    all_edges = list(combinations(range(n), 2))
    
    def build_adj_and_deg(edge_set):
        """Build adjacency sets and degree array from edge set."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def check_valid(edge_set):
        """Check if edge_set satisfies the constraint."""
        if time.time() > deadline:
            return False
        adj, deg = build_adj_and_deg(edge_set)
        
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def count_violations(edge_set):
        """Count number of violating paths."""
        adj, deg = build_adj_and_deg(edge_set)
        violations = 0
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        violations += 1
        return violations
    
    def try_add_edge(edge_set, e):
        """Try to add edge e to edge_set, return new set if valid else None."""
        if e in edge_set:
            return edge_set
        candidate = edge_set | {e}
        if check_valid(candidate):
            return candidate
        return None
    
    def local_search_add(edge_set):
        """Repeatedly try to add any missing edge that doesn't violate."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for e in all_edges:
                if e in edge_set:
                    continue
                new_set = try_add_edge(edge_set, e)
                if new_set is not None:
                    edge_set = new_set
                    improved = True
                    break
        return edge_set
    
    def swap_search(edge_set, num_trials=100):
        """Try removing one edge and adding another to improve."""
        best = edge_set
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                break
            if len(edge_set) == 0:
                break
            e_remove = random.choice(list(edge_set))
            candidate = edge_set - {e_remove}
            # Try to add back a different edge
            for e_add in all_edges:
                if e_add in candidate or e_add == e_remove:
                    continue
                new_set = try_add_edge(candidate, e_add)
                if new_set is not None and len(new_set) > len(best):
                    best = new_set
                    edge_set = new_set
                    break
        return best
    
    # Strategy 1: Threshold graph construction (guaranteed distinct degrees)
    # A threshold graph is built by adding vertices in order, each either
    # isolated or dominating (connected to all previous). This gives distinct degrees.
    # We try all 2^14 orderings (which vertex is isolated vs dominating at each step).
    
    def build_threshold_graph(order):
        """Build a threshold graph given the order of 'dominating' vs 'isolated'.
        order[i] = True means vertex i is dominating (connected to all previous),
        False means isolated (connected to no previous).
        We assign vertices 0..14 in some sequence."""
        edge_set = set()
        # We'll assign vertices in order 0, 1, 2, ..., 14
        # For each vertex v, if it's dominating, connect to all previous vertices
        for v in range(n):
            if order[v]:
                for u in range(v):
                    edge_set.add((u, v))
        return edge_set
    
    best_edges = set()
    best_count = 0
    
    # Try all threshold graph constructions
    # There are 2^14 = 16384 possible threshold graphs (vertex 0 is always first, 
    # but its type doesn't matter since it has no predecessors)
    # Actually vertex 0's type is irrelevant (no previous vertices), so 2^14 for vertices 1-14
    for mask in range(1 << 14):
        if time.time() > deadline - 30:
            break
        order = [False] * n
        for i in range(14):
            order[i + 1] = (mask >> i) & 1
        # Vertex 0: doesn't matter, set to False
        order[0] = False
        
        edge_set = build_threshold_graph(order)
        if check_valid(edge_set):
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edges = set(edge_set)
    
    # Strategy 2: Greedy + local search from random starts
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed * 7919 + 13)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        current_edges = set()
        for e in shuffled:
            if time.time() > deadline - 10:
                break
            new_set = try_add_edge(current_edges, e)
            if new_set is not None:
                current_edges = new_set
        
        # Local search to fill in
        current_edges = local_search_add(current_edges)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    # Strategy 3: Start from best found, do more aggressive local search
    if time.time() < deadline - 5:
        best_edges = local_search_add(best_edges)
    
    # Strategy 4: Try to improve by swapping edges
    if time.time() < deadline - 3:
        best_edges = swap_search(best_edges, num_trials=200)
    
    # Final polish
    if time.time() < deadline - 2:
        best_edges = local_search_add(best_edges)
    
    # Verify final result
    if not check_valid(best_edges):
        # Fallback: use a known valid small graph
        # Complete graph K_8 on vertices 0-7 has all degrees 7, which violates!
        # Use empty graph as ultimate fallback
        best_edges = set()
    
    # Convert to list of [u, v] with proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
