import time
import random
from math import gcd
from functools import lru_cache
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    num_set = set(numbers)
    
    # 1. Identify all bad triples
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = a * b // gcd(a, b)
            for c in range(b + 1, n + 1):
                lac = a * c // gcd(a, c)
                if lac != lab:
                    continue
                lbc = b * c // gcd(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers
    
    # 2. Precompute conflict structures
    # For each vertex x, conflicts[x] is a list of pairs (u,v) such that (x,u,v) is a bad triple
    conflicts = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Helper function to check if x can be added to current_set
    def can_add(x, current_set):
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Strategy 1: Multi-start Greedy + Local Search (warm start)
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        S = set(numbers)
        
        # Compute degree for each vertex
        degree = {x: 0 for x in numbers}
        for t in bad_triples:
            for x in t:
                degree[x] += 1
        
        # Greedy removal
        while True:
            if time.time() > deadline - 30.0:
                break
                
            max_deg = 0
            candidates = []
            for x in S:
                if degree[x] > max_deg:
                    max_deg = degree[x]
                    candidates = [x]
                elif degree[x] == max_deg and max_deg > 0:
                    candidates.append(x)
            
            if not candidates or max_deg == 0:
                break
                
            target = random.choice(candidates)
            S.remove(target)
            
            for t in bad_triples:
                if target in t:
                    others = [x for x in t if x != target]
                    if others[0] in S and others[1] in S:
                        degree[others[0]] -= 1
                        degree[others[1]] -= 1
        
        # Local search: add back vertices
        improved = True
        local_iter = 0
        while improved and local_iter < 50:
            improved = False
            local_iter += 1
            if time.time() > deadline - 30.0:
                break
            
            addable = []
            for x in numbers:
                if x not in S and can_add(x, S):
                    addable.append(x)
            
            if not addable:
                break
            
            target = random.choice(addable)
            S.add(target)
            improved = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 2: Branch and Bound with tight upper bounds
    # Model as Maximum Independent Set on 3-uniform hypergraph
    # Upper bound: use maximum matching on the 2-shadow conflict graph
    
    # Build conflict graph (2-shadow): edge between u,v if there exists w such that (u,v,w) is a bad triple
    conflict_graph = {x: set() for x in numbers}
    for a, b, c in bad_triples:
        conflict_graph[a].add(b)
        conflict_graph[a].add(c)
        conflict_graph[b].add(a)
        conflict_graph[b].add(c)
        conflict_graph[c].add(a)
        conflict_graph[c].add(b)
    
    # Compute maximum matching on conflict graph (greedy approximation for upper bound)
    # A matching of size m means we need at least m vertices to cover all edges
    # So |S| <= n - m (we must remove at least m vertices to break all edges)
    # Actually, we need a vertex cover, not matching. But matching gives a lower bound on vertex cover.
    # Upper bound on independent set = n - minimum vertex cover
    # Minimum vertex cover >= maximum matching (Kőnig's theorem for bipartite, but this isn't bipartite)
    # So we use: |S| <= n - max_matching_size
    
    def compute_max_matching_upper_bound(available_set):
        """Compute a lower bound on vertex cover using greedy matching."""
        if not available_set:
            return 0
        
        # Greedy matching: repeatedly pick an edge and add to matching
        matching_size = 0
        matched = set()
        edges = []
        for u in available_set:
            for v in conflict_graph[u]:
                if v in available_set and u < v:
                    edges.append((u, v))
        
        # Sort edges by some heuristic (e.g., degree sum)
        edges.sort(key=lambda e: len(conflict_graph[e[0]] & available_set) + len(conflict_graph[e[1]] & available_set), reverse=True)
        
        for u, v in edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                matching_size += 1
        
        return matching_size
    
    # Branch and bound
    def branch_and_bound(available, current_set, current_size, depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline - 5.0:
            return
        
        # Upper bound: current_size + (n - max_matching)
        # But we can't use all remaining, only those that can be added
        # Better bound: current_size + number of vertices in available that can be added without conflict
        # Even better: use the matching bound on the induced subgraph
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Pruning: if current_size + len(available) <= best_size, no improvement possible
        if current_size + len(available) <= best_size:
            return
        
        # Compute upper bound using matching
        matching_lb = compute_max_matching_upper_bound(available)
        upper_bound = current_size + len(available) - matching_lb
        if upper_bound <= best_size:
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in conflict graph
        max_deg = -1
        pivot = None
        for x in available:
            deg = len(conflict_graph[x] & available)
            if deg > max_deg:
                max_deg = deg
                pivot = x
        
        if pivot is None:
            # No conflicts, take all
            if current_size + len(available) > best_size:
                best_size = current_size + len(available)
                best_set = set(current_set) | set(available)
            return
        
        # Branch 1: Include pivot
        # Check if pivot can be added to current_set
        if can_add(pivot, current_set):
            new_available = set(available)
            new_available.remove(pivot)
            # Remove all vertices that conflict with pivot
            for neighbor in conflict_graph[pivot]:
                if neighbor in new_available:
                    # Check if neighbor can still be added (it might conflict with current_set)
                    # Actually, if we include pivot, we can't include any neighbor that forms a bad triple with pivot and some vertex in current_set
                    # But for simplicity, we just remove neighbors from available (they can't be in the independent set if pivot is)
                    # Wait, this is not correct. In a 3-uniform hypergraph, including pivot doesn't necessarily exclude all neighbors.
                    # We need to be more careful.
                    pass
            
            # For now, let's use a simpler approach: just try adding pivot and see what remains
            # This is a heuristic, not exact
            branch_and_bound(new_available, current_set | {pivot}, current_size + 1, depth + 1)
        
        # Branch 2: Exclude pivot
        new_available = set(available)
        new_available.remove(pivot)
        branch_and_bound(new_available, current_set, current_size, depth + 1)
    
    # Run branch and bound with time limit
    available = set(numbers)
    current_set = set(best_set) if best_set else set()
    current_size = len(current_set)
    
    # Remove vertices already in current_set from available
    available -= current_set
    
    branch_and_bound(available, current_set, current_size, 0)
    
    if best_size == 0:
        return [1]
        
    return list(best_set)
