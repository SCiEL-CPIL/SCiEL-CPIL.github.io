"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    base_edges = [(u, v) for u, v in P['edges']]
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # All candidate edges: not in base, and |u-v| >= 3 (to avoid triangles with path)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            if abs(v - u) <= 2:
                continue
            all_candidates.append((u, v))
    
    def _build_adj(added_edges):
        """Build adjacency list from base + added edges."""
        adj = [set(s) for s in base_adj]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def _is_triangle_free(added_edges):
        """Check if base + added is triangle-free."""
        adj = _build_adj(added_edges)
        # For each vertex, check if any two neighbors are connected
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
    
    def _diameter_leq_four(added_edges):
        """Check if diameter <= 4 and graph is connected."""
        adj = _build_adj(added_edges)
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > 4:
                            return False
                        queue.append(nb)
            if -1 in dist:
                return False
        return True
    
    def _is_valid(added_edges):
        """Full validation."""
        if not _is_triangle_free(added_edges):
            return False
        if not _diameter_leq_four(added_edges):
            return False
        return True
    
    def _validate_and_count(added_edges):
        """Validate and return True if valid."""
        return _is_valid(added_edges)
    
    # Precompute: for each candidate edge, check if it's triangle-free with base
    # An edge (u,v) is safe if u and v have no common neighbor in base
    safe_candidates = []
    for u, v in all_candidates:
        if base_adj[u] & base_adj[v]:
            continue
        safe_candidates.append((u, v))
    
    # Strategy: Use simulated annealing / local search with bipartite-aware structure.
    # Key insight: Triangle-free graphs are bipartite if we can 2-color them.
    # The path is bipartite with even/odd coloring. Adding edges between same-parity
    # vertices breaks bipartiteness but may still be triangle-free.
    
    # Let's use a different approach: iterative improvement with smart initialization.
    
    # First, try to find a good solution using a structured approach.
    # The path has 24 vertices. We need diameter <= 4.
    
    # Approach: Use a set of "hub" edges that create shortcuts.
    # Try various configurations and use local search to improve.
    
    best_solution = None
    best_len = float('inf')
    
    # Try multiple random initializations and local search
    random.seed(42)
    
    def _local_search(initial_edges, max_iter=5000):
        """Local search: try removing edges, swapping edges."""
        current = list(initial_edges)
        if not _is_valid(current):
            # Try to fix by removing problematic edges
            # For now, just return as-is
            return current
        
        current_len = len(current)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try removing an edge
            for i in range(len(current)):
                candidate = [e for j, e in enumerate(current) if j != i]
                if _is_valid(candidate):
                    current = candidate
                    current_len -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            if len(current) > 0:
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    # Try adding a random safe candidate
                    for _ in range(20):
                        new_edge = random.choice(safe_candidates)
                        if new_edge in current:
                            continue
                        candidate = [e for j, e in enumerate(current) if j != i]
                        candidate.append(new_edge)
                        if _is_valid(candidate):
                            current = candidate
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    # Generate multiple initial solutions and run local search
    initial_solutions = []
    
    # Solution 1: Star-like from center
    sol1 = [(0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
            (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)]
    initial_solutions.append(sol1)
    
    # Solution 2: Two hubs
    sol2 = [(0, 8), (2, 8), (4, 8), (6, 8),
            (14, 16), (16, 23), (18, 16), (20, 16), (22, 16),
            (8, 16)]
    initial_solutions.append(sol2)
    
    # Solution 3: Ring-like
    sol3 = [(0, 6), (6, 12), (12, 18), (18, 23),
            (0, 12), (6, 18), (12, 23)]
    initial_solutions.append(sol3)
    
    # Solution 4: Sparse shortcuts
    sol4 = [(0, 10), (10, 20), (20, 23), (0, 14), (14, 23), (5, 15), (15, 23)]
    initial_solutions.append(sol4)
    
    # Solution 5: Based on distance-2 jumps
    sol5 = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23),
            (2, 7), (7, 12), (12, 17), (17, 22)]
    initial_solutions.append(sol5)
    
    # Random initializations
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Random set of 6-10 edges
        k = random.randint(6, 10)
        sol = random.sample(safe_candidates, k)
        initial_solutions.append(sol)
    
    # Run local search on each
    for sol in initial_solutions:
        if time.time() > deadline - 10:
            break
        result = _local_search(sol)
        if _is_valid(result):
            if len(result) < best_len:
                best_len = len(result)
                best_solution = result
    
    # If we found a good solution, try to improve further with more aggressive search
    if best_solution is not None:
        # Try to reduce further
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline - 5:
                    break
                candidate = [e for j, e in enumerate(best_solution) if j != i]
                if _is_valid(candidate):
                    best_solution = candidate
                    improved = True
                    break
            if not improved:
                # Try edge swaps
                for i in range(len(best_solution)):
                    if time.time() > deadline - 5:
                        break
                    for _ in range(50):
                        new_edge = random.choice(safe_candidates)
                        if new_edge in best_solution:
                            continue
                        candidate = [e for j, e in enumerate(best_solution) if j != i]
                        candidate.append(new_edge)
                        if _is_valid(candidate):
                            best_solution = candidate
                            improved = True
                            break
                    if improved:
                        break
    
    # Final check
    if best_solution is not None and _is_valid(best_solution):
        return [[u, v] for u, v in best_solution]
    
    # Fallback: return a known valid solution
    fallback = [(0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
                (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)]
    if _is_valid(fallback):
        return [[u, v] for u, v in fallback]
    
    return []

# EVOLVE-BLOCK-END
