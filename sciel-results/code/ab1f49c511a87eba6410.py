"""Find maximum reciprocal-subsum-free subset of [1,24]."""
import time
from fractions import Fraction
from itertools import combinations

# Precompute all conflict relationships
# For each a, find all subsets T of [1..24]\{a} where 1/a = sum(1/t for t in T)
# Then a conflicts with every element in T

def _build_conflict_graph():
    """Build adjacency list for conflict graph.
    Two elements conflict if one's reciprocal is a sum of reciprocals of a subset containing the other.
    """
    n = 24
    universe = list(range(1, n + 1))
    
    # For each a, find all subsets T of [1..24]\{a} with sum(1/t) = 1/a
    # Use DFS with pruning
    conflicts = {a: set() for a in universe}
    
    def find_subsets_for_target(a, available, target, start, current_sum, current):
        """Find subsets of available[start:] that sum to target."""
        if current_sum == target and len(current) > 0:
            for x in current:
                conflicts[a].add(x)
            return
        if current_sum > target:
            return
        if start >= len(available):
            return
        
        remaining = target - current_sum
        # Upper bound: sum of all remaining elements from start
        max_possible = sum(Fraction(1, x) for x in available[start:])
        if current_sum + max_possible < target:
            return
        
        # Try including available[start]
        x = available[start]
        new_sum = current_sum + Fraction(1, x)
        if new_sum <= target:
            find_subsets_for_target(a, available, target, start + 1, new_sum, current + [x])
        
        # Try excluding available[start]
        find_subsets_for_target(a, available, target, start + 1, current_sum, current)
    
    for a in universe:
        available = [x for x in universe if x != a]
        available.sort()
        target = Fraction(1, a)
        find_subsets_for_target(a, available, target, 0, Fraction(0), [])
    
    # Make conflict graph symmetric
    for a in universe:
        for b in conflicts[a]:
            conflicts[b].add(a)
    
    return conflicts


def _max_clique_branch_and_bound(adj, n, deadline):
    """Find maximum clique in graph adj (list of sets) using branch and bound.
    adj[i] is the set of neighbors of vertex i.
    """
    best_clique = []
    best_size = 0
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    # Remap: position in order -> vertex
    vertex_at = [order[i] for i in range(n)]
    pos_of = {order[i]: i for i in range(n)}
    
    # Build remapped adjacency
    remapped_adj = [set() for _ in range(n)]
    for i in range(n):
        v = vertex_at[i]
        for u in adj[v]:
            remapped_adj[i].add(pos_of[u])
    
    # Bron-Kerbosch with pivot and coloring for maximum clique
    # Use a simple branch and bound with upper bound from remaining vertices
    
    def upper_bound(candidates):
        """Upper bound on clique size from candidates: 1 + max clique in neighbors intersection."""
        # Simple bound: number of candidates
        return len(candidates)
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if len(clique) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        # Pick a pivot vertex to minimize branching
        # Use the vertex with max degree in candidates
        pivot_candidates = list(candidates)
        if not pivot_candidates:
            return
        
        # Choose pivot: vertex in candidates with most neighbors in candidates
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & candidates))
        
        # Branch on non-neighbors of pivot in candidates
        # (vertices that don't conflict with pivot can potentially be in the clique)
        non_neighbors = [v for v in candidates if v not in adj[pivot]]
        
        if not non_neighbors:
            # All candidates are neighbors of pivot, so all can be added
            if len(clique) + len(candidates) > best_size:
                new_clique = clique + list(candidates)
                if len(new_clique) > best_size:
                    best_size = len(new_clique)
                    best_clique = new_clique
            return
        
        for v in non_neighbors:
            if time.time() > deadline:
                return
            # Include v in clique
            new_clique = clique + [v]
            # New candidates: intersection of current candidates with neighbors of v
            new_candidates = candidates & adj[v]
            new_candidates.discard(v)
            expand(new_clique, new_candidates)
            # Remove v from candidates for next iterations
            candidates.discard(v)
            if len(clique) + len(candidates) <= best_size:
                break
    
    # Start with empty clique and all vertices as candidates
    all_vertices = set(range(n))
    expand([], all_vertices)
    
    return best_clique


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = 24
    universe = list(range(1, n + 1))
    
    # Build conflict graph
    conflicts = _build_conflict_graph()
    
    # Create adjacency list for non-conflict graph (complement)
    # Two vertices are adjacent in non-conflict graph if they don't conflict
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in conflicts[universe[i]]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find maximum clique in non-conflict graph
    best_clique = _max_clique_branch_and_bound(adj, n, deadline)
    
    if time.time() > deadline:
        # Fallback: return a known good set
        return list(range(13, 25))
    
    if not best_clique:
        return list(range(13, 25))
    
    # Convert back to original elements
    result = [universe[v] for v in best_clique]
    result.sort()
    
    # Verify the result is valid
    def is_valid(S):
        S = set(S)
        for a in S:
            others = S - {a}
            # Check if 1/a equals sum of reciprocals of any subset of others
            for subset_size in range(1, len(others) + 1):
                for subset in combinations(sorted(others), subset_size):
                    s = sum(Fraction(1, x) for x in subset)
                    if s == Fraction(1, a):
                        return False
        return True
    
    if not is_valid(result):
        # Fallback
        return list(range(13, 25))
    
    return result
