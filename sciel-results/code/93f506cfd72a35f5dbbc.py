<<<<<<< SEARCH
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
=======
"""Find maximum reciprocal-subsum-free subset of [1,24]."""
import time
import random
from fractions import Fraction
from itertools import combinations

def _deadline_hit(start_time, limit=85.0):
    return (time.time() - start_time) > limit

def _can_add(S, x):
    """Check if adding x to set S keeps the set valid.
    S is a set of already-selected elements, x is the candidate.
    We need to verify:
    1. 1/x is not the sum of reciprocals of any subset of S
    2. For each a in S, 1/a is not the sum of reciprocals of any subset of (S - {a}) ∪ {x}
       This reduces to: 1/a is not the sum of reciprocals of any subset of S - {a} that includes x,
       i.e., 1/a - 1/x is not the sum of reciprocals of any subset of S - {a}.
    """
    S = set(S)
    # Check 1: 1/x not sum of reciprocals of subset of S
    for r in range(1, len(S) + 1):
        for subset in combinations(sorted(S), r):
            if sum(Fraction(1, v) for v in subset) == Fraction(1, x):
                return False
    
    # Check 2: for each a in S, 1/a not sum of reciprocals of subset of (S-{a}) ∪ {x}
    for a in S:
        others = S - {a}
        # Subsets of (others ∪ {x}) that sum to 1/a
        # Either don't include x: already checked (S is valid)
        # Or include x: need 1/a - 1/x = sum of reciprocals of subset of others
        target = Fraction(1, a) - Fraction(1, x)
        if target > 0:
            for r in range(0, len(others) + 1):
                for subset in combinations(sorted(others), r):
                    if sum(Fraction(1, v) for v in subset) == target:
                        return False
    
    return True

def _is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    S = set(S)
    for a in S:
        others = S - {a}
        for r in range(1, len(others) + 1):
            for subset in combinations(sorted(others), r):
                if sum(Fraction(1, v) for v in subset) == Fraction(1, a):
                    return False
    return True

def _greedy_construct(order, start_time):
    """Greedy construction: add elements in given order if valid."""
    S = set()
    for x in order:
        if _deadline_hit(start_time):
            break
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, start_time, max_iters=500):
    """2-opt local search: try swapping one element out for another to increase size.
    Also try adding elements that were previously rejected.
    """
    S = set(S)
    universe = set(range(1, 25))
    best_S = set(S)
    best_size = len(S)
    
    for iteration in range(max_iters):
        if _deadline_hit(start_time):
            break
        
        improved = False
        
        # Try adding any element not in S
        for x in sorted(universe - S, reverse=True):
            if _deadline_hit(start_time):
                break
            if _can_add(S, x):
                S.add(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                break  # restart after one addition
        
        if not improved:
            # Try 1-1 swaps: remove one, add another
            found_swap = False
            for remove_x in sorted(S):
                if _deadline_hit(start_time):
                    break
                S_new = S - {remove_x}
                for add_x in sorted(universe - S_new, reverse=True):
                    if _deadline_hit(start_time):
                        break
                    if add_x == remove_x:
                        continue
                    if _can_add(S_new, add_x):
                        S = S_new | {add_x}
                        found_swap = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                if found_swap:
                    break
        
        if not found_swap and not improved:
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline_limit = 85.0  # safety margin before 90s kill
    
    # Strategy: greedy from largest to smallest (large elements are "safer" as targets
    # because 1/large is small and hard to sum from other reciprocals), then local search.
    
    # Primary approach: greedy largest-first
    order_largest_first = list(range(24, 0, -1))
    S = _greedy_construct(order_largest_first, start_time)
    
    # Local search to improve
    S = _local_search(S, start_time)
    
    # Try additional strategies if time permits
    if not _deadline_hit(start_time, deadline_limit - 5.0):
        # Greedy smallest-first
        order_smallest_first = list(range(1, 25))
        S2 = _greedy_construct(order_smallest_first, start_time)
        S2 = _local_search(S2, start_time)
        if len(S2) > len(S):
            S = S2
    
    if not _deadline_hit(start_time, deadline_limit - 5.0):
        # Randomized restarts
        random.seed(42)
        for _ in range(20):
            if _deadline_hit(start_time, deadline_limit - 5.0):
                break
            order = list(range(1, 25))
            random.shuffle(order)
            S3 = _greedy_construct(order, start_time)
            S3 = _local_search(S3, start_time, max_iters=100)
            if len(S3) > len(S):
                S = S3
    
    result = sorted(S)
    
    # Final validation
    if not _is_valid(set(result)):
        # Fallback to a known valid set
        result = list(range(13, 25))
    
    if not result:
        result = [12]
    
    return result
>>>>>>> REPLACE
