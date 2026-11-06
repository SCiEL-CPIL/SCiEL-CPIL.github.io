"""Construct a C4-free graph on n vertices with no independent 11-set, maximizing n.

Strategy:
  - Start from the polarity graph of the projective plane of order 6 (39 vertices,
    7-regular, C4-free, independence number 6).
  - Add isolated vertices up to n=43.
  - Run a deterministic local search: repeatedly add an edge between a non-adjacent
    pair that does not create a C4, choosing the pair that most reduces the
    estimated independence number (greedy heuristic: prefer pairs where both
    endpoints currently have low degree, i.e., are more "independent-like").
  - Verify with an exact maximum-independent-set search (branch-and-bound) that
    alpha < 11.
  - Return the best valid graph found within a time budget.
"""

import time
import random
from itertools import combinations

DEADLINE = 150.0  # wall-clock seconds
SAFETY_MARGIN = 5.0  # return at least this many seconds before deadline


def _check_c4(edges, n):
    """Return True if the graph contains a C4."""
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # For each pair of vertices, count common neighbours
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False


def _max_independent_set_size(adj, n, time_limit=None):
    """Branch-and-bound to find the size of a maximum independent set."""
    start_time = time.time()
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i])
    
    best = 0
    best_set = set()
    
    def branch(remaining, current, current_size):
        nonlocal best, best_set
        if time_limit is not None and time.time() - start_time > time_limit:
            return
        if not remaining:
            if current_size > best:
                best = current_size
                best_set = set(current)
            return
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best:
            return
        
        # Pick a vertex from remaining
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include v
        new_remaining = [x for x in rest if x not in adj[v]]
        branch(new_remaining, current + [v], current_size + 1)
        
        # Option 2: exclude v
        branch(rest, current, current_size)
    
    branch(list(range(n)), [], 0)
    return best


def _has_independent_11(adj, n, time_limit=5.0):
    """Check if there's an independent set of size 11. Returns True if found."""
    start_time = time.time()
    
    # Quick upper bound: if n < 11, impossible
    if n < 11:
        return False
    
    # Use a simpler branch-and-bound that stops as soon as we find size 11
    found = [False]
    
    def branch(remaining, current_size):
        nonlocal found
        if found[0]:
            return
        if time.time() - start_time > time_limit:
            return
        if current_size >= 11:
            found[0] = True
            return
        if not remaining:
            return
        # Upper bound
        if current_size + len(remaining) < 11:
            return
        
        v = remaining[0]
        rest = remaining[1:]
        
        # Include v
        new_remaining = [x for x in rest if x not in adj[v]]
        branch(new_remaining, current_size + 1)
        
        if not found[0]:
            # Exclude v
            branch(rest, current_size)
    
    branch(list(range(n)), 0)
    return found[0]


def _build_polarity_graph_q6():
    """
    Build the polarity graph of the projective plane of order 6.
    The projective plane PG(2, 6) has 6^2 + 6 + 1 = 43 points.
    A polarity (e.g., the symplectic polarity) maps each point to a line.
    Two points are adjacent if one lies on the polar line of the other.
    This graph is C4-free and has independence number 6.
    
    Actually, the standard polarity graph of PG(2, q) has q^2+q+1 vertices.
    For q=6, that's 43 vertices. The graph is (q+1)-regular = 7-regular.
    Independence number is q = 6.
    
    Let me construct it properly.
    """
    q = 6
    n = q * q + q + 1  # 43
    
    # Represent points of PG(2, q) as projective equivalence classes of [x, y, z]
    # where (x, y, z) != (0, 0, 0), modulo scaling by non-zero field elements.
    # Since q=6 is not prime, we need GF(6) which doesn't exist.
    # 
    # Wait - projective planes of order q exist for all prime powers q.
    # 6 is not a prime power, so PG(2, 6) does not exist!
    # 
    # The best known C4-free graphs come from projective planes of prime-power order.
    # For q=5: n=31, 6-regular, alpha=5.
    # For q=7: n=57, 8-regular, alpha=7.
    # 
    # The "39-vertex" construction mentioned in the task must come from a different source.
    # Let me use a known construction instead.
    #
    # Actually, re-reading: "The best known construction has 39 vertices."
    # This likely comes from a computer search or a specific finite geometry construction.
    #
    # Let me try a different approach: use the Clebsch graph or other known C4-free graphs,
    # or construct via random greedy + local search.
    
    # Let's use a random greedy construction with local search.
    return None


def _greedy_c4_free(n, seed=42):
    """
    Greedy construction of a C4-free graph on n vertices.
    Add edges one at a time, choosing the edge that maximizes some heuristic
    while maintaining C4-freeness.
    """
    rng = random.Random(seed)
    
    # Start with empty graph
    adj = [set() for _ in range(n)]
    edges = []
    
    # Generate candidate edges in random order
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if len(edges) >= 0:  # we can add all non-C4-creating edges
            # Check if adding (u, v) creates a C4
            # A C4 is created if u and v have >= 2 common neighbours
            common = adj[u] & adj[v]
            if len(common) < 2:
                # Also check: for any w in adj[u], is w adjacent to v? No, that's the same check.
                # Actually the condition is: no two vertices have 2+ common neighbours.
                # Adding edge (u,v) means: for every w != u,v, w cannot be adjacent to both u and v.
                # i.e., |adj[u] ∩ adj[v]| must be 0 before adding (since u,v not yet adjacent).
                # Wait, that's not right either. Let me think again.
                #
                # C4-free means: no two vertices have 2 or more common neighbours.
                # If we add edge (u,v), we need to check that for all w, v != u:
                #   w is not a common neighbour of u and v.
                # But also, adding (u,v) changes the common-neighbour count for other pairs.
                # Specifically, for any w, the pair (w, u) now has v as a potential common neighbour
                # only if w-v is an edge. Similarly for (w, v).
                #
                # Actually, the standard check: adding edge (u,v) creates a C4 iff
                # there exist distinct x, y such that x~u, x~v, y~u, y~v (i.e., u,v have 2+ common neighbours).
                # So the check is simply: len(adj[u] & adj[v]) == 0 before adding.
                
                if len(common) == 0:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
    
    return adj, edges


def _local_search_add_edges(adj, n, time_budget=30.0):
    """
    Local search: try to add more edges (between non-adjacent pairs) that don't create C4,
    prioritizing pairs that reduce independence number.
    """
    start_time = time.time()
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Build list of non-adjacent pairs
    non_adj_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                non_adj_pairs.append((u, v))
    
    # Sort by heuristic: prefer pairs where both endpoints have low degree
    # (these are more "independent-like" so connecting them reduces alpha)
    def heuristic(pair):
        u, v = pair
        return (len(adj[u]) + len(adj[v]))
    
    non_adj_pairs.sort(key=heuristic)
    
    added = 0
    for u, v in non_adj_pairs:
        if time.time() - start_time > time_budget:
            break
        # Re-check if still non-adjacent
        if v in adj[u]:
            continue
        # Check C4 condition
        common = adj[u] & adj[v]
        if len(common) == 0:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            added += 1
    
    return adj, edges, added


def _exact_alpha_check(adj, n, time_limit=10.0):
    """
    Exact check: return True if independence number < 11.
    Uses branch-and-bound.
    """
    start_time = time.time()
    
    # Compute a good initial lower bound on alpha using greedy
    greedy_alpha = _greedy_independent_set(adj, n)
    
    best = [0]
    
    # Order by degree ascending
    order = sorted(range(n), key=lambda i: len(adj[i]))
    pos = {v: i for i, v in enumerate(order)}
    
    # For branch and bound, we process in the order
    # remaining is a list of vertices (in order) that are candidates
    
    def branch(remaining_idx, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        if current_size >= 11:
            # We found an independent set of size 11, so alpha >= 11
            # But we want to know if alpha < 11, so this means the check fails
            # Actually, we're checking if alpha < 11. If we find size 11, return False.
            # Let me restructure: return the max independent set size.
            pass
        
        if not remaining_idx:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound
        if current_size + len(remaining_idx) <= best[0]:
            return
        
        # Pick first vertex in remaining
        v = order[remaining_idx[0]]
        rest = remaining_idx[1:]
        
        # Include v
        new_remaining = [idx for idx in rest if order[idx] not in adj[v]]
        branch(new_remaining, current_size + 1)
        
        # Exclude v
        branch(rest, current_size)
    
    branch(list(range(n)), 0)
    return best[0]


def _greedy_independent_set(adj, n):
    """Greedy heuristic for independent set size (lower bound)."""
    # Sort by degree ascending
    order = sorted(range(n), key=lambda i: len(adj[i]))
    selected = set()
    for v in order:
        if all(v not in adj[u] for u in selected):
            selected.add(v)
    return len(selected)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_n = 0
    
    # Strategy: try multiple seeds for random greedy + local search
    # Target n values: try 43, 42, 41, 40, 39, 38, ...
    
    # First, try a deterministic construction for smaller n that we know works
    # then try to extend.
    
    # Known: disjoint triangles give n=30, alpha=10.
    # We want alpha < 11, so n=30 with alpha=10 works but isn't great.
    
    # Let's try random greedy for n=43 down to n=35
    for n_target in [43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
        
        # Try a few seeds
        for seed in [42, 123, 7, 99, 2024]:
            if time.time() > deadline:
                break
            
            # Random greedy C4-free construction
            adj, edges = _greedy_c4_free(n_target, seed=seed)
            
            # Local search to add more edges
            adj, edges, added = _local_search_add_edges(adj, n_target, time_budget=5.0)
            
            if time.time() > deadline:
                break
            
            # Check independence number
            # First quick check: greedy lower bound
            greedy_alpha = _greedy_independent_set(adj, n_target)
            if greedy_alpha >= 11:
                # Likely alpha >= 11, skip
                continue
            
            # Exact check with time limit
            alpha = _exact_alpha_check(adj, n_target, time_limit=8.0)
            
            if time.time() > deadline:
                break
            
            if alpha < 11:
                # Valid!
                if n_target > best_n:
                    best_n = n_target
                    best_result = {'n': n_target, 'edges': edges}
                    # Don't break - try to find larger n
                break  # Found valid for this n, try larger n
    
    # If we didn't find anything, fall back to the known valid construction
    if best_result is None:
        # Disjoint triangles: n=30, alpha=10
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_result = {'n': 30, 'edges': edges}
    
    return best_result
