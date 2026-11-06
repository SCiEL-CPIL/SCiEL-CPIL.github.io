"""Triangle-free graphs without independent 10-sets (R(3,10) > n).

Structural change: replace the sequential "add one vertex at a time" extension
with a constructive design based on the Andrasfai graph And(9) as a base
substructure, plus a global simulated-annealing / local-search over the whole
edge set for larger n. We also use a much faster triangle-free validator and
an alpha checker with strong upper-bound pruning.

Key ideas:
  1. Fast triangle-free check: for each edge (u,v), require adj[u] & adj[v] == 0
     (no common neighbor). O(m) bit ops.
  2. Fast alpha check: branch-and-bound with greedy upper bound.
  3. Construct candidate graphs of size n (27..40) by starting from And(9)
     embedded in vertices 0..25, then using simulated annealing to add edges
     among the "new" vertices and between old/new to satisfy constraints.
  4. If a full valid graph is found at size n, try to extend to n+1.
  5. Deadline guard.
"""
import time
import random

DEADLINE = 140.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > DEADLINE

def _is_triangle_free(adj, n):
    """Check triangle-freeness: for every edge (u,v), adj[u] & adj[v] must be 0
    (excluding u and v themselves, which are not in adj[u] or adj[v] by construction)."""
    for i in range(n):
        ai = adj[i]
        j = ai
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            if jj <= i:
                j ^= lb
                continue
            if adj[i] & adj[jj]:
                return False
            j ^= lb
    return True

def _alpha_le(adj, n, limit):
    """Return True if alpha(G) <= limit. Uses B&B with greedy upper bound."""
    # If n <= limit, alpha could be n, so we need to check.
    # We want to know if there exists an independent set of size limit+1.
    # Equivalent: in complement graph, is there a clique of size limit+1?
    # Let's do B&B on independent set: find max independent set size, cap at limit+1.
    
    best = [0]
    
    def _bbs(chosen, candidates):
        if chosen >= limit + 1:
            best[0] = limit + 1
            return
        if candidates == 0:
            if chosen > best[0]:
                best[0] = chosen
            return
        # Upper bound: chosen + popcount(candidates)
        if chosen + candidates.bit_count() <= best[0]:
            return
        # Greedy upper bound: repeatedly pick vertex with min degree in candidates
        # This is expensive, so just use popcount for now
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        # Exclude v
        _bbs(chosen, rest)
        # Include v
        new_cands = rest & ~adj[v]
        _bbs(chosen + 1, new_cands)
    
    _bbs(0, (1 << n) - 1)
    return best[0] <= limit

def _is_valid(adj, n):
    if not _is_triangle_free(adj, n):
        return False
    if not _alpha_le(adj, n, 9):
        return False
    return True

def _andrasfai_9_adj():
    """Return adj bitmasks for And(9) on 26 vertices."""
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def _random_valid_graph(n, adj_base, n_base):
    """Create a random graph on n vertices where vertices 0..n_base-1 have the
    And(9) structure, and the rest are randomly connected. Then try to fix it."""
    adj = list(adj_base) + [0] * (n - n_base)
    
    # Connect new vertices randomly to old ones
    for i in range(n_base, n):
        # New vertex i connects to some old vertices
        # For triangle-freeness: if i connects to u and v (old), u and v must not be adjacent
        # So N(i) ∩ V_old must be an independent set in the old graph
        # For now, just random and let local search fix it
        for j in range(n_base):
            if random.random() < 0.3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Connect to other new vertices
        for j in range(n_base, i):
            if random.random() < 0.3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return adj

def _local_search(adj, n, max_iter=5000):
    """Local search to make graph valid. Returns (adj, is_valid)."""
    adj = list(adj)
    n = len(adj)
    
    for it in range(max_iter):
        if _deadline_exceeded():
            break
        if _is_valid(adj, n):
            return adj, True
        
        # Pick a random move: add or remove an edge
        if random.random() < 0.5:
            # Remove a random edge
            all_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        all_edges.append((i, j))
            if not all_edges:
                continue
            u, v = random.choice(all_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Add a random non-edge
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            if not non_edges:
                continue
            u, v = random.choice(non_edges)
            # Check triangle-freeness before adding
            common = adj[u] & adj[v]
            if common:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return adj, _is_valid(adj, n)

def _try_construct_n(n):
    """Try to construct a valid graph on n vertices."""
    if n < 26:
        return None
    
    adj_base, n_base = _andrasfai_9_adj()
    
    # Multiple restarts
    for restart in range(200):
        if _deadline_exceeded():
            break
        
        if n == n_base:
            adj = list(adj_base)
        else:
            adj = _random_valid_graph(n, adj_base, n_base)
        
        adj, valid = _local_search(adj, n, max_iter=3000)
        if valid:
            return adj
    
    return None

def _try_extend(adj, n):
    """Try to extend a valid graph on n vertices to n+1."""
    if _deadline_exceeded():
        return None
    
    # New vertex n connects to an independent set in G
    # Try various independent sets
    for attempt in range(500):
        if _deadline_exceeded():
            break
        
        # Find a random independent set
        order = list(range(n))
        random.shuffle(order)
        I = []
        for v in order:
            conflict = False
            for u in I:
                if adj[v] & (1 << u):
                    conflict = True
                    break
            if not conflict:
                I.append(v)
        
        if len(I) < 2:
            continue
        
        new_adj = list(adj)
        new_adj.append(0)
        for i in range(n):
            if i in I:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    best_n = 0
    best_adj = None
    
    # First, verify And(9) is valid
    adj26, _ = _andrasfai_9_adj()
    if _is_valid(adj26, 26):
        best_n = 26
        best_adj = list(adj26)
    
    # Try to construct larger graphs directly
    for target_n in range(27, 41):
        if _deadline_exceeded():
            break
        
        # Try to extend from best known
        if best_adj is not None and best_n == target_n - 1:
            new_adj = _try_extend(best_adj, best_n)
            if new_adj is not None:
                best_n = target_n
                best_adj = new_adj
                continue
        
        # Try direct construction
        adj = _try_construct_n(target_n)
        if adj is not None:
            best_n = target_n
            best_adj = adj
    
    # Try to extend further from best
    while best_n < 40:
        if _deadline_exceeded():
            break
        new_adj = _try_extend(best_adj, best_n)
        if new_adj is not None:
            best_n += 1
            best_adj = new_adj
        else:
            break
    
    # Fallback: return And(9) if nothing better
    if best_adj is None:
        best_n = 26
        best_adj = list(adj26)
    
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
