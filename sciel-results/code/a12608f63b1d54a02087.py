"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _paley_37():
    """Return adjacency bitsets for Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k using bit-parallel method."""
    if k <= 1:
        return True
    if k == 2:
        return any(adj[i] != 0 for i in range(n))
    # For k >= 3, use recursive bit-parallel search
    def _search(candidates, depth):
        if depth == 0:
            return True
        while candidates:
            v = (candidates & -candidates).bit_length() - 1
            candidates &= candidates - 1
            if _search(candidates & adj[v], depth - 1):
                return True
        return False
    return _search((1 << n) - 1, k)

def _has_k_independent(adj, n, k):
    """Check if graph has an independent set of size k."""
    # Independent set in G = clique in complement of G
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = ((1 << n) - 1) & ~(adj[i] | (1 << i))
    return _has_k_clique(comp_adj, n, k)

def _is_valid_55_free(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent 5-set."""
    return not _has_k_clique(adj, n, 5) and not _has_k_independent(adj, n, 5)

def _try_add_vertex(adj, n, k_clique=5, k_indep=5):
    """Try to add a new vertex connected to a subset of existing vertices.
    Returns new adj list and new n if successful, else None."""
    # Try different connection patterns
    for attempt in range(20):
        # Random subset of existing vertices to connect to
        if attempt == 0:
            # Connect to no vertices (isolated)
            new_adj = adj + [0]
            new_n = n + 1
            # Check: no new K5 (impossible since vertex isolated), 
            # but might create independent 5-set if there's an independent 4-set
            if _has_k_independent(new_adj, new_n, 5):
                continue
            return new_adj, new_n
        elif attempt == 1:
            # Connect to all vertices
            new_adj = adj + [((1 << n) - 1)]
            for i in range(n):
                new_adj[i] |= (1 << n)
            new_n = n + 1
            # Check: no new K5 (if there's a K4, adding universal vertex makes K5)
            if _has_k_clique(new_adj, new_n, 5):
                continue
            return new_adj, new_n
        else:
            # Random subset
            mask = 0
            for i in range(n):
                if random.random() < 0.5:
                    mask |= (1 << i)
            new_adj = adj + [mask]
            for i in range(n):
                if mask & (1 << i):
                    new_adj[i] |= (1 << n)
            new_n = n + 1
            if not _is_valid_55_free(new_adj, new_n):
                continue
            return new_adj, new_n
    return None

def _try_remove_edge_and_add(adj, n, k_clique=5, k_indep=5):
    """Try removing an edge and adding a vertex."""
    # This is complex; skip for now and just try simple additions
    return None

def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Start with Paley(37)
    n, adj = _paley_37()
    best_n = n
    best_adj = adj[:]
    
    # Try to add vertices
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        result = _try_add_vertex(adj, n)
        if result is not None:
            adj, n = result
            best_n = n
            best_adj = adj[:]
        else:
            # Try a few more random attempts before giving up
            for _ in range(10):
                if time.time() >= deadline:
                    break
                result = _try_add_vertex(adj, n)
                if result is not None:
                    adj, n = result
                    best_n = n
                    best_adj = adj[:]
                    break
            else:
                # Give up if we can't add any more
                break
    
    # Convert adj bitsets back to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return best_n, edges
# EVOLVE-BLOCK-END
