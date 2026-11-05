"""Path decomposition via greedy fragment merging with a deadline guard."""
import time
import random
from collections import defaultdict

_DEADLINE = 145.0  # safety margin before the 150s kill

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, deadline):
    """Return a list of paths (vertex lists) that are edge-disjoint and cover all edges."""
    if not edges:
        return []

    edge_set = set(tuple(sorted(e)) for e in edges)
    n = max(max(e) for e in edge_set) + 1

    # Build adjacency
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    # Initialize: each edge is a fragment (path of length 1)
    # fragment: (frozenset of vertices in path, list of vertices in order)
    # We'll track fragments as lists of vertices; the edges used are implied.
    # For efficiency, we maintain:
    #   - a list of current paths (each is a list of vertices)
    #   - for each vertex, which fragment it belongs to (if it's an endpoint)
    #   - endpoint map: vertex -> fragment index (only endpoints are stored)
    
    paths = []  # list of vertex lists
    # endpoint_to_frag: maps a vertex (if it's an endpoint of some path) to the fragment index
    endpoint_to_frag = {}
    
    for u, v in edge_set:
        idx = len(paths)
        paths.append([u, v])
        endpoint_to_frag[u] = idx
        endpoint_to_frag[v] = idx
    
    def _is_endpoint(vertex, frag_idx):
        """Check if vertex is an endpoint of paths[frag_idx]."""
        p = paths[frag_idx]
        return p[0] == vertex or p[-1] == vertex
    
    def _remove_endpoint(vertex, frag_idx):
        """Remove vertex from endpoint_to_frag if it's an endpoint of frag_idx."""
        if vertex in endpoint_to_frag and endpoint_to_frag[vertex] == frag_idx:
            del endpoint_to_frag[vertex]
    
    def _add_endpoint(vertex, frag_idx):
        """Add vertex to endpoint_to_frag as an endpoint of frag_idx."""
        endpoint_to_frag[vertex] = frag_idx
    
    def _try_merge(frag_a, frag_b):
        """
        Try to merge paths[frag_a] and paths[frag_b] by connecting an endpoint of one
        to an endpoint of the other. Returns True if merged, False otherwise.
        After merge, frag_b is removed (its index becomes invalid).
        """
        pa = paths[frag_a]
        pb = paths[frag_b]
        
        # Endpoints of pa: pa[0], pa[-1]
        # Endpoints of pb: pb[0], pb[-1]
        # We can connect if there's an edge between an endpoint of pa and an endpoint of pb,
        # and the resulting path is simple (no repeated vertices).
        
        candidates = [
            (pa[0], pb[0]),
            (pa[0], pb[-1]),
            (pa[-1], pb[0]),
            (pa[-1], pb[-1]),
        ]
        
        for ua, ub in candidates:
            if (min(ua, ub), max(ua, ub)) in edge_set:
                # Check simplicity
                # The merged path would be pa + pb (reversed if needed) or similar
                # If connecting pa[-1] to pb[0]: merged = pa + pb[1:]
                # If connecting pa[-1] to pb[-1]: merged = pa + reversed(pb)[1:]
                # If connecting pa[0] to pb[0]: merged = reversed(pa)[1:] + pb
                # If connecting pa[0] to pb[-1]: merged = reversed(pa)[1:] + reversed(pb)
                
                if ua == pa[-1] and ub == pb[0]:
                    merged = pa + pb[1:]
                elif ua == pa[-1] and ub == pb[-1]:
                    merged = pa + list(reversed(pb))[1:]
                elif ua == pa[0] and ub == pb[0]:
                    merged = list(reversed(pa))[1:] + pb
                else:  # ua == pa[0] and ub == pb[-1]
                    merged = list(reversed(pa))[1:] + list(reversed(pb))
                
                # Check simplicity
                if len(merged) == len(set(merged)):
                    # Perform merge
                    # Remove endpoints that are no longer endpoints
                    _remove_endpoint(ua, frag_a)
                    _remove_endpoint(ub, frag_b)
                    
                    # The new endpoints are:
                    # If we connected pa[-1] to pb[0]: new endpoints are pa[0] and pb[-1]
                    # If we connected pa[-1] to pb[-1]: new endpoints are pa[0] and pb[0]
                    # If we connected pa[0] to pb[0]: new endpoints are pa[-1] and pb[-1]
                    # If we connected pa[0] to pb[-1]: new endpoints are pa[-1] and pb[0]
                    
                    if ua == pa[-1] and ub == pb[0]:
                        new_ep_a = pa[0]
                        new_ep_b = pb[-1]
                    elif ua == pa[-1] and ub == pb[-1]:
                        new_ep_a = pa[0]
                        new_ep_b = pb[0]
                    elif ua == pa[0] and ub == pb[0]:
                        new_ep_a = pa[-1]
                        new_ep_b = pb[-1]
                    else:
                        new_ep_a = pa[-1]
                        new_ep_b = pb[0]
                    
                    paths[frag_a] = merged
                    _add_endpoint(new_ep_a, frag_a)
                    _add_endpoint(new_ep_b, frag_a)
                    
                    # Mark frag_b as removed
                    paths[frag_b] = None
                    return True
        return False
    
    # Iteratively try to merge fragments
    # We maintain a list of active fragment indices
    active = list(range(len(paths)))
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Build a list of active fragments sorted by some heuristic
        # Try all pairs
        i = 0
        while i < len(active) and time.time() < deadline - 5:
            j = i + 1
            while j < len(active) and time.time() < deadline - 5:
                fi = active[i]
                fj = active[j]
                if paths[fi] is None or paths[fj] is None:
                    j += 1
                    continue
                if _try_merge(fi, fj):
                    # fj is now removed
                    active[j] = active[-1]
                    active.pop()
                    changed = True
                    # Don't increment j, check new element at j
                    continue
                j += 1
            i += 1
    
    # Collect remaining paths
    result = [p for p in paths if p is not None]
    return result

def construct():
    start = time.time()
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        if time.time() > _DEADLINE - 5:
            # Fallback: return trivial decomposition for remaining
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose(edges, _DEADLINE)
        results.append(result)
    return results
