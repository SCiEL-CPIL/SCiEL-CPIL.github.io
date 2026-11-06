"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict, deque

def _build_instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i + 1]
        edges.add((min(u, v), max(u, v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return sorted(edges)


def _path_decomposition_flow(edges, n):
    """
    Decompose edges into minimum number of edge-disjoint paths using min-cost flow.
    
    The key insight: we want to cover all edges with minimum number of paths.
    This is equivalent to finding a minimum path cover in the line graph sense,
    but more directly: we can model this as finding a minimum-cost flow where
    each edge is used exactly once, and we minimize the number of path starts.
    
    Actually, a cleaner formulation: 
    - Each edge (u,v) must be traversed exactly once.
    - We want to pair up edges at vertices to form longer paths.
    - At each vertex, edges incident to it can be paired: if two edges share a vertex,
      we can chain them (one ends at vertex, next starts at vertex).
    - The number of paths = (total edges) - (number of successful pairings).
    - So maximizing pairings minimizes paths.
    
    This is a maximum matching problem on a bipartite graph:
    - Left side: "tail" copies of each edge (the starting point)
    - Right side: "head" copies of each edge (the ending point)
    - An edge from tail(e1) to head(e2) exists if e1 and e2 share a vertex,
      and e1 ends at that shared vertex while e2 starts at it.
    
    But since edges are undirected, each edge (u,v) can be oriented either way.
    This makes it more complex. Let's use a different approach.
    
    Alternative: Use the fact that in any path decomposition, the number of paths
    is at least max(1, ceil(odd_degree_vertices / 2)) where odd_degree_vertices
    are vertices with odd degree. This is a lower bound from Gallai's conjecture context.
    
    For an exact algorithm, let's use min-cost max-flow on a network:
    - Create a node for each vertex: v_in and v_out for each v.
    - For each edge (u, v) in the graph, add two directed arcs: u_out -> v_in and v_out -> u_in, each with capacity 1 and cost 0.
    - Add an arc v_in -> v_out for each vertex v with capacity infinity and cost 0.
    - Add a super source S and super sink T.
    - For each vertex v, add S -> v_out with capacity 1 and cost 1 (starting a path at v).
    - For each vertex v, add v_in -> T with capacity 1 and cost 1 (ending a path at v).
    
    Wait, this doesn't quite work because we need to use every edge exactly once.
    
    Let me reconsider. The standard reduction for minimum path cover in a directed graph
    uses bipartite matching. For undirected graphs, we can orient edges.
    
    Simpler approach: Use a greedy + local improvement heuristic with beam search,
    which has been shown to perform well.
    """
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set(edges)
    
    # Greedy: repeatedly find a long path using beam search, remove its edges, repeat
    paths = []
    remaining_edges = set(edges)
    remaining_adj = defaultdict(set)
    for u, v in edges:
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    def find_long_path(start, beam_width=50, max_len=100):
        """Beam search for a long simple path starting at 'start'."""
        # State: (path_tuple, set_of_used_edges)
        # We'll use a priority queue sorted by path length (descending)
        best_path = [start]
        used_edges = set()
        
        # Beam: list of (path_list, used_edges_set)
        beam = [([start], set())]
        
        for step in range(max_len):
            next_beam = []
            seen_states = set()
            for path, used in beam:
                last = path[-1]
                for nb in remaining_adj.get(last, []):
                    if nb in path:  # Simple path constraint
                        continue
                    e = (min(last, nb), max(last, nb))
                    if e in used:
                        continue
                    if e not in remaining_edges:
                        continue
                    new_path = path + [nb]
                    new_used = used | {e}
                    state_key = tuple(new_path)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    next_beam.append((new_path, new_used))
            
            if not next_beam:
                break
            
            # Sort by length descending, keep top beam_width
            next_beam.sort(key=lambda x: -len(x[0]))
            beam = next_beam[:beam_width]
            
            if len(beam[0][0]) > len(best_path):
                best_path = beam[0][0][:]
        
        return best_path
    
    # Find paths greedily
    while remaining_edges:
        # Pick a start vertex with highest remaining degree
        deg = {v: len(remaining_adj[v]) for v in remaining_adj if remaining_adj[v]}
        if not deg:
            break
        start = max(deg, key=lambda v: deg[v])
        
        path = find_long_path(start, beam_width=30, max_len=80)
        
        if len(path) < 2:
            # Single vertex, find any incident edge
            for nb in remaining_adj.get(start, []):
                e = (min(start, nb), max(start, nb))
                if e in remaining_edges:
                    path = [start, nb]
                    break
            if len(path) < 2:
                # Shouldn't happen if remaining_edges is non-empty
                break
        
        # Extract edges from path
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                path_edges.append(e)
        
        if not path_edges:
            # Fallback: just take one edge
            e = next(iter(remaining_edges))
            path = [e[0], e[1]]
            path_edges = [e]
        
        paths.append(path)
        
        # Remove used edges
        for e in path_edges:
            remaining_edges.discard(e)
            u, v = e
            if v in remaining_adj.get(u, set()):
                remaining_adj[u].discard(v)
            if u in remaining_adj.get(v, set()):
                remaining_adj[v].discard(u)
    
    return paths


def _path_decomposition_lp(edges, n):
    """
    Use a more principled approach: formulate as integer program solved via
    branch and bound, or use the bipartite matching reduction.
    
    For minimum path decomposition of an undirected graph:
    - Orient each edge arbitrarily to get a directed graph.
    - The minimum number of paths to cover all edges equals 
      (number of edges) - (maximum number of edge pairings at vertices).
    - Edge pairings at a vertex v: if two edges e1=(u,v) and e2=(v,w) are paired,
      they form a path u-v-w. The number of such pairings at v is at most floor(deg(v)/2).
    - Total paths = E - sum_v (pairings_at_v).
    - To minimize paths, maximize total pairings.
    
    This can be formulated as a maximum b-matching or flow problem.
    """
    # Use the flow-based approach
    # For each vertex v, we can pair up to floor(deg(v)/2) pairs of edges.
    # This is equivalent to: at each vertex, we choose which edges are "connected" 
    # (forming a path through v) vs "endpoints" (path starts/ends at v).
    
    # The minimum number of paths is:
    # sum over vertices of max(0, deg(v) - 2 * floor(deg(v)/2)) ... no that's just odd vertices
    # Actually: #paths = (number of odd-degree vertices) / 2 + (number of components that are ... )
    # For a connected graph: min paths = max(1, odd_deg_count // 2) if we can achieve it.
    
    # Let's just use the greedy beam search which works well in practice.
    return _path_decomposition_flow(edges, n)


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    results = []
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: single-edge paths
            edges = _build_instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _build_instance(n, p, seed)
        paths = _path_decomposition_flow(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
