"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _spectral_radius_signed_fast(graph, signs):
    """Compute spectral radius of signed adjacency matrix efficiently."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = float(signs[i])
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return max(abs(eigvals[0]), abs(eigvals[-1]))

def _is_valid_lift_fast(graph, signs):
    """Check if the lift is valid (connected, 3-regular) using BFS."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    lift_n = 2 * n
    adj = [[] for _ in range(lift_n)]
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    for v in range(lift_n):
        if len(adj[v]) != 3:
            return False
    
    visited = [False] * lift_n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == lift_n

def _get_cycle_structure(graph):
    """Extract cycle edge indices and matching edge indices."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(i)
        elif u >= m and v >= m:
            top_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    return bottom_cycle_edges, top_cycle_edges, matching_edges

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    bottom_cycle_edges, top_cycle_edges, matching_edges = _get_cycle_structure(graph)
    
    best_signs = None
    best_radius = np.inf
    
    def try_signs(s):
        nonlocal best_signs, best_radius
        if time.time() > deadline:
            return
        if not _is_valid_lift_fast(graph, s):
            return
        r = _spectral_radius_signed_fast(graph, s)
        if r < best_radius:
            best_radius = r
            best_signs = s.copy()
    
    # KEY INSIGHT: The signed spectrum is invariant under vertex sign flips (DAD similarity).
    # Two signings are equivalent if they differ by a switching vector.
    # For a prism graph, the switching classes are determined by:
    #   - product of signs around bottom cycle (frustration_b)
    #   - product of signs around top cycle (frustration_t)  
    #   - product of signs around the "ladder" cycles (one per matching edge)
    # 
    # Actually, the number of distinct switching classes for a graph with E edges
    # and cyclomatic number mu = E - V + 1 is 2^mu.
    # For prism: V = 2m, E = 3m, mu = 3m - 2m + 1 = m + 1.
    # So there are 2^(m+1) switching classes.
    #
    # We can enumerate representatives of each switching class by fixing
    # a spanning tree and varying signs on the non-tree edges.
    #
    # For the prism graph, a natural spanning tree: all bottom cycle edges except one,
    # all matching edges, and top cycle edges except one.
    # Non-tree edges: 2 edges (one from each cycle).
    # Wait, mu = m+1, so we need m+1 non-tree edges to vary.
    #
    # Better approach: enumerate all 2^(m+1) switching class representatives directly.
    # A switching class representative can be parameterized by which edges in a
    # fundamental cycle basis are negative.
    
    # For small m (<=7), 2^(m+1) <= 256, so we can enumerate all switching classes.
    # For larger m, we use a smarter approach.
    
    if m <= 7:
        # Enumerate all 2^(m+1) switching classes.
        # Fix a spanning tree: use all matching edges + all bottom cycle edges 
        # except the last one + all top cycle edges except the last one.
        # Non-tree edges: bottom_cycle_edges[-1] and top_cycle_edges[-1].
        # But mu = m+1, so we need m+1 independent cycles.
        
        # Actually, let's think differently. The switching classes correspond to
        # elements of H_1(G, Z_2). For a prism graph, we can parameterize by:
        # - which subset of "rungs" have odd number of -1s in the corresponding ladder cycle
        # 
        # Simpler: just enumerate all 2^(m+1) patterns by fixing a canonical form.
        # Fix vertex 0 to have sign +1 (gauge fixing). Then the switching class
        # is determined by the signs relative to vertex 0.
        #
        # Even simpler: for a graph with cyclomatic number mu, there are 2^mu switching classes.
        # We can enumerate them by choosing which of the mu independent cycles have frustration.
        
        # Let's use a direct enumeration: fix the first m+1 edges to be +1 (gauge),
        # and vary the remaining... no, that's not quite right.
        
        # Cleanest approach: enumerate all 2^(m+1) switching classes by
        # varying the signs on a set of m+1 independent edges, with the rest
        # determined to give a canonical representative.
        
        # For the prism, pick a spanning tree T. The non-tree edges form a
        # fundamental cycle basis. There are mu = m+1 non-tree edges.
        # A switching class representative has +1 on all tree edges and
        # arbitrary signs on non-tree edges.
        
        # Spanning tree for prism: 
        # - All m matching edges
        # - Bottom cycle: edges (0,1), (1,2), ..., (m-2, m-1) [m-1 edges]
        # - Top cycle: edges (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1) [m-1 edges]
        # Total tree edges: m + (m-1) + (m-1) = 3m - 2
        # Non-tree edges: 3m - (3m-2) = 2... that's not m+1.
        
        # Wait, I made an error. Let me recount.
        # Prism graph: 2m vertices, 3m edges.
        # Spanning tree has 2m - 1 edges.
        # Non-tree edges: 3m - (2m-1) = m + 1. Correct.
        
        # Spanning tree:
        # - All m matching edges: (i, i+m) for i in 0..m-1
        # - Bottom path: (0,1), (1,2), ..., (m-2,m-1): m-1 edges
        # - Top path: (m, m+1), ..., (2m-2, 2m-1): m-1 edges
        # Total: m + (m-1) + (m-1) = 3m - 2. But we need 2m - 1.
        # 3m - 2 = 2m - 1 => m = 1. That's wrong for m > 1.
        
        # I see the issue: the prism is not a tree plus a few edges in that way.
        # Let me just directly enumerate 2^(m+1) patterns.
        
        # Alternative clean approach: the switching class is determined by the
        # product of signs around each face. For the prism, there are m+1 faces:
        # m rectangular faces and 2 cycle faces... no, that's not right either.
        
        # Let me just do a brute force over all 2^(m+1) switching classes by
        # fixing a gauge and enumerating.
        
        # Gauge fixing: set signs on a spanning tree to +1.
        # The non-tree edges (m+1 of them) can be independently +1 or -1.
        
        # Build spanning tree:
        # - All matching edges (m edges)
        # - Bottom cycle edges except the last: (m-1) edges  
        # - Top cycle edges except the last: (m-1) edges
        # Total: 3m - 2. But spanning tree should have 2m-1 edges.
        # 3m-2 vs 2m-1: difference is m-1. So I'm overcounting.
        
        # The issue: the prism graph has 2m vertices. A spanning tree has 2m-1 edges.
        # I listed m + (m-1) + (m-1) = 3m-2 edges, which is too many for m>1.
        # This means my "spanning tree" has cycles!
        
        # Let me construct a proper spanning tree:
        # - Matching edges: (0,m), (1,m+1), ..., (m-1, 2m-1): m edges
        # - Bottom path: (0,1), (1,2), ..., (m-2,m-1): m-1 edges
        # - Top path: (m, m+1), ..., (2m-2, 2m-1): m-1 edges
        # This gives m + (m-1) + (m-1) = 3m-2 edges connecting 2m vertices.
        # But 3m-2 > 2m-1 for m>1, so this has cycles.
        
        # The problem: both bottom and top paths create cycles when combined with matchings.
        # E.g., (0,1)-(1,m+1)-(m,m+1)-(m,0)-(0,1) is a 4-cycle.
        
        # Proper spanning tree:
        # - Matching edges: all m of them
        # - Bottom path: (0,1), ..., (m-2,m-1): m-1 edges
        # - Top: only connect via matchings, no top cycle edges needed!
        # Total: m + (m-1) = 2m-1. Perfect!
        
        # So spanning tree = all matching edges + bottom cycle path (all but last edge).
        # Non-tree edges: top cycle edges (m of them) + bottom last edge (1) = m+1.
        
        # Enumerate all 2^(m+1) assignments to these non-tree edges.
        non_tree_edges = top_cycle_edges + [bottom_cycle_edges[-1]]
        # top_cycle_edges has m edges, bottom_cycle_edges[-1] is 1 edge, total m+1.
        
        for mask in range(2 ** (m + 1)):
            if time.time() > deadline:
                break
            s = np.ones(num_edges, dtype=int)
            # Set non-tree edges based on mask
            for bit, edge_idx in enumerate(non_tree_edges):
                if mask & (1 << bit):
                    s[edge_idx] = -1
            # Tree edges are all +1
            try_signs(s)
    
    else:
        # For m=9, 2^10 = 1024 switching classes, still very manageable!
        # Actually, let's just do the same enumeration for all m.
        non_tree_edges = top_cycle_edges + [bottom_cycle_edges[-1]]
        
        for mask in range(2 ** (m + 1)):
            if time.time() > deadline:
                break
            s = np.ones(num_edges, dtype=int)
            for bit, edge_idx in enumerate(non_tree_edges):
                if mask & (1 << bit):
                    s[edge_idx] = -1
            try_signs(s)
    
    # If no valid signs found (shouldn't happen), fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
        if _is_valid_lift_fast(graph, best_signs):
            best_radius = _spectral_radius_signed_fast(graph, best_signs)
    
    # Local search: flip one edge at a time to potentially find better within same class
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    max_local_iterations = 50
    for _ in range(max_local_iterations):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if _is_valid_lift_fast(graph, trial):
                r = _spectral_radius_signed_fast(graph, trial)
                if r < current_radius - 1e-10:
                    current_signs = trial
                    current_radius = r
                    improved = True
                    best_radius = r
                    best_signs = trial.copy()
                    break
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
