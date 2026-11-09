import time
import random
import itertools
from collections import deque

def _has_c5(n, adj):
    """Check if graph has a C5 (5-cycle) as a subgraph. Returns True if C5 exists."""
    # For n <= 35, we can use a bitmask approach or optimized enumeration.
    # adj is a list of sets or lists. We'll convert to bitmasks for speed.
    adj_masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_masks[u] = mask

    # Check all 5-subsets
    # C(35, 5) is ~325k. For each, check 12 permutations.
    # This is ~4M operations. In Python, this might be slow if not optimized.
    # Let's try a different approach: enumerate paths of length 4.
    
    # A C5 is a path of length 4 that closes.
    # v0-v1-v2-v3-v4-v0.
    # We can fix v0 and v2. They are at distance 2.
    # v0-v1-v2 implies v1 is a common neighbor of v0 and v2.
    # v2-v3-v4-v0 implies v3, v4 form a path of length 2 from v2 to v0? No, v2-v3-v4-v0 is path length 3.
    
    # Let's stick to the subset enumeration but optimize the inner loop.
    # For each 5-subset {a,b,c,d,e}, check if any permutation forms a cycle.
    # There are 12 distinct cycles.
    # We can check:
    # a-b-c-d-e-a
    # a-b-c-e-d-a
    # a-b-d-c-e-a
    # ...
    
    # Actually, a faster check for C5 in small graphs:
    # For each pair (u, v) at distance 2 (share a neighbor), 
    # check if there is a path of length 3 from u to v not using the shared neighbor?
    # No, that's for induced C5. We want subgraph C5.
    
    # Let's use the bitmask check for the 12 cycles.
    # Precompute the 12 permutations of (1,2,3,4) relative to 0.
    # Cycle: 0-p1-p2-p3-p4-0
    # Permutations of (1,2,3,4) are 24. Each cycle counted twice. 12 unique.
    
    perms = []
    seen = set()
    for p in itertools.permutations([1, 2, 3, 4]):
        # Normalize to avoid duplicates (reverse is same cycle)
        # Just check all 24, it's fast enough for 325k * 24 = 7.8M checks.
        # 7.8M simple bit ops in Python might take ~1-2 seconds.
        # We have 150s. So this is fine.
        pass

    # Actually, let's just iterate over all 5-subsets and check connectivity/cycle.
    # A set of 5 vertices forms a C5 subgraph if the induced subgraph contains a C5.
    # The induced subgraph on 5 vertices has at most 10 edges.
    # It contains a C5 if it has at least 5 edges and is not a tree or forest with <5 edges?
    # No, K5 has C5. K5 minus 1 edge has C5.
    # The only graphs on 5 vertices with NO C5 are those with <5 edges?
    # No, a tree has 4 edges, no cycle.
    # A graph with 5 edges might be a tree + 1 edge (cycle of length <=4) or C5.
    # If a graph on 5 vertices has 5 or more edges, does it necessarily contain a C5?
    # No. K_{2,3} has 5 edges, no odd cycle.
    # So we must check explicitly.
    
    # Efficient check for a specific 5-set:
    # Get the 5x5 adjacency submatrix.
    # Check if any of the 12 Hamiltonian cycles exists.
    
    for subset in itertools.combinations(range(n), 5):
        a, b, c, d, e = subset
        # Check 12 cycles
        # a-b-c-d-e-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> c) & 1 and (adj_masks[c] >> d) & 1 and (adj_masks[d] >> e) & 1 and (adj_masks[e] >> a) & 1:
            return True
        # a-b-c-e-d-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> c) & 1 and (adj_masks[c] >> e) & 1 and (adj_masks[e] >> d) & 1 and (adj_masks[d] >> a) & 1:
            return True
        # a-b-d-c-e-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> d) & 1 and (adj_masks[d] >> c) & 1 and (adj_masks[c] >> e) & 1 and (adj_masks[e] >> a) & 1:
            return True
        # a-b-d-e-c-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> d) & 1 and (adj_masks[d] >> e) & 1 and (adj_masks[e] >> c) & 1 and (adj_masks[c] >> a) & 1:
            return True
        # a-b-e-c-d-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> e) & 1 and (adj_masks[e] >> c) & 1 and (adj_masks[c] >> d) & 1 and (adj_masks[d] >> a) & 1:
            return True
        # a-b-e-d-c-a
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> e) & 1 and (adj_masks[e] >> d) & 1 and (adj_masks[d] >> c) & 1 and (adj_masks[c] >> a) & 1:
            return True
        # a-c-b-d-e-a (same as a-e-d-b-c-a? No, a-b-c-d-e-a is one. a-c-b-d-e-a is another)
        # We can just check all 24 permutations to be safe, or the 12 unique ones.
        # The above 6 are half of them?
        # Let's just check all 24 permutations of (b,c,d,e) for path a-?-?-?-?-a
        # It's 24 checks per subset. 325k * 24 = 7.8M.
        # In Python, 7.8M iterations might be ~1 second.
        # We only do this if the previous checks failed.
        
        # Optimization: If the induced subgraph has < 5 edges, skip.
        # Count edges in subset
        edge_count = 0
        verts = [a, b, c, d, e]
        for i in range(5):
            for j in range(i+1, 5):
                if (adj_masks[verts[i]] >> verts[j]) & 1:
                    edge_count += 1
        if edge_count < 5:
            continue
            
        # Check permutations
        found = False
        for p1, p2, p3, p4 in itertools.permutations((b, c, d, e)):
            if ((adj_masks[a] >> p1) & 1 and 
                (adj_masks[p1] >> p2) & 1 and 
                (adj_masks[p2] >> p3) & 1 and 
                (adj_masks[p3] >> p4) & 1 and 
                (adj_masks[p4] >> a) & 1):
                found = True
                break
        if found:
            return True

    return False

def _has_independent_set_9(n, adj):
    """Check if graph has an independent set of size 9. Returns True if it does."""
    adj_sets = [set(adj[i]) for i in range(n)]
    found = [False]
    
    def _search(chosen, candidates):
        if found[0]:
            return
        if len(chosen) >= 9:
            found[0] = True
            return
        if len(chosen) + len(candidates) < 9:
            return
        if not candidates:
            return
        
        # Heuristic: pick vertex with min degree in candidate subgraph
        min_deg = float('inf')
        min_v = None
        for v in candidates:
            deg = len(adj_sets[v] & candidates)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        v = min_v
        
        # Include v
        new_candidates = candidates - {v} - adj_sets[v]
        _search(chosen + [v], new_candidates)
        if found[0]:
            return
        
        # Exclude v
        _search(chosen, candidates - {v})
        
    _search([], set(range(n)))
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Construct the known n=32 solution: 8 disjoint K4s
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    
    # Verify base solution
    adj_base = [set() for _ in range(n_base)]
    for u, v in edges_base:
        adj_base[u].add(v)
        adj_base[v].add(u)
        
    if not _has_c5(n_base, adj_base) and not _has_independent_set_9(n_base, adj_base):
        best_n = n_base
        best_edges = edges_base
    else:
        # Fallback to random search for n=32 if base fails (shouldn't happen)
        best_n = 0
        best_edges = []

    # 2. Try to find n=33 solution
    # Since 8K4 cannot be extended, we try a random search.
    # This is unlikely to succeed quickly, so we limit the time.
    if time.time() < deadline - 10:
        # Try a few random graphs on 33 vertices
        # Bias towards sparse graphs to avoid C5? Or dense to avoid alpha?
        # C5-free graphs are often sparse or structured.
        # Let's try starting from a random graph and removing edges to kill C5, then checking alpha.
        
        # Actually, a better heuristic:
        # Start with 8K4 + 1 vertex.
        # Connect the new vertex to 1 vertex in each K4.
        # This gives alpha=9.
        # We need to reduce alpha.
        # Add edges between the K4s?
        # If we add edges between K4s, we might create C5s.
        
        # Given the difficulty, we will just do a very short random search.
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            n_try = 33
            # Random graph with p=0.5
            edges_try = []
            for u in range(n_try):
                for v in range(u + 1, n_try):
                    if random.random() < 0.5:
                        edges_try.append([u, v])
            
            adj_try = [set() for _ in range(n_try)]
            for u, v in edges_try:
                adj_try[u].add(v)
                adj_try[v].add(u)
                
            if not _has_c5(n_try, adj_try) and not _has_independent_set_9(n_try, adj_try):
                best_n = n_try
                best_edges = edges_try
                break

    return {'n': best_n, 'edges': best_edges}
