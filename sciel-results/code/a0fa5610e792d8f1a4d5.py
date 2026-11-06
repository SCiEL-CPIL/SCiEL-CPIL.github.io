"""Constraint propagation with precomputed regularity and conflict-maximizing edge flips."""
import itertools
import random
import time
from collections import defaultdict

def _check_regular_fast(vertices, adj_bits):
    """Check if induced subgraph on vertices is regular using bitmask adjacency."""
    k = len(vertices)
    if k <= 1:
        return True
    degs = []
    for v in vertices:
        d = 0
        for u in vertices:
            if u != v and (adj_bits[v] >> u) & 1:
                d += 1
        degs.append(d)
    first = degs[0]
    return all(d == first for d in degs[1:])

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices."""
    
    def try_build(n, max_time=55):
        """Try to build a valid graph on n vertices."""
        start = time.time()
        random.seed(n * 7919 + 13)  # Deterministic seed for reproducibility
        
        # Use bitmask representation for fast adjacency checks
        # adj_bits[v] is a bitmask where bit u is 1 if edge (v,u) exists
        adj_bits = [0] * n
        
        # Initialize with random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Precompute all 6-vertex subsets
        subsets6 = list(itertools.combinations(range(n), 6))
        
        # For each 6-subset, precompute the list of edges (as (min,max) pairs)
        # and for each edge, which subsets it belongs to
        subset_edges = {}
        edge_to_subsets = defaultdict(list)
        
        for idx, subset in enumerate(subsets6):
            edges_in_subset = []
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = subset[i], subset[j]
                    key = (min(u, v), max(u, v))
                    edges_in_subset.append(key)
                    edge_to_subsets[key].append(idx)
            subset_edges[idx] = edges_in_subset
        
        # Helper: compute regularity status for all 6-subsets given current adj_bits
        def compute_violations():
            """Return set of indices of violating 6-subsets."""
            viol = set()
            for idx, subset in enumerate(subsets6):
                if _check_regular_fast(subset, adj_bits):
                    viol.add(idx)
            return viol
        
        # Helper: verify no subset of size >= 7 is regular
        def verify_no_large_regular():
            """Check all subsets of size 7 to n for regularity."""
            for k in range(7, n + 1):
                for subset in itertools.combinations(range(n), k):
                    if _check_regular_fast(subset, adj_bits):
                        return False
            return True
        
        # Main optimization loop
        for iteration in range(50000):
            if time.time() - start > max_time:
                break
            
            viol = compute_violations()
            
            if not viol:
                # No 6-vertex violations; check larger subsets
                if verify_no_large_regular():
                    # Success! Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_bits[i] >> j) & 1:
                                edges.append([i, j])
                    return edges, True
                else:
                    # Find a large regular subset and break it
                    large_viol = None
                    for k in range(7, n + 1):
                        for subset in itertools.combinations(range(n), k):
                            if _check_regular_fast(subset, adj_bits):
                                large_viol = subset
                                break
                        if large_viol:
                            break
                    
                    if large_viol:
                        # Try flipping each edge in the large subset
                        best_flip = None
                        best_viol_count = len(compute_violations()) + 1000
                        for i in range(len(large_viol)):
                            for j in range(i + 1, len(large_viol)):
                                u, v = large_viol[i], large_viol[j]
                                # Flip edge
                                adj_bits[u] ^= (1 << v)
                                adj_bits[v] ^= (1 << u)
                                new_viol = compute_violations()
                                # Restore
                                adj_bits[u] ^= (1 << v)
                                adj_bits[v] ^= (1 << u)
                                if len(new_viol) < best_viol_count:
                                    best_viol_count = len(new_viol)
                                    best_flip = (u, v)
                        
                        if best_flip:
                            u, v = best_flip
                            adj_bits[u] ^= (1 << v)
                            adj_bits[v] ^= (1 << u)
                    continue
            
            # We have violations. For each edge, compute how many violations it could resolve.
            # An edge flip changes the degree parity in all subsets containing that edge.
            # Strategy: for each candidate edge, estimate the net change in violations.
            
            # Build edge -> list of violating subsets it belongs to
            edge_viol_subsets = defaultdict(list)
            for idx in viol:
                for edge in subset_edges[idx]:
                    edge_viol_subsets[edge].append(idx)
            
            # For each edge in violating subsets, try flipping and count new violations
            # among subsets affected by this flip
            candidate_edges = list(edge_viol_subsets.keys())
            if not candidate_edges:
                # Random perturbation
                for _ in range(3):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj_bits[u] ^= (1 << v)
                        adj_bits[v] ^= (1 << u)
                continue
            
            # Score each candidate edge by how many violations it's in
            # and try the top ones
            scored_edges = [(edge, len(edge_viol_subsets[edge])) for edge in candidate_edges]
            scored_edges.sort(key=lambda x: -x[1])
            
            improved = False
            for (u, v), count in scored_edges[:15]:
                # Flip the edge
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                # Recompute violations only for subsets containing this edge
                # plus a random sample of others for safety
                new_viol = set()
                # Check all subsets containing this edge
                for idx in edge_to_subsets[(min(u,v), max(u,v))]:
                    subset = subsets6[idx]
                    if _check_regular_fast(subset, adj_bits):
                        new_viol.add(idx)
                # Also check a random sample of other subsets
                for _ in range(50):
                    idx = random.randint(0, len(subsets6) - 1)
                    subset = subsets6[idx]
                    if _check_regular_fast(subset, adj_bits):
                        new_viol.add(idx)
                
                if len(new_viol) < len(viol):
                    improved = True
                    break
                else:
                    # Undo flip
                    adj_bits[u] ^= (1 << v)
                    adj_bits[v] ^= (1 << u)
            
            if not improved:
                # Simulated annealing style: accept worse moves with some probability
                # or do random flips
                if random.random() < 0.3:
                    # Accept a random flip
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj_bits[u] ^= (1 << v)
                        adj_bits[v] ^= (1 << u)
                else:
                    # Try multiple random flips
                    for _ in range(5):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u != v:
                            adj_bits[u] ^= (1 << v)
                            adj_bits[v] ^= (1 << u)
        
        return None, False
    
    # Try n=21 first, then 20
    for n in [21, 20]:
        edges, success = try_build(n)
        if success:
            return {'n': n, 'edges': edges}
    
    # Fallback: return a trivially valid small graph
    # A graph with 5 vertices and no edges is valid (no 6-vertex subsets)
    return {'n': 5, 'edges': []}
