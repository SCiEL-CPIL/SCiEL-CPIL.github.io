"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy: Start with random regular graphs (high density) and repair violations.
    # Theoretical max for K2,3-free is roughly n^2/4 + O(n) or similar.
    # For n=20, we can try degrees around 9 or 10.
    
    # We will try multiple random restarts of the repair process.
    # Each restart:
    # 1. Generate a random d-regular graph (or close to it).
    # 2. Find all K2,3 subgraphs.
    # 3. Use a local search (greedy or simulated annealing) to remove edges
    #    that break the most K2,3 subgraphs.
    
    seeds = range(500)
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        
        random.seed(seed)
        
        # Choose a target degree. 
        # K2,3-free implies for any pair u,v, |N(u) ∩ N(v)| <= 2.
        # Sum of degrees = 2m.
        # If d-regular, m = nd/2.
        # Let's try d=9, 10, 11.
        # For n=20, d=10 gives m=100. d=11 gives m=110.
        # Let's try a few degrees.
        
        for d in [9, 10, 11]:
            if time.time() > deadline - 5.0:
                break
                
            # Generate random d-regular graph
            edges = _generate_random_regular_graph(n, d)
            if edges is None:
                continue
            
            # Repair phase
            repaired_edges = _repair_k23(edges, n, time_limit=2.0)
            
            if len(repaired_edges) > best_score:
                best_score = len(repaired_edges)
                best_edges = repaired_edges
                
    # Final check: ensure best_edges is valid and not empty
    if not best_edges:
        # Fallback to a simple valid graph if repair failed everywhere
        # e.g. 5 disjoint K4s (score 30) or a sparse random graph
        best_edges = _fallback_graph(n)
        
    return best_edges

def _generate_random_regular_graph(n, d):
    """Generates a random d-regular graph on n vertices. Returns list of edges or None."""
    if d > n - 1:
        return None
    if (n * d) % 2 != 0:
        return None
        
    # Configuration model
    stubs = []
    for v in range(n):
        for _ in range(d):
            stubs.append(v)
            
    random.shuffle(stubs)
    
    edges = []
    adjacency = {v: set() for v in range(n)}
    
    for i in range(0, n * d, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry or swap
            # Simple fix: find another stub not pointing to u
            # For simplicity in this heuristic, we can just skip and try next pair
            # But this breaks regularity. 
            # Better: re-shuffle or use a more robust generator.
            # Given n=20 is small, we can use a simple backtracking or just accept 
            # that some attempts might fail or we use a different method.
            # Let's use a simple check: if self-loop, swap with next available.
            # Actually, for d-regular graphs on small n, self-loops are common in naive config model.
            # We'll just return None if we encounter a self-loop to keep it simple and robust, 
            # relying on multiple restarts.
            return None
        if v in adjacency[u]:
            # Multi-edge
            return None
        edges.append([u, v])
        adjacency[u].add(v)
        adjacency[v].add(u)
        
    return edges

def _find_k23_violations(edges, n):
    """
    Returns a list of sets of edges that form a K2,3 subgraph.
    A K2,3 is defined by two vertices u, v and three common neighbors w1, w2, w3.
    The edges are (u,w1), (u,w2), (u,w3), (v,w1), (v,w2), (v,w3).
    To break a K2,3, we must remove at least one of these 6 edges.
    """
    adjacency = {v: set() for v in range(n)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
        
    violations = []
    # Check all pairs (u, v)
    for u, v in itertools.combinations(range(n), 2):
        common = adjacency[u] & adjacency[v]
        if len(common) >= 3:
            # Pick any 3 common neighbors to define a specific K2,3 instance
            # We can just record the set of 6 edges for the first 3 common neighbors
            # Or better, record the set of all edges involved in *any* K2,3 containing u,v.
            # Actually, for repair, we want to know which edges are in *some* K2,3.
            # Let's collect all edges that are part of at least one K2,3.
            # And maybe group them by the (u,v) pair causing the violation.
            pass
            
    # Alternative: Return a list of (u, v, common_neighbors) tuples where len(common)>=3
    # Then in repair, we can pick a specific K2,3 instance (u,v,3 neighbors) to break.
    viol_list = []
    for u, v in itertools.combinations(range(n), 2):
        common = adjacency[u] & adjacency[v]
        if len(common) >= 3:
            # We have a violation. 
            # We can store the pair and the common neighbors.
            viol_list.append((u, v, list(common)))
            
    return viol_list

def _repair_k23(edges, n, time_limit=2.0):
    """
    Greedily remove edges to eliminate all K2,3 subgraphs.
    """
    start_time = time.time()
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_edges = list(edge_set)
    
    # Precompute adjacency for fast checks? 
    # We'll recompute or maintain it.
    
    while time.time() < start_time + time_limit:
        violations = _find_k23_violations(current_edges, n)
        if not violations:
            break
            
        # Pick a violation (u, v, common)
        # To be effective, pick the violation with the most common neighbors?
        # Or just the first one.
        # Let's pick the one with the most common neighbors to break the "worst" pair first.
        violations.sort(key=lambda x: len(x[2]), reverse=True)
        u, v, common = violations[0]
        
        # We need to remove an edge to break this K2,3.
        # The edges in this K2,3 are:
        # (u, w) for w in common
        # (v, w) for w in common
        # Removing any one of these breaks *this specific* K2,3 instance.
        # However, removing an edge might break multiple K2,3 instances.
        # Let's count how many K2,3 instances each candidate edge is part of.
        
        candidate_edges = set()
        for w in common:
            candidate_edges.add((u, w) if u < w else (w, u))
            candidate_edges.add((v, w) if v < w else (w, v))
            
        # For each candidate edge, count how many violations it appears in.
        # This is expensive if we re-scan all violations for each candidate.
        # Given n=20, number of violations is small.
        
        best_edge_to_remove = None
        max_benefit = -1
        
        for e in candidate_edges:
            benefit = 0
            # Check how many violations contain this edge
            # A violation (u2, v2, common2) contains edge e if e is one of the 6 edges.
            # This check is O(|violations|).
            for u2, v2, common2 in violations:
                # Check if e is in the K2,3 formed by u2, v2 and any 3 from common2
                # Actually, the K2,3 is defined by u2, v2 and *any* 3 neighbors.
                # So e is in *some* K2,3 for pair (u2, v2) if:
                # 1. e is (u2, w) or (v2, w) for some w in common2
                # 2. AND len(common2) >= 3 (already true)
                
                # So we just check if e connects u2 or v2 to a node in common2.
                if e[0] == u2 or e[1] == u2 or e[0] == v2 or e[1] == v2:
                    # It's an edge incident to u2 or v2.
                    # Check if the other endpoint is in common2
                    other = e[1] if e[0] in (u2, v2) else e[0]
                    if other in common2:
                        benefit += 1
                        # Note: This counts the pair (u2, v2) once.
                        # But a single pair (u2, v2) with k common neighbors has C(k,3) K2,3 subgraphs.
                        # Breaking one edge breaks all C(k,3) subgraphs involving that edge?
                        # No, removing edge (u2, w) breaks all K2,3s that use (u2, w).
                        # So the benefit is proportional to the number of K2,3s using that edge.
                        # For a pair (u2, v2) with k common neighbors, the edge (u2, w) is in C(k-1, 2) K2,3s?
                        # Let's simplify: just count the number of pairs (u2, v2) for which this edge is a "candidate".
                        # This is a heuristic.
            
            if benefit > max_benefit:
                max_benefit = benefit
                best_edge_to_remove = e
                
        if best_edge_to_remove:
            current_edges.remove(best_edge_to_remove)
            edge_set.remove(best_edge_to_remove)
            
    return [list(e) for e in current_edges]

def _fallback_graph(n):
    # 5 disjoint K4s
    edges = []
    for k in range(5):
        edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return edges
# EVOLVE-BLOCK-END
