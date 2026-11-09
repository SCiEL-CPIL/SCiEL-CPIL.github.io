"""Clique edge partitioning via Lagrangian-style iterative assignment."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_LIMIT = 145.0  # Safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch_maximal_cliques(adj, n):
    """
    Enumerate all maximal cliques in the graph using Bron-Kerbosch with pivot.
    Returns a list of sets of vertices.
    """
    cliques = []
    
    def bk(r, p, x):
        if not p and not x:
            if len(r) >= 2:
                cliques.append(set(r))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if p:
            # Pick vertex with max degree in P ∪ X
            pivot_candidates = list(p) + list(x)
            u = max(pivot_candidates, key=lambda v: sum(1 for w in pivot_candidates if w != v and adj[v, w]))
        else:
            u = None
        
        # Branch on vertices in P \ N(u)
        # We iterate over a copy because P changes
        p_copy = list(p)
        for v in p_copy:
            if u is not None and adj[v, u]:
                continue
            
            # Recurse
            bk(r | {v}, p & adj[v, :], x & adj[v, :])
            
            # Update sets
            p.remove(v)
            x.add(v)
            
            if not _deadline_ok():
                return

    # Initialize
    p = set(range(n))
    x = set()
    r = set()
    
    # Optimization: Only run if graph isn't too dense/large for this specific heuristic
    # For n=64, maximal cliques can be many, but manageable.
    bk(r, p, x)
    
    return cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    edge_set = set(edges)
    edge_list = list(edge_set)
    m = len(edge_list)
    
    # 1. Enumerate Maximal Cliques
    # This is the most expensive part. If it's too slow, we fall back to greedy.
    max_cliques = []
    try:
        # Use a time-limited version or just run it. 
        # For n=64, p=0.55, the number of maximal cliques is usually manageable.
        max_cliques = _bron_kerbosch_maximal_cliques(adj, n)
    except Exception:
        pass
    
    # Filter cliques: must contain at least 2 vertices
    max_cliques = [c for c in max_cliques if len(c) >= 2]
    
    # 2. Precompute edge indices for each clique
    # clique_edges[i] = list of edge indices in the i-th maximal clique
    clique_edges = []
    for c in max_cliques:
        c_edges = []
        for u, v in combinations(sorted(c), 2):
            e = (u, v) if u < v else (v, u)
            if e in edge_set:
                c_edges.append(e)
        if c_edges:
            clique_edges.append(c_edges)
    
    # 3. Lagrangian / Iterative Assignment Strategy
    # We want to assign each edge to exactly one clique.
    # Objective: Minimize sum(1/|C_i|) ? No, minimize count of cliques used.
    # This is equivalent to Set Cover with exact cover constraint.
    
    # Heuristic: 
    # Initialize "weight" of each edge.
    # Edges with few common neighbors are "scarce" and should be prioritized into larger cliques if possible.
    # Or, we can view it as: we want to pick a subset of cliques such that every edge is covered exactly once.
    
    # Alternative approach: Greedy with "Regret" or "Opportunity Cost"
    # Let's try a simpler global heuristic:
    # Sort maximal cliques by "efficiency": size / (number of edges they cover that are not yet covered by better cliques)?
    
    # Let's try the "Lagrangian Relaxation" idea from the prompt:
    # Assign a potential to each edge. 
    # Actually, a simpler and often effective method for Exact Cover of edges by cliques is:
    # 1. Start with all edges unassigned.
    # 2. While edges remain:
    #    a. For each candidate clique (from maximal cliques), calculate score = (sum of weights of unassigned edges in clique) / len(clique)
    #    b. Pick clique with max score.
    #    c. Assign those edges.
    #    d. Update weights: increase weight of edges that are "hard" to cover (few other cliques cover them).
    
    # Let's refine the weights:
    # weight[e] = 1 / (number of maximal cliques containing e)
    # This prioritizes edges that are in fewer cliques (more constrained).
    
    edge_to_cliques = {e: [] for e in edge_set}
    for i, c_edges in enumerate(clique_edges):
        for e in c_edges:
            edge_to_cliques[e].append(i)
    
    # If an edge is not in any maximal clique (shouldn't happen for edges in graph, but maybe isolated?), handle it.
    # Actually, every edge is in at least one maximal clique (itself).
    
    # Initialize weights
    weights = {}
    for e in edge_set:
        count = len(edge_to_cliques[e])
        weights[e] = 1.0 / count if count > 0 else 100.0
    
    # Iterative Greedy Assignment
    unassigned = set(edge_set)
    selected_cliques = []
    
    # To avoid picking the same clique repeatedly or picking suboptimal ones,
    # we can use a "discount" for cliques that share many edges with already selected ones?
    # No, we want disjoint edge sets.
    
    # Let's iterate
    # We need to be careful with time.
    # Maximal cliques list can be large.
    
    # Optimization: Precompute clique edge sets as frozensets for fast intersection?
    # Or just use sets.
    
    clique_edge_sets = [set(c_edges) for c_edges in clique_edges]
    
    # Sort cliques initially by size descending? Or by potential score?
    # Let's do a few restarts with different weight initializations.
    
    best_partition = None
    best_score = float('inf')
    
    # Strategy A: Weight by inverse frequency
    # Strategy B: Weight by 1 (uniform)
    # Strategy C: Weight by (1 - common_neighbors)
    
    common = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common[u, v] = np.sum(adj[u] & adj[v])
                common[v, u] = common[u, v]
    
    weight_strategies = []
    
    # Strategy 1: Inverse frequency
    w1 = {e: 1.0 / max(1, len(edge_to_cliques[e])) for e in edge_set}
    weight_strategies.append(w1)
    
    # Strategy 2: Uniform
    w2 = {e: 1.0 for e in edge_set}
    weight_strategies.append(w2)
    
    # Strategy 3: Inverse common neighbors (scarce edges first)
    w3 = {}
    for e in edge_set:
        u, v = e
        cn = common[u, v]
        w3[e] = 1.0 / (cn + 1)
    weight_strategies.append(w3)
    
    for w_init in weight_strategies:
        if not _deadline_ok():
            break
            
        # Reset
        unassigned = set(edge_set)
        selected_cliques = []
        current_weights = dict(w_init)
        
        # We need to efficiently find the best clique.
        # Naive: O(|Cliques| * |Edges in Clique|) per step.
        # |Cliques| can be ~1000-10000 for n=64.
        # Steps = |Edges| ~ 2000.
        # 2000 * 10000 * 10 = 200M ops. Might be too slow in Python.
        
        # Optimization: 
        # Only consider cliques that contain at least one unassigned edge.
        # We can maintain a list of "active" cliques.
        
        # Let's try a simpler heuristic that is faster:
        # Greedy Set Cover with "Gain" = (Unassigned Edges Covered) / (Clique Size)
        # This minimizes the number of cliques.
        
        # To speed up, we can pre-filter cliques.
        # But let's try to make the inner loop fast.
        
        # Alternative: Use the "Hardest Edge" heuristic.
        # Pick the unassigned edge with the highest weight.
        # Among all cliques covering it, pick the one that covers the most *other* unassigned edges (weighted).
        
        # This is O(|Unassigned Edges| * |Clique Candidates for that Edge| * |Clique Size|)
        # |Clique Candidates| is small (edges are in few maximal cliques usually).
        
        # Let's implement this "Hardest Edge" approach.
        
        # Precompute: for each edge, list of clique indices containing it.
        # Already have edge_to_cliques.
        
        while unassigned:
            if not _deadline_ok():
                break
                
            # Pick edge with max weight
            # To avoid O(N) scan every time, we could use a heap, but weights change? 
            # In this strategy, weights are static or updated slowly. 
            # Let's keep weights static for simplicity in this inner loop, or update them.
            # If static, we can just sort once? No, unassigned set changes.
            # Let's just pick max from unassigned. 
            # To make it faster, we can approximate or use a random sample if set is large?
            # For m=2000, max() is fast.
            
            e_max = max(unassigned, key=lambda e: current_weights[e])
            
            # Get candidates
            candidates = edge_to_cliques[e_max]
            
            if not candidates:
                # Should not happen
                selected_cliques.append([e_max[0], e_max[1]])
                unassigned.remove(e_max)
                continue
            
            # Evaluate candidates
            best_c_idx = -1
            best_gain = -1
            
            for c_idx in candidates:
                c_edges = clique_edge_sets[c_idx]
                # Intersection with unassigned
                covered = c_edges & unassigned
                if not covered:
                    continue
                
                # Gain: Sum of weights of covered edges / Size of Clique?
                # Or just sum of weights?
                # We want to minimize count. So gain = 1 / |clique| * sum(weights)?
                # Actually, if we pick this clique, we "pay" 1 clique. We "gain" the coverage of these edges.
                # So we want to maximize (sum of weights of covered edges) / 1? 
                # No, we want to cover as many "hard" edges as possible per clique.
                # So Score = sum(weights[e] for e in covered)
                
                score = sum(current_weights[e] for e in covered)
                
                # Tie-breaker: Prefer larger cliques (cover more edges)
                if score > best_gain or (score == best_gain and len(clique_edge_sets[best_c_idx]) < len(c_edges)):
                    best_gain = score
                    best_c_idx = c_idx
            
            if best_c_idx == -1:
                # Fallback
                selected_cliques.append([e_max[0], e_max[1]])
                unassigned.remove(e_max)
            else:
                c_edges = clique_edge_sets[best_c_idx]
                # Add clique to partition
                # We need the vertex list
                selected_cliques.append(sorted(list(max_cliques[best_c_idx])))
                
                # Remove covered edges from unassigned
                unassigned -= c_edges
                
                # Update weights?
                # Optional: Increase weight of remaining edges if they are in fewer cliques now?
                # Not necessary for basic version.
        
        if unassigned:
            # Add remaining as 2-cliques
            for e in unassigned:
                selected_cliques.append([e[0], e[1]])
        
        score = len(selected_cliques)
        if score < best_score:
            best_score = score
            best_partition = [list(c) for c in selected_cliques]
    
    if best_partition is None:
        return [[e[0], e[1]] for e in edges]
    
    # Local Search: Merge
    # Try to merge pairs of cliques if their union is a clique and edges are disjoint/covered.
    # This is similar to the previous program's local search.
    
    def _local_search_improve(partition, n, adj, edge_set):
        if not _deadline_ok():
            return partition
        
        current_partition = [list(c) for c in partition]
        
        # Build edge to clique index map
        def build_edge_map(part):
            e_map = {}
            for i, c in enumerate(part):
                for u, v in combinations(c, 2):
                    e = (u, v) if u < v else (v, u)
                    e_map[e] = i
            return e_map
        
        e_map = build_edge_map(current_partition)
        
        improved = True
        max_iters = 20
        for _ in range(max_iters):
            if not _deadline_ok():
                break
            improved = False
            
            # Try to merge
            # To speed up, only try cliques that share vertices?
            # Or just random pairs?
            # Let's try all pairs for small n, but n=64 might have many cliques.
            # Limit iterations.
            
            # Shuffle to avoid bias?
            indices = list(range(len(current_partition)))
            random.shuffle(indices)
            
            for i_idx in indices:
                if not _deadline_ok():
                    break
                i = i_idx
                if i >= len(current_partition): continue
                for j_idx in indices:
                    if not _deadline_ok():
                        break
                    j = j_idx
                    if j <= i or j >= len(current_partition):
                        continue
                    
                    C1 = set(current_partition[i])
                    C2 = set(current_partition[j])
                    union_vertices = C1 | C2
                    
                    # Check if union is a clique
                    is_clique = True
                    for u, v in combinations(sorted(union_vertices), 2):
                        if not adj[u, v]:
                            is_clique = False
                            break
                    
                    if not is_clique:
                        continue
                    
                    # Check if edges are covered exactly by i and j
                    involved = set()
                    valid = True
                    for u, v in combinations(sorted(union_vertices), 2):
                        e = (u, v) if u < v else (v, u)
                        if e not in e_map:
                            valid = False
                            break
                        involved.add(e_map[e])
                    
                    if not valid or involved != {i, j}:
                        continue
                    
                    # Merge
                    new_clique = sorted(union_vertices)
                    current_partition.pop(j)
                    current_partition.pop(i)
                    current_partition.append(new_clique)
                    
                    e_map = build_edge_map(current_partition)
                    improved = True
                    break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_partition
    
    if _deadline_ok():
        improved_partition = _local_search_improve(best_partition, n, adj, edge_set)
        if len(improved_partition) < best_score:
            best_partition = improved_partition
    
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
