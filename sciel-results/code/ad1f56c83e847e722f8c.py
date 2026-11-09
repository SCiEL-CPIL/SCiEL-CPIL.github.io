import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _common_neighbors_count(u, v, adj, current_edges):
    """Count common neighbors of u,v using set intersection (faster than manual loop)."""
    # Common neighbors must be adjacent to both u and v
    smaller = adj[u] if len(adj[u]) < len(adj[v]) else adj[v]
    larger = adj[v] if smaller is adj[u] else adj[u]
    common = smaller & larger
    # Filter to only those where both edges are in current_edges
    count = 0
    for w in common:
        if w == u or w == v:
            continue
        if _edge_key(u, w) in current_edges and _edge_key(v, w) in current_edges:
            count += 1
    return count

def _try_merge_cliques(partition, adj, current_edges_set):
    """Try to merge pairs of cliques if their union is also a clique in the original graph.
    Returns improved partition or None if no improvement found."""
    if len(partition) < 2:
        return None
    
    # Build edge set for quick lookup
    all_edges = set()
    for clique in partition:
        for a, b in combinations(clique, 2):
            all_edges.add(_edge_key(a, b))
    
    improved = True
    new_partition = list(partition)
    
    while improved:
        improved = False
        i = 0
        while i < len(new_partition) - 1:
            j = i + 1
            while j < len(new_partition):
                c1 = new_partition[i]
                c2 = new_partition[j]
                # Check if union is a clique: every pair across c1,c2 must be an edge
                can_merge = True
                for a in c1:
                    for b in c2:
                        if _edge_key(a, b) not in all_edges:
                            can_merge = False
                            break
                    if not can_merge:
                        break
                if can_merge:
                    # Merge: remove j, extend i
                    new_partition[i] = c1 + c2
                    new_partition.pop(j)
                    improved = True
                    break
                j += 1
            if improved:
                break
            i += 1
    
    if improved:
        return new_partition
    return None

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    initial_edges = set(edges)
    
    best_partition = None
    best_count = float('inf')
    
    # Run multiple random restarts with different strategies
    # Strategy 0: min common neighbors (hard edges first) - most effective
    # Strategy 1: max common neighbors (dense regions first)
    # Strategy 2: random
    # Strategy 3: second-lowest common neighbors
    max_restarts = 60
    for run_seed in range(max_restarts):
        if not _deadline_ok():
            break
            
        current_edges = set(initial_edges)
        partition = []
        rng = random.Random(seed + run_seed)
        
        while current_edges:
            if not _deadline_ok():
                # Fallback: just add remaining edges as K2
                for e in current_edges:
                    partition.append([e[0], e[1]])
                break
            
            # 1. Find edge based on strategy
            strategy = run_seed % 4
            
            if strategy == 0:
                # Min common neighbors (hard edges first)
                target_common = float('inf')
                candidates = []
                for u, v in current_edges:
                    common = _common_neighbors_count(u, v, adj, current_edges)
                    if common < target_common:
                        target_common = common
                        candidates = [(u, v)]
                    elif common == target_common:
                        candidates.append((u, v))
            elif strategy == 1:
                # Max common neighbors (dense regions first)
                target_common = -1
                candidates = []
                for u, v in current_edges:
                    common = _common_neighbors_count(u, v, adj, current_edges)
                    if common > target_common:
                        target_common = common
                        candidates = [(u, v)]
                    elif common == target_common:
                        candidates.append((u, v))
            elif strategy == 2:
                # Random edge
                candidates = list(current_edges)
            else:
                # Second-lowest: pick among edges with common <= median
                commons = []
                for u, v in current_edges:
                    common = _common_neighbors_count(u, v, adj, current_edges)
                    commons.append((common, u, v))
                commons.sort()
                # Take bottom 30%
                k = max(1, len(commons) // 3)
                candidates = [(u, v) for _, u, v in commons[:k]]
            
            if not candidates:
                break
                
            u, v = rng.choice(candidates)
            
            # 2. Expand to maximal clique
            clique = [u, v]
            
            # Find candidates for expansion: vertices connected to all in clique
            candidates_w = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e_uw = _edge_key(u, w)
                e_vw = _edge_key(v, w)
                if e_uw in current_edges and e_vw in current_edges:
                    candidates_w.append(w)
            
            while candidates_w:
                if not _deadline_ok():
                    break
                
                # Pick w with max degree in the subgraph induced by candidates_w
                max_deg = -1
                best_w = -1
                tie_list = []
                
                for w in candidates_w:
                    deg = 0
                    for x in candidates_w:
                        if w == x:
                            continue
                        e_wx = _edge_key(w, x)
                        if e_wx in current_edges:
                            deg += 1
                    
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                        tie_list = [w]
                    elif deg == max_deg:
                        tie_list.append(w)
                
                if best_w == -1:
                    break
                    
                if len(tie_list) > 1:
                    best_w = rng.choice(tie_list)
                    
                clique.append(best_w)
                candidates_w.remove(best_w)
                
                # Filter candidates_w: must be connected to best_w
                new_candidates_w = []
                for w in candidates_w:
                    e_wbw = _edge_key(w, best_w)
                    if e_wbw in current_edges:
                        new_candidates_w.append(w)
                candidates_w = new_candidates_w
                
            partition.append(clique)
            
            # 3. Remove edges of the clique
            for a, b in combinations(clique, 2):
                e = _edge_key(a, b)
                if e in current_edges:
                    current_edges.remove(e)
                    
        count = len(partition)
        
        # Local refinement: try to merge cliques
        if count < best_count:
            merged = _try_merge_cliques(partition, adj, initial_edges)
            if merged is not None:
                count = len(merged)
                partition = merged
        
        if count < best_count:
            best_count = count
            best_partition = partition
            
    if best_partition is None:
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
