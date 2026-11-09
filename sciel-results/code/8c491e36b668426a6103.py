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

def _greedy_partition_from_state(current_edges, adj, n, rng, strategy):
    """
    Runs the greedy clique peeling algorithm with a given strategy.
    Returns partition list.
    """
    partition = []
    if not current_edges:
        return partition
    
    while current_edges:
        if not _deadline_ok():
            for e in current_edges:
                partition.append([e[0], e[1]])
            break
        
        # Select edge based on strategy
        if strategy == 0:  # max common neighbors
            max_common = -1
            candidates = []
            for u, v in current_edges:
                common = 0
                if len(adj[u]) < len(adj[v]):
                    for w in adj[u]:
                        if w == v: continue
                        e_vw = _edge_key(v, w)
                        e_uw = _edge_key(u, w)
                        if e_vw in current_edges and e_uw in current_edges:
                            common += 1
                else:
                    for w in adj[v]:
                        if w == u: continue
                        e_uw = _edge_key(u, w)
                        e_vw = _edge_key(v, w)
                        if e_uw in current_edges and e_vw in current_edges:
                            common += 1
                if common > max_common:
                    max_common = common
                    candidates = [(u, v)]
                elif common == max_common:
                    candidates.append((u, v))
        elif strategy == 1:  # min degree sum
            min_deg = float('inf')
            candidates = []
            for u, v in current_edges:
                deg = len(adj[u]) + len(adj[v])
                if deg < min_deg:
                    min_deg = deg
                    candidates = [(u, v)]
                elif deg == min_deg:
                    candidates.append((u, v))
        else:  # strategy == 2: random
            candidates = list(current_edges)
        
        if not candidates:
            break
            
        u, v = rng.choice(candidates)
        
        # Expand to maximal clique
        clique = [u, v]
        
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
            
            # Pick w with max degree in subgraph induced by candidates_w
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
            
            new_candidates_w = []
            for w in candidates_w:
                e_wbw = _edge_key(w, best_w)
                if e_wbw in current_edges:
                    new_candidates_w.append(w)
            candidates_w = new_candidates_w
        
        partition.append(clique)
        
        for a, b in combinations(clique, 2):
            e = _edge_key(a, b)
            if e in current_edges:
                current_edges.remove(e)
                    
    return partition

def _local_search(partition, adj, n, rng, max_iter=50):
    """
    Try to improve the partition by merging cliques or reassigning edges.
    """
    if not _deadline_ok():
        return partition
    
    current_partition = [list(c) for c in partition]
    current_count = len(current_partition)
    
    # Build edge -> clique index mapping
    edge_to_clique = {}
    for idx, clique in enumerate(current_partition):
        for a, b in combinations(clique, 2):
            edge_to_clique[_edge_key(a, b)] = idx
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter and _deadline_ok():
        improved = False
        iterations += 1
        
        # Try merging pairs of cliques
        for i in range(len(current_partition)):
            if not _deadline_ok():
                break
            for j in range(i + 1, len(current_partition)):
                if not _deadline_ok():
                    break
                
                c1 = set(current_partition[i])
                c2 = set(current_partition[j])
                
                # Check if union is a clique
                union = c1 | c2
                is_clique = True
                for a, b in combinations(union, 2):
                    if b not in adj[a]:
                        is_clique = False
                        break
                
                if is_clique:
                    # Merge: remove all edges from i and j, add union clique
                    new_clique = list(union)
                    
                    # Remove edges belonging to clique i
                    edges_to_remove = []
                    for a, b in combinations(current_partition[i], 2):
                        e = _edge_key(a, b)
                        if e in edge_to_clique:
                            edges_to_remove.append(e)
                    
                    # Remove edges belonging to clique j  
                    for a, b in combinations(current_partition[j], 2):
                        e = _edge_key(a, b)
                        if e in edge_to_clique:
                            edges_to_remove.append(e)
                    
                    for e in edges_to_remove:
                        del edge_to_clique[e]
                    
                    # Remove cliques i and j (adjust indices)
                    current_partition.pop(j)
                    current_partition.pop(i)
                    current_partition.append(new_clique)
                    
                    # Update edge_to_clique for new clique
                    new_idx = len(current_partition) - 1
                    for a, b in combinations(new_clique, 2):
                        edge_to_clique[_edge_key(a, b)] = new_idx
                    
                    current_count -= 1
                    improved = True
                    break
            
            if improved:
                break
    
    return current_partition

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
    for run_seed in range(15):
        if not _deadline_ok():
            break
        
        current_edges = set(initial_edges)
        rng = random.Random(seed + run_seed * 1000)
        
        # Cycle through strategies
        strategy = run_seed % 3
        
        partition = _greedy_partition_from_state(current_edges, adj, n, rng, strategy)
        
        # Local search improvement
        if _deadline_ok():
            partition = _local_search(partition, adj, n, rng, max_iter=30)
        
        count = len(partition)
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
