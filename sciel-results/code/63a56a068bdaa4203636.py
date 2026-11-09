"""Rewritten program for clique edge partitioning with improved heuristics."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

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

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix (0/1)
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    edge_set = set(edges)
    
    # Precompute common neighbor counts for all pairs
    # common[u,v] = number of w such that (u,w) and (v,w) are edges
    common = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common[u, v] = np.sum(adj[u] & adj[v])
                common[v, u] = common[u, v]
    
    # Greedy partition with multiple strategies and restarts
    def _greedy_partition(edges, n, adj, common, seed_val, strategy):
        current_edges = set(edges)
        partition = []
        rng = random.Random(seed_val)
        
        if strategy == 'random':
            edge_list = list(current_edges)
            rng.shuffle(edge_list)
        elif strategy == 'max_common':
            # Sort edges by common neighbor count (descending)
            edge_list = sorted(current_edges, key=lambda e: -common[e[0], e[1]])
        else:
            edge_list = list(current_edges)
        
        for u, v in edge_list:
            if not _deadline_ok():
                break
            if (u, v) not in current_edges:
                continue
            
            # Build clique starting from (u, v)
            clique = [u, v]
            
            # Candidates: vertices adjacent to all in clique
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                if (u, w) in current_edges and (v, w) in current_edges:
                    candidates.append(w)
            
            # Greedily add vertices
            while candidates:
                if not _deadline_ok():
                    break
                
                # Score each candidate by number of common neighbors with other candidates
                # that are also in current_edges
                best_w = None
                best_score = -1
                
                for w in candidates:
                    score = 0
                    for x in candidates:
                        if w == x:
                            continue
                        e_wx = (w, x) if w < x else (x, w)
                        if e_wx in current_edges:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w is None or best_score < 0:
                    break
                
                clique.append(best_w)
                candidates.remove(best_w)
                
                # Filter candidates: must be adjacent to best_w
                new_candidates = []
                for w in candidates:
                    e_wbw = (w, best_w) if w < best_w else (best_w, w)
                    if e_wbw in current_edges:
                        new_candidates.append(w)
                candidates = new_candidates
            
            partition.append(clique)
            
            # Remove all edges in this clique from current_edges
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                current_edges.discard(e)
        
        # Add remaining edges as 2-cliques
        for e in current_edges:
            partition.append([e[0], e[1]])
        
        return partition
    
    # Local search: try to merge cliques that together form a larger clique
    def _local_search_improve(partition, n, adj, edge_set):
        if not _deadline_ok():
            return partition
        
        # Build edge-to-clique map
        def build_edge_map(part):
            e_map = {}
            for i, c in enumerate(part):
                for u, v in combinations(c, 2):
                    e = (u, v) if u < v else (v, u)
                    e_map[e] = i
            return e_map
        
        current_partition = [list(c) for c in partition]
        e_map = build_edge_map(current_partition)
        
        improved = True
        max_iters = 50
        for _ in range(max_iters):
            if not _deadline_ok():
                break
            
            improved = False
            
            # Try to merge pairs of cliques
            for i in range(len(current_partition)):
                if not _deadline_ok():
                    break
                for j in range(i + 1, len(current_partition)):
                    if not _deadline_ok():
                        break
                    
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
                    
                    # Check if the edges of the union are covered exactly by C1 and C2
                    # i.e., no other clique covers any edge within union_vertices
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
                    
                    # Merge: replace C1 and C2 with union_vertices
                    new_clique = sorted(union_vertices)
                    current_partition.pop(j)
                    current_partition.pop(i)
                    current_partition.append(new_clique)
                    
                    # Rebuild edge map
                    e_map = build_edge_map(current_partition)
                    improved = True
                    break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_partition
    
    # Try multiple strategies and restarts
    best_partition = None
    best_count = float('inf')
    
    rng_master = random.Random(seed + 1000)
    
    # Strategy 1: Random order with multiple restarts
    for restart in range(20):
        if not _deadline_ok():
            break
        s = seed + restart * 7919
        partition = _greedy_partition(edges, n, adj, common, s, 'random')
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = [list(c) for c in partition]
    
    # Strategy 2: Max common neighbors order
    for restart in range(10):
        if not _deadline_ok():
            break
        s = seed + 5000 + restart * 7919
        partition = _greedy_partition(edges, n, adj, common, s, 'max_common')
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = [list(c) for c in partition]
    
    if best_partition is None:
        return [[e[0], e[1]] for e in edges]
    
    # Local search improvement
    if _deadline_ok():
        improved_partition = _local_search_improve(best_partition, n, adj, edge_set)
        if len(improved_partition) < best_count:
            best_partition = improved_partition
            best_count = len(best_partition)
    
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
