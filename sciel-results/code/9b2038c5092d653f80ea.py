"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0
    n = 20
    trees = instance_trees(n, 611)
    
    best_result = None
    best_score = -1
    
    max_restarts = 300
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(611 * 1000 + restart * 7919)
        
        used_edges = set()
        result = [None] * n
        
        # Embed from largest to smallest
        order = list(range(n - 1, -1, -1))
        
        success = True
        for idx in order:
            if time.time() > deadline:
                success = False
                break
                
            k = idx + 1
            tree_edges = trees[idx]
            
            if k == 1:
                avail = list(range(n))
                rng.shuffle(avail)
                result[idx] = [avail[0]]
                continue
            
            if k == 2:
                candidates = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            candidates.append((u, v))
                if not candidates:
                    success = False
                    break
                u, v = rng.choice(candidates)
                result[idx] = [u, v]
                used_edges.add((u, v))
                continue
            
            # For k >= 3, use CSP with MRV + Forward Checking
            embedding = _embed_csp(tree_edges, used_edges, n, deadline, rng)
            if embedding is None:
                success = False
                break
            result[idx] = embedding
            for (a, b) in tree_edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        
        if success:
            total_edges = sum(len(trees[idx]) for idx in range(n))
            score = total_edges
            if score > best_score:
                best_score = score
                best_result = [list(r) if r is not None else None for r in result]
                if best_score == 190:
                    break
    
    if best_result is None:
        best_result = [None] * n
    
    return best_result

def _embed_csp(tree_edges, used_edges, n, deadline, rng):
    """Embed a tree using CSP with MRV and Forward Checking."""
    k = len(tree_edges) + 1
    
    # Build adjacency for the tree
    tree_adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    
    # Precompute free host edges as a set of frozensets for fast lookup
    free_host_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in used_edges:
                free_host_edges.add((u, v))
    
    # Domains: for each tree vertex, which host vertices it can map to
    domains = [list(range(n)) for _ in range(k)]
    
    # Track assignments
    assignment = [-1] * k
    assigned = [False] * k
    
    # Precompute: for each tree vertex, its neighbors
    # and for fast edge checking
    
    def check_edge(hv1, hv2):
        """Check if edge between host vertices hv1, hv2 is free and they're distinct."""
        if hv1 == hv2:
            return False
        u, v = min(hv1, hv2), max(hv1, hv2)
        return (u, v) in free_host_edges
    
    def is_consistent(tv, hv):
        """Check if assigning tv->hv is consistent with current assignments."""
        for nb in tree_adj[tv]:
            if assigned[nb]:
                if not check_edge(hv, assignment[nb]):
                    return False
        return True
    
    def prune_domains(tv, hv, domains):
        """Prune domains of unassigned neighbors of tv after assigning tv->hv.
        Returns False if any domain becomes empty."""
        for nb in tree_adj[tv]:
            if not assigned[nb]:
                new_domain = []
                for hv2 in domains[nb]:
                    if check_edge(hv, hv2):
                        new_domain.append(hv2)
                if not new_domain:
                    return False
                domains[nb] = new_domain
        return True
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # MRV: pick unassigned tree vertex with smallest domain
        min_size = float('inf')
        min_tv = -1
        for tv in range(k):
            if not assigned[tv]:
                sz = len(domains[tv])
                if sz < min_size:
                    min_size = sz
                    min_tv = tv
                    if sz <= 1:
                        break
        
        if min_tv == -1:
            return True  # All assigned
        
        if min_size == 0:
            return False
        
        tv = min_tv
        candidates = domains[tv][:]
        rng.shuffle(candidates)
        
        for hv in candidates:
            if not is_consistent(tv, hv):
                continue
            
            # Save state
            old_domains = [d[:] for d in domains]
            old_assigned = assigned[:]
            old_assignment = assignment[:]
            
            assigned[tv] = True
            assignment[tv] = hv
            domains[tv] = [hv]
            
            if prune_domains(tv, hv, domains):
                if backtrack():
                    return True
            
            # Restore
            for i in range(k):
                assigned[i] = old_assigned[i]
                assignment[i] = old_assignment[i]
                domains[i] = old_domains[i]
        
        return False
    
    if backtrack():
        return [assignment[i] for i in range(k)]
    
    return None
# EVOLVE-BLOCK-END
