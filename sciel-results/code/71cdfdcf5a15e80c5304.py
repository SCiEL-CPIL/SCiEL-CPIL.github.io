"""Combined packing program: large-first backtracking with multi-seed restarts."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    trees = instance_trees(20, 611)
    n = 20

    # Sort tree indices by size (descending) to pack larger trees first
    tree_indices_desc = list(range(n))
    tree_indices_desc.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Also try ascending order
    tree_indices_asc = list(range(n))
    tree_indices_asc.sort(key=lambda i: len(trees[i]))
    
    # Try a few different orderings
    orderings = [tree_indices_desc, tree_indices_asc]
    
    # Add some random orderings
    rng_order = Random(42)
    for _ in range(4):
        idx = list(range(n))
        rng_order.shuffle(idx)
        orderings.append(idx)

    def find_embedding_systematic(tree_edges, used_edges_global, rng, order_hint=None):
        """
        Finds an injective embedding of tree_edges into K20 
        using edges not in used_edges_global.
        Returns a list of length k where result[i] is host vertex for tree vertex i, or None.
        """
        k = len(tree_edges) + 1
        if k == 1:
            return [0]
        
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS order from 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
                    
        host = [-1] * k
        used_hosts = set()
        
        # Precompute available hosts for each tree vertex based on parent constraints
        # For root, all hosts are available
        # For others, hosts must not be connected to parent's host via a used edge
        
        def get_candidates(tv, p_tv, p_hv):
            """Get candidate host vertices for tree vertex tv."""
            cands = []
            for hv in range(n):
                if hv in used_hosts:
                    continue
                if p_tv != -1:
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges_global:
                        continue
                cands.append(hv)
            return cands
        
        def backtrack(idx, nodes_backtracked=0):
            if time.time() > deadline:
                return False
                
            if idx == k:
                return True
            
            tv = order[idx]
            p_tv = parent[tv]
            p_hv = host[p_tv] if p_tv != -1 else -1
            
            cands = get_candidates(tv, p_tv, p_hv)
            
            if not cands:
                return False
            
            # Sort candidates: prefer hosts with fewer remaining options for children
            # (this is a simple forward-checking heuristic)
            if len(cands) > 1:
                def score_candidate(hv):
                    # Count how many unused edges from hv
                    count = 0
                    for nh in range(n):
                        if nh not in used_hosts and nh != hv:
                            e = (min(hv, nh), max(hv, nh))
                            if e not in used_edges_global:
                                count += 1
                    return count
                cands.sort(key=score_candidate)
            
            # Add some randomness
            if rng.random() < 0.3:
                rng.shuffle(cands)
            
            for hv in cands:
                host[tv] = hv
                used_hosts.add(hv)
                
                if backtrack(idx + 1, nodes_backtracked + 1):
                    return True
                
                used_hosts.remove(hv)
                host[tv] = -1
            
            return False
        
        if backtrack(0):
            return host[:]
        else:
            return None

    def try_pack_greedy(seed, tree_order):
        rng = Random(seed)
        used = set()
        res = [None] * n
        score = 0
        
        for idx in tree_order:
            if time.time() > deadline:
                break
                
            tree = trees[idx]
            embedding = find_embedding_systematic(tree, used, rng)
            
            if embedding is not None:
                res[idx] = embedding
                score += len(tree)
                for u, v in tree:
                    hu = embedding[u]
                    hv = embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.add(e)
                    
        return res, score

    best_res = None
    best_score = 0
    
    seed_counter = 0
    for tree_order in orderings:
        for seed in range(100):
            if time.time() > deadline:
                break
            seed_counter += 1
            
            res, score = try_pack_greedy(seed_counter, tree_order)
            
            if score > best_score:
                best_score = score
                best_res = res
                
            if best_score == 190:
                break
        if best_score == 190:
            break

    if best_res is None:
        # Fallback: embed T1 and T2 at least
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    # Ensure types are correct
    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
            
    return final_result
