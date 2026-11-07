"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance_trees(n=20, seed=611):
    rng = Random(seed)
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

def _build_adj(tree_edges, k):
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree_backtrack(tree_edges, k, used_edges, rng, max_attempts=200):
    """Embed tree into K20 using unused edges via randomized DFS with backtracking."""
    if k == 1:
        # Pick any vertex
        return [rng.randint(0, 19)]
    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    return [u, v]
        return None
    
    adj = _build_adj(tree_edges, k)
    
    # Precompute for each tree vertex, the list of neighbors
    # We'll use a randomized DFS approach
    
    for attempt in range(max_attempts):
        # Random root
        root = rng.randint(0, k - 1)
        
        # Build a BFS/DFS ordering from root
        visited = [False] * k
        order = []
        parent = [-1] * k
        stack = [root]
        visited[root] = True
        while stack:
            cur = stack.pop()
            order.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = cur
                    stack.append(nb)
        
        # Shuffle the order slightly to add randomness (but keep parent before child)
        # Actually, let's just use the DFS order as-is but with random root
        
        assignment = [-1] * k
        available = [True] * 20
        
        def dfs(idx):
            if idx == len(order):
                return True
            tv = order[idx]
            if tv == root:
                # Try all available host vertices
                hosts = [h for h in range(20) if available[h]]
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False
            else:
                p = parent[tv]
                ph = assignment[p]
                if ph == -1:
                    return False
                # Try all available host vertices adjacent to ph with unused edge
                hosts = []
                for h in range(20):
                    if available[h] and h != ph:
                        if (ph, h) not in used_edges and (h, ph) not in used_edges:
                            hosts.append(h)
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False
        
        if dfs(0):
            # Verify
            for u, v in tree_edges:
                hu, hv = assignment[u], assignment[v]
                if (hu, hv) in used_edges or (hv, hu) in used_edges:
                    return None
            return [assignment[i] for i in range(k)]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    trees = _instance_trees(20, 611)
    
    # Initial best: just T20
    best_result = [None] * 19 + [list(range(20))]
    best_value = 19
    
    # Try multiple trials with different random seeds
    # Strategy: embed trees in descending order of size (harder first)
    # Actually, let's try both ascending and descending
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        rng = Random(trial * 7919 + 12345)
        used_edges = set()
        result = [None] * 20
        total_edges = 0
        success = True
        
        # Decide order: alternate between ascending and descending
        if trial % 2 == 0:
            order = list(range(1, 21))  # ascending
        else:
            order = list(range(20, 0, -1))  # descending
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
            
            tree_edges = trees[k - 1]
            if not tree_edges:
                # T1: single vertex
                # Pick a random available vertex
                avail = [v for v in range(20)]
                result[k - 1] = [rng.choice(avail)]
                continue
            
            embedding = _embed_tree_backtrack(tree_edges, k, used_edges, rng, max_attempts=100)
            if embedding is None:
                success = False
                break
            
            result[k - 1] = embedding
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_edges.add((hu, hv))
                used_edges.add((hv, hu))
            total_edges += len(tree_edges)
        
        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break
    
    # Ensure types are correct
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    
    return final
# EVOLVE-BLOCK-END
