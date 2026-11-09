import time
from random import Random

def _gen_trees(n=20, seed=611):
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    N = 20
    trees = _gen_trees(N, 611)
    
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_info.append({
            'k': k,
            'edges': edges,
            'deg': deg,
            'adj': adj,
            'nedges': len(edges)
        })
        
    order = sorted(range(N), key=lambda i: (-tree_info[i]['k'], -max(tree_info[i]['deg'])))
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_result = [None] * N
    best_value = 0
    used_edges = set()
    result = [None] * N
    
    rng = Random(42) # Fixed seed for reproducibility of search heuristics

    def value_of(res):
        return sum(tree_info[i]['nedges'] for i in range(N) if res[i] is not None)

    def embedding_generator(tree_idx):
        info = tree_info[tree_idx]
        k = info['k']
        degs = info['deg']
        adj = info['adj']
        
        if k == 1:
            # Any host vertex works. Return all permutations of [0]?
            # Just return [0] to keep it simple, or randomize?
            # Since vertices are symmetric in K_N, [0] is fine.
            yield [0], []
            return
            
        perm = sorted(range(k), key=lambda i: -degs[i])
        mapping = [-1] * k
        host_used = [False] * N
        
        def dfs(pos):
            if pos == k:
                edges_to_add = []
                for u, v in info['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    edges_to_add.append((min(hu, hv), max(hu, hv)))
                yield mapping[:], edges_to_add
                return
            
            v = perm[pos]
            candidates = []
            for h in range(N):
                if host_used[h]:
                    continue
                valid = True
                for u in adj[v]:
                    if mapping[u] != -1:
                        hu = mapping[u]
                        e = (min(h, hu), max(h, hu))
                        if e in used_edges:
                            valid = False
                            break
                if valid:
                    candidates.append(h)
                    
            if not candidates:
                return

            # Shuffle candidates to avoid deterministic bias
            rng.shuffle(candidates)
            
            for h in candidates:
                mapping[v] = h
                host_used[h] = True
                
                yield from dfs(pos + 1)
                
                host_used[h] = False
                mapping[v] = -1
                
        yield from dfs(0)

    def solve(idx_in_order):
        nonlocal best_result, best_value
        
        if time.time() > deadline:
            return False
            
        if idx_in_order == len(order):
            val = value_of(result)
            if val > best_value:
                best_value = val
                best_result = [x[:] if x is not None else None for x in result]
            return True
            
        tree_idx = order[idx_in_order]
        
        # To prevent infinite loops if a tree has too many embeddings,
        # we could limit the count, but we need to find a solution.
        # The pruning by used_edges should be strong.
        
        for mapping, edges_to_add in embedding_generator(tree_idx):
            # Check time again before committing
            if time.time() > deadline:
                return False
                
            for e in edges_to_add:
                used_edges.add(e)
            result[tree_idx] = [int(x) for x in mapping]
            
            if solve(idx_in_order + 1):
                return True
                
            for e in edges_to_add:
                used_edges.remove(e)
            result[tree_idx] = None
            
        return False

    solve(0)
    
    if best_result[0] is None:
        best_result[0] = [0]
        
    return best_result
