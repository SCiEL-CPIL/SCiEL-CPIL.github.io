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
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n = 20
    
    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Precompute tree info
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k <= 2:
            tree_info.append((tree_edges, None, None, k))
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order by degree descending
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((tree_edges, tree_adj, ordered_verts, k))
    
    def try_embed_deterministic(t_idx, available, rng):
        """Try to embed tree t_idx into available edges using backtracking."""
        tree_edges, tree_adj, ordered_verts, k = tree_info[t_idx]
        
        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for edge in available:
                return [edge[0], edge[1]]
            return None
        
        host_map = {}
        used_hosts = set()
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available:
                        return False
                return True
            
            tv = ordered_verts[pos]
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available:
                            ok = False
                            break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort candidates by degree in available graph (prefer high-degree)
            def candidate_score(h):
                count = 0
                for nb_h in range(n):
                    if nb_h == h:
                        continue
                    edge = (min(h, nb_h), max(h, nb_h))
                    if edge in available:
                        count += 1
                return -count  # prefer higher degree
            
            candidates.sort(key=candidate_score)
            # Add some randomness
            if rng.random() < 0.3:
                rng.shuffle(candidates)
            
            for h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)
            
            return False
        
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None
    
    # Process trees from largest to smallest
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: -(idx + 1))
    
    best_result = None
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 17 + 13)
        avail = set(all_edges)
        res = [None] * 20
        success = True
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = tree_info[t_idx][3]
            
            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue
            
            embedded = None
            for trial in range(8):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed_deterministic(t_idx, avail, rng)
                if embedded is not None:
                    break
            
            if not success:
                break
            if embedded is None:
                success = False
                break
            
            res[t_idx] = embedded
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)
        
        if success and len(avail) == 0:
            best_result = res
            break
    
    if best_result is None:
        # Fallback: try a simpler approach
        return [None] * 19 + [list(range(20))]
    
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
