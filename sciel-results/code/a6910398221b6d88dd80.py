"""Packing trees T1..T20 into K20 - improved backtracking with better pruning."""
import time
from random import Random
from itertools import product

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Precompute tree structures
    tree_info = []
    for k_idx in range(20):
        k = k_idx + 1
        if k == 1:
            tree_info.append({
                'k': 1,
                'edges': [],
                'adj': {},
                'verts': [0],
                'order': [0],
                'deg': [0]
            })
            continue
        edges = trees[k_idx]
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        verts = sorted(verts)
        adj = {v: [] for v in verts}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        deg = [len(adj[v]) for v in verts]
        # Order by degree descending, then by index for determinism
        order = sorted(verts, key=lambda v: (-len(adj[v]), v))
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'verts': verts,
            'order': order,
            'deg': deg
        })
    
    # Sort tree indices by difficulty: most edges first
    tree_order = sorted(range(20), key=lambda idx: (-(len(trees[idx])), -idx))
    
    def embed_tree(t_idx, available, rng, max_tries=200):
        """Try to embed tree t_idx into available edges.
        Returns list of host vertices (length k) or None."""
        info = tree_info[t_idx]
        k = info['k']
        if k == 1:
            return [rng.randint(0, n - 1)]
        
        edges = info['edges']
        adj = info['adj']
        order = info['order']
        
        for trial in range(max_tries):
            if time.time() > deadline:
                return None
            
            # Randomize order within same degree groups
            deg_groups = {}
            for v in order:
                d = len(adj[v])
                if d not in deg_groups:
                    deg_groups[d] = []
                deg_groups[d].append(v)
            
            randomized_order = []
            for d in sorted(deg_groups.keys(), reverse=True):
                group = deg_groups[d][:]
                rng.shuffle(group)
                randomized_order.extend(group)
            
            host_map = {}
            used_hosts = set()
            
            # Precompute for each tree vertex in order: which neighbors come earlier in order
            earlier_neighbors = {}
            seen_in_order = set()
            for tv in randomized_order:
                earlier_neighbors[tv] = [nb for nb in adj[tv] if nb in seen_in_order]
                seen_in_order.add(tv)
            
            # Backtracking
            def bt(pos):
                if pos == len(randomized_order):
                    # All vertices assigned - verify all edges
                    for u, v in edges:
                        hu, hv = host_map[u], host_map[v]
                        edge = (min(hu, hv), max(hu, hv))
                        if edge not in available:
                            return False
                    return True
                
                tv = randomized_order[pos]
                # Get candidate hosts: must be connected to all earlier-assigned neighbors
                candidates = []
                for h in range(n):
                    if h in used_hosts:
                        continue
                    ok = True
                    for nb in earlier_neighbors[tv]:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available:
                            ok = False
                            break
                    if ok:
                        candidates.append(h)
                
                # Shuffle candidates for randomization
                rng.shuffle(candidates)
                
                for h in candidates:
                    host_map[tv] = h
                    used_hosts.add(h)
                    if bt(pos + 1):
                        return True
                    del host_map[tv]
                    used_hosts.remove(h)
                
                return False
            
            if bt(0):
                result = [host_map[i] for i in range(k)]
                return result
        
        return None
    
    # Main search with randomized restarts
    best_result = None
    total_tree_edges = sum(len(trees[i]) for i in range(20))  # should be 190
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 13)
        
        available = set(all_edges)
        result = [None] * 20
        success = True
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            
            embedded = embed_tree(t_idx, available, rng)
            
            if embedded is None:
                success = False
                break
            
            result[t_idx] = embedded
            
            # Remove used edges
            info = tree_info[t_idx]
            for u, v in info['edges']:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
        
        if success:
            # Verify: all 190 edges should be used
            if len(available) == 0:
                best_result = result
                break
            else:
                # All trees embedded but some edges remain - still valid (edge-disjoint)
                # But we want to maximize edges used, so keep searching
                # Actually, the task says raw value is total number of embedded tree edges
                # Each tree T_k has k-1 edges, so total is always 190 if all embedded
                # So this is a valid complete solution
                best_result = result
                break
    
    if best_result is None:
        # Fallback: try a more systematic approach
        # Process trees in a different order
        for alt_order in [
            sorted(range(20), key=lambda idx: (len(trees[idx]), idx)),  # smallest first
            list(range(20)),  # natural order
            sorted(range(20), key=lambda idx: (-tree_info[idx]['k'], idx)),  # by k
        ]:
            if time.time() > deadline:
                break
            
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                
                rng = Random(attempt * 104729 + 7)
                available = set(all_edges)
                result = [None] * 20
                success = True
                
                for t_idx in alt_order:
                    if time.time() > deadline:
                        success = False
                        break
                    
                    embedded = embed_tree(t_idx, available, rng, max_tries=100)
                    
                    if embedded is None:
                        success = False
                        break
                    
                    result[t_idx] = embedded
                    
                    info = tree_info[t_idx]
                    for u, v in info['edges']:
                        hu, hv = embedded[u], embedded[v]
                        edge = (min(hu, hv), max(hu, hv))
                        available.discard(edge)
                
                if success:
                    best_result = result
                    break
            
            if best_result is not None:
                break
    
    if best_result is None:
        # Last resort: return T1 as [0] and rest as None
        # This gives at least 1 edge (T2) if we can embed it
        # Actually, let's try to embed as many as possible greedily
        result = [None] * 20
        available = set(all_edges)
        
        for t_idx in tree_order:
            if time.time() > deadline:
                break
            rng = Random(42)
            embedded = embed_tree(t_idx, available, rng, max_tries=50)
            if embedded is not None:
                result[t_idx] = embedded
                info = tree_info[t_idx]
                for u, v in info['edges']:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    available.discard(edge)
        
        best_result = result
    
    # Ensure types are correct
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
