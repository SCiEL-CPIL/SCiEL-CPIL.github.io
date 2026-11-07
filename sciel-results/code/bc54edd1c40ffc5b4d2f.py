"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random, sample
from itertools import combinations

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
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Order trees: largest first for constraint propagation
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: (-(len(trees[idx])), -idx))
    
    # Precompute tree structures
    tree_data = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_data.append(None)
            continue
        tree_verts = set()
        for u, v in tree_edges:
            tree_verts.add(u)
            tree_verts.add(v)
        tree_verts = sorted(tree_verts)
        tree_adj = {v: [] for v in tree_verts}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order by degree descending
        ordered_tv = sorted(tree_verts, key=lambda v: -len(tree_adj[v]))
        tree_data.append((tree_edges, tree_verts, tree_adj, ordered_tv, k))
    
    def try_embed_fast(t_idx, available_edges, rng):
        """Try to embed tree t_idx into available_edges using backtracking with pruning."""
        if t_idx == 0:  # k=1
            return [rng.randint(0, n - 1)]
        
        tree_edges, tree_verts, tree_adj, ordered_tv, k = tree_data[t_idx]
        
        # Build candidate lists for each tree vertex based on available edges
        # For a tree vertex v, candidates are host vertices h such that
        # h has edges to all already-assigned neighbors of v
        
        host_map = {}
        used_hosts = set()
        
        # Precompute: for each host vertex, which other host vertices it's connected to in available_edges
        host_adj = {h: set() for h in range(n)}
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)
        
        # For ordering: try vertices with higher degree first
        # But also consider: if a tree vertex has many neighbors, it constrains more
        
        def backtrack(pos):
            if pos == len(ordered_tv):
                # Verify all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True
            
            tv = ordered_tv[pos]
            tv_degree = len(tree_adj[tv])
            
            # Get already-assigned neighbors of tv
            assigned_neighbors = [(nb, host_map[nb]) for nb in tree_adj[tv] if nb in host_map]
            
            # Find candidate host vertices
            if assigned_neighbors:
                # Candidate must be connected to all assigned neighbors
                candidate_sets = [host_adj[hn] for _, hn in assigned_neighbors]
                candidates = set(candidate_sets[0])
                for cs in candidate_sets[1:]:
                    candidates &= cs
            else:
                candidates = set(range(n))
            
            # Remove used hosts
            candidates -= used_hosts
            
            # Sort candidates: prefer those with higher degree in available_edges
            # This helps with constraint propagation
            candidates_list = sorted(candidates, key=lambda h: -len(host_adj[h]))
            
            # Randomize slightly for diversity
            if len(candidates_list) > 1:
                # Shuffle a random subset
                rng.shuffle(candidates_list)
            
            for h in candidates_list:
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
    
    best_result = None
    
    for attempt in range(5000):
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
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
            
            embedded = try_embed_fast(t_idx, available, rng)
            
            if embedded is None:
                success = False
                break
            
            result[t_idx] = embedded
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
        
        if success:
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190 and len(available) == 0:
                best_result = result
                break
    
    if best_result is None:
        return [None] * 19 + [list(range(20))]
    
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
