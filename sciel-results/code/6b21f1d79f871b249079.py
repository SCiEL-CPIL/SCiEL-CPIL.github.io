import time
import random
from random import Random

def instance_trees(n=20, seed=611):
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

def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_bfs_order(adj, k):
    """
    Returns a list of vertices in BFS order starting from 0.
    This ensures parents are placed before children.
    """
    if k == 1:
        return [0]
    visited = [False] * k
    order = [0]
    visited[0] = True
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                order.append(v)
                queue.append(v)
    return order

def construct():
    deadline = time.time() + 140  # 10s margin
    trees = instance_trees(20, 611)
    n = 20
    num_hosts = 20
    
    # Precompute tree structures
    tree_adjs = [_build_adjacency(trees[k-1], k) for k in range(1, n + 1)]
    tree_orders = [_get_bfs_order(tree_adjs[k-1], k) for k in range(1, n + 1)]
    
    # We want to pack larger trees first.
    # Indices 0..19 correspond to T1..T20.
    # T_k has k-1 edges.
    # Sort by size descending.
    embed_order = list(range(n, 0, -1)) # 20, 19, ..., 1
    
    best_result = [None] * n
    best_score = 0
    
    def try_pack(seed_offset):
        """
        Attempt to pack all trees. Returns (success, result_list, score).
        """
        rng = Random(seed_offset)
        used_edges = set()
        result = [None] * n
        
        # Map from tree index (1-based) to embedding
        embeddings = {}
        
        # We process in embed_order
        # For backtracking, we might need to undo, but since we process one tree at a time
        # and only commit if the whole tree is embedded, we can just do a DFS for each tree.
        # If a tree fails, we return failure for this whole attempt (or skip it? 
        # The problem says "Entry k-1 is None or...". It implies we can leave some None.
        # But to maximize score, we should try to pack as many as possible.
        # However, if we fail T20, we can still try T19, etc.
        # But if we skip T20, we might block T19? No, T19 is smaller.
        # Actually, if we fail T20, we just leave it None and move to T19.
        # This greedy approach might be suboptimal (e.g. T20 takes an edge needed by T19+T18).
        # But T20 has 19 edges, T19 has 18.
        # A perfect packing uses all 190 edges.
        # If we can't pack T20, we lose 19 edges.
        # If we skip T20, we have 171 edges left for 19 trees.
        # Sum of edges T1..T19 = 171.
        # So if we fail T20, we *must* pack all others perfectly to get 171.
        # If we pack T20, we have 171 edges left for T1..T19.
        # So packing larger trees first is crucial.
        
        def embed_tree(k):
            """
            Try to embed tree T_k into K_20 avoiding used_edges.
            Returns mapping list or None.
            """
            if k == 1:
                # T1 has no edges. Can be mapped to any vertex.
                # To be safe and distinct from others? 
                # "list of k distinct int host vertices".
                # T1 is just 1 vertex.
                # We should pick a vertex that doesn't cause issues? 
                # T1 has no edges, so it doesn't consume any edges.
                # It just occupies a vertex.
                # But vertices can be shared? "Embedded trees must be edge-disjoint".
                # Vertices CAN be shared.
                # So T1 can be [0].
                return [0]
            
            if k == 2:
                # T2 is an edge.
                # Find an unused edge.
                for a in range(num_hosts):
                    for b in range(a + 1, num_hosts):
                        if (a, b) not in used_edges:
                            return [a, b]
                return None

            adj = tree_adjs[k-1]
            order = tree_orders[k-1]
            mapping = [None] * k
            
            # Precompute neighbors in order
            # For each vertex in order, we need to know which neighbors are "previous" in the order.
            # Since it's BFS, parents are previous. Children are later.
            # So for vertex v at index i in order, its neighbors that are already placed are its parents.
            # In a tree, each vertex (except root) has exactly one parent in the BFS tree.
            # However, the tree structure is given by edges.
            # We can just check all neighbors and see if they are mapped.
            
            def dfs(idx):
                if idx == k:
                    return True
                
                v = order[idx]
                
                # Find placed neighbors
                placed_neighbors = []
                for nb in adj[v]:
                    if mapping[nb] is not None:
                        placed_neighbors.append(nb)
                
                # Candidate hosts
                # 1. Not used by current tree
                used_in_tree = set(mapping[i] for i in range(k) if mapping[i] is not None)
                
                # 2. Check edge conflicts with placed neighbors
                candidates = []
                for h in range(num_hosts):
                    if h in used_in_tree:
                        continue
                    ok = True
                    for nb in placed_neighbors:
                        h_nb = mapping[nb]
                        e_key = (min(h, h_nb), max(h, h_nb))
                        if e_key in used_edges:
                            ok = False
                            break
                    if ok:
                        candidates.append(h)
                        
                if not candidates:
                    return False
                
                # Shuffle for randomization
                rng.shuffle(candidates)
                
                for h in candidates:
                    mapping[v] = h
                    if dfs(idx + 1):
                        return True
                    mapping[v] = None
                    
                return False
            
            if dfs(0):
                return mapping
            return None
        
        # Process trees
        for k in embed_order:
            if time.time() > deadline:
                break
            
            if k == 1:
                result[k-1] = [0]
                continue
                
            mapping = embed_tree(k)
            if mapping is not None:
                result[k-1] = mapping
                # Add edges to used_edges
                for u, v in trees[k-1]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                result[k-1] = None
                # If we fail a large tree, we might still try smaller ones.
                # But if we fail T20, we have 171 edges left.
                # We should continue trying T19, etc.
                
        # Calculate score
        score = 0
        for k in range(1, n + 1):
            if result[k-1] is not None:
                score += len(trees[k-1])
                
        return result, score

    # Try multiple seeds to find a good packing
    # The first few seeds are often enough if the algorithm is robust.
    # We will try 10 restarts.
    
    for i in range(20):
        if time.time() > deadline:
            break
        result, score = try_pack(611 + i)
        if score > best_score:
            best_score = score
            best_result = result
            
        # If we found a perfect packing, stop
        if best_score == 190:
            break
            
    return best_result

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k - 1] is not None:
            r = res[k - 1]
            if not isinstance(r, list):
                print(f"T{k}: not a list")
                ok = False
                continue
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v!r}")
                    ok = False
            for (u, v) in trees[k - 1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k - 1]) for k in range(1, 21) if res[k - 1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
