"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

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
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    
    def solve_tree_csp(tree_edges, k, used_edges, deadline, rng):
        """
        CSP-based solver for embedding a tree into K20.
        Uses DFS on tree structure with forward checking.
        """
        if k == 1:
            # Find any unused vertex
            for h in range(20):
                return [h]
            return None
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None

        # Build tree adjacency
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Root the tree at vertex 0, get parent and BFS order
        parent = [-1] * k
        order = []
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        # For each vertex, get its children
        children = [[] for _ in range(k)]
        for v in range(k):
            if parent[v] != -1:
                children[parent[v]].append(v)
        
        host = [None] * k
        used_hosts = set()
        
        # Precompute for each tree vertex, which host vertices are available
        # given the used_edges constraint with its parent
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            v = order[idx]
            p = parent[v]
            hp = host[p] if p != -1 else None
            
            # Find valid candidates for v
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                if hp is not None:
                    edge = (min(hp, h), max(hp, h))
                    if edge in used_edges:
                        continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            # Heuristic: prefer candidates that leave more options for children
            # Score each candidate by how many of its neighbors (in host graph) 
            # are still available for the children
            def score_candidate(h):
                if not children[v]:
                    return 0
                # Count available neighbors of h that could serve children
                count = 0
                for ch in children[v]:
                    # For each child, check if h has available neighbors
                    # that are not in used_hosts and not blocked by used_edges
                    for nh in range(20):
                        if nh == h or nh in used_hosts:
                            continue
                        edge = (min(h, nh), max(h, nh))
                        if edge not in used_edges:
                            count += 1
                            break
                return count
            
            # Sort candidates: try those with higher score first (more flexibility)
            # Actually, for constraint propagation, we want to try candidates that
            # are most constrained first. But here, we want to leave options for children.
            # Let's try both: randomize with slight preference for high-degree host vertices
            rng.shuffle(candidates)
            
            for h in candidates:
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
            
            return False
        
        # Try multiple times with different randomizations
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Reset
            host = [None] * k
            used_hosts = set()
            if backtrack(0):
                return host
        return None

    def solve_tree_csp_fast(tree_edges, k, used_edges, deadline, rng):
        """
        Faster solver using greedy + limited backtracking.
        """
        if k == 1:
            for h in range(20):
                return [h]
            return None
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None

        # Build tree adjacency
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Find a centroid-like root to balance the tree
        # For simplicity, root at 0
        parent = [-1] * k
        order = []
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        host = [None] * k
        used_hosts = set()
        
        # Greedy with backtracking
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            v = order[idx]
            p = parent[v]
            hp = host[p] if p != -1 else None
            
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                if hp is not None:
                    edge = (min(hp, h), max(hp, h))
                    if edge in used_edges:
                        continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            # Try candidates in random order
            rng.shuffle(candidates)
            
            for h in candidates:
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
            
            return False
        
        # Try a few times
        for attempt in range(50):
            if time.time() > deadline:
                break
            host = [None] * k
            used_hosts = set()
            if backtrack(0):
                return host
        return None

    best_score = 0
    best_embeddings = [None] * 20
    
    # T1 always maps to a single vertex
    best_embeddings[0] = [0]
    
    # Strategy: Try different orders of embedding trees
    # Order 1: T20 first (biggest), then T19, T18, ...
    # Order 2: Smaller first, T20 last
    # Order 3: Alternating
    
    orders = [
        list(range(19, 0, -1)),  # T20, T19, ..., T2
        list(range(1, 20)),      # T2, T3, ..., T20
        [19] + list(range(1, 19)),  # T20, T2, T3, ..., T19
    ]
    
    for order_idx, embed_order in enumerate(orders):
        for restart in range(30):
            if time.time() > deadline - 2:
                break
                
            rng = Random(1234 + order_idx * 1000 + restart)
            used_edges = set()
            embeddings = [None] * 20
            embeddings[0] = [0]  # T1
            
            success = True
            for k_idx in embed_order:
                k = k_idx + 1  # k_idx is 0-based for T2..T20, so k_idx=19 is T20
                if time.time() > deadline - 2:
                    success = False
                    break
                
                # Choose solver based on tree size
                if k >= 15:
                    res = solve_tree_csp(trees[k_idx], k, used_edges, deadline, rng)
                else:
                    res = solve_tree_csp_fast(trees[k_idx], k, used_edges, deadline, rng)
                
                if res is None:
                    success = False
                    break
                
                embeddings[k_idx] = res
                for u, v in trees[k_idx]:
                    used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
                    
            if success:
                score = 0
                for k in range(1, 21):
                    if embeddings[k-1] is not None:
                        score += k - 1
                if score > best_score:
                    best_score = score
                    best_embeddings = embeddings[:]
                    
                if best_score == 190:
                    break
        
        if best_score == 190:
            break
    
    # Fallback: ensure we have at least T1 and T20
    if best_embeddings[19] is None:
        # Try identity mapping for T20
        used_edges = set()
        for k in range(2, 20):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        # Try to find a valid embedding for T20
        for perm_seed in range(100):
            if time.time() > deadline - 1:
                break
            rng2 = Random(9999 + perm_seed)
            perm = list(range(20))
            rng2.shuffle(perm)
            
            valid = True
            for u, v in trees[19]:
                edge = (min(perm[u], perm[v]), max(perm[u], perm[v]))
                if edge in used_edges:
                    valid = False
                    break
            
            if valid:
                best_embeddings[19] = perm
                best_score += 19
                break
    
    # Try to embed T2 if not already done
    if best_embeddings[1] is None:
        used_edges = set()
        for k in range(2, 21):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        for u in range(20):
            for v in range(u+1, 20):
                if (u, v) not in used_edges:
                    best_embeddings[1] = [u, v]
                    best_score += 1
                    break
            if best_embeddings[1] is not None:
                break
    
    # Final validation
    if best_embeddings[0] is None:
        best_embeddings[0] = [0]
        
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None
            else:
                if len(set(best_embeddings[i])) != i + 1:
                    best_embeddings[i] = None
                # Ensure all are Python ints
                for j in range(len(best_embeddings[i])):
                    best_embeddings[i][j] = int(best_embeddings[i][j])

    return best_embeddings

# EVOLVE-BLOCK-END
