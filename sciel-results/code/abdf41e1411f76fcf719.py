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
    
    # Precompute all K20 edges
    K20_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            K20_edges.add((u, v))
            
    def solve_tree(tree_edges, k, used_edges, deadline, rng):
        """
        Try to embed a tree with k vertices into K20 avoiding used_edges.
        Returns a list of k distinct host vertices, or None.
        """
        if k == 1:
            # T1 is empty, but if called, map to any vertex? 
            # The contract says T1 is empty. But if we need to map a vertex, it's trivial.
            # However, T1 has 0 edges. The entry is [] or list of 1 vertex?
            # "Entry k-1 is None or a list of k distinct int host vertices"
            # For k=1, list of 1 vertex.
            # But T1 has no edges, so any single vertex works.
            return [0]
            
        if k == 2:
            # T2 is a single edge.
            # Find any unused edge.
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None

        # Build adjacency list
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS order starting from 0
        visited = [False] * k
        order = [0]
        visited[0] = True
        queue = [0]
        parent = [-1] * k
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    parent[v] = u
                    queue.append(v)
                    
        host = [None] * k
        host[0] = 0
        used_hosts = {0}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                # All vertices assigned. Verify edges are not used.
                # We checked incrementally, but just in case.
                for u, v in tree_edges:
                    hu, hv = host[u], host[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return False
                return True
            
            v = order[idx]
            # Candidates: host vertices not yet used
            # Heuristic: prioritize vertices that connect to already assigned neighbors
            # Actually, v's parent is already assigned.
            # So we MUST connect to host[parent[v]].
            # Let's check if the edge to parent is available.
            # If not, this assignment is invalid? No, the parent's host is fixed.
            # So the edge (host[parent[v]], h) must NOT be in used_edges.
            
            p = parent[v]
            hp = host[p]
            
            candidates = [h for h in range(20) if h not in used_hosts]
            # Sort candidates? Maybe by degree in remaining graph?
            # Or just randomize.
            rng.shuffle(candidates)
            
            for h in candidates:
                # Check edge to parent
                edge_to_parent = (min(hp, h), max(hp, h))
                if edge_to_parent in used_edges:
                    continue
                
                # Check edges to any other already assigned neighbors?
                # In a tree, v only has one parent in the BFS tree.
                # But it might have other neighbors in the tree that are already assigned?
                # No, BFS ensures that all neighbors except parent are unvisited (children) 
                # OR visited but not parent? 
                # In a tree, a node's neighbors are its parent and its children.
                # In BFS order, children are visited AFTER the node.
                # So when we assign v, its children are NOT assigned yet.
                # Its parent IS assigned.
                # Are there any other assigned neighbors?
                # Only if the tree had cycles, but it's a tree.
                # So only the parent edge needs checking against used_edges.
                # Wait, what if the tree is not rooted at 0? 
                # We rooted it at 0. BFS from 0.
                # So for any v != 0, parent[v] is assigned.
                # Children of v are not assigned.
                # So only edge to parent matters.
                
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
                
            return False
            
        if backtrack(1):
            return host
        return None

    # Strategy:
    # 1. T1 is trivial: []
    # 2. T20 is a spanning tree. Embed it first? Or last?
    #    If we embed T20 first, we consume 19 edges.
    #    Remaining 171 edges for T1..T19 (171 edges total).
    #    This is a tight packing.
    #    If we embed T20 last, we need the remaining graph to contain T20.
    #    Let's try embedding T20 first to secure the biggest tree.
    #    Then embed T19, T18, ...
    
    best_score = 0
    best_embeddings = [None] * 20
    
    # T1 always maps to a single vertex, e.g., 0. It has no edges, so it doesn't conflict.
    # The contract requires a list of k distinct int host vertices. For k=1, this is [v].
    best_embeddings[0] = [0]
    
    # We will try a few restarts with different random seeds for the solver
    for restart in range(50):
        if time.time() > deadline - 2:
            break
            
        rng = Random(1234 + restart)
        used_edges = set()
        embeddings = [None] * 20
        
        # Embed T20
        # T20 has 20 vertices. It must use all 20 host vertices?
        # "list of k distinct int host vertices in 0..19"
        # Yes, distinct. So T20 uses all 20 vertices.
        # This means T20 is a spanning tree of K20.
        # The edges used by T20 are a subset of K20 edges.
        
        res20 = solve_tree(trees[19], 20, used_edges, deadline, rng)
        if res20 is None:
            continue
        embeddings[19] = res20
        for u, v in trees[19]:
            used_edges.add((min(res20[u], res20[v]), max(res20[u], res20[v])))
            
        success = True
        # Embed T19 down to T2
        for k in range(19, 1, -1):
            if time.time() > deadline - 2:
                success = False
                break
            res = solve_tree(trees[k-1], k, used_edges, deadline, rng)
            if res is None:
                success = False
                break
            embeddings[k-1] = res
            for u, v in trees[k-1]:
                used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
                
        if success:
            embeddings[0] = [0] # T1
            # Calculate score
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
                
            if best_score == 190:
                break # Perfect packing

    # If we didn't get a perfect packing, try a different strategy:
    # Maybe embed smaller trees first?
    # Or just return the best we have.
    
    # Fallback: If best_score is 0 (or low), ensure we have a valid partial solution
    # T1 is already set to [0]
    
    # If T20 wasn't embedded in the main loop (best_score < 19 implies T20 might be missing or partial failed)
    # Actually, the main loop breaks if T20 fails. So if best_score is 0, T20 failed in all restarts?
    # Or maybe T20 succeeded but T19 failed, so success was False, so best_embeddings wasn't updated from restarts?
    # Wait, if success is False, we don't update best_embeddings.
    # So if best_score is 0, best_embeddings is still the initial [0] + Nones.
    
    if best_embeddings[19] is None:
        # Try to embed T20 with a simple greedy/heuristic if backtracking failed or wasn't run?
        # The backtracking in solve_tree is quite strong for T20 on K20.
        # If it failed, maybe the seed was bad?
        # Let's just try identity mapping for T20 as a last resort fallback to get *some* score.
        # But we must ensure edge disjointness with others. Since others are None, it's safe.
        best_embeddings[19] = list(range(20))
        best_score += 19
        
    # Try to embed T2 if not already done
    if best_embeddings[1] is None:
        # Collect used edges from any existing embeddings
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

    # Final validation and cleanup
    # Ensure T1 is valid
    if best_embeddings[0] is None:
        best_embeddings[0] = [0]
        
    # Verify lengths and types
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None # Invalid length, discard
            else:
                # Ensure they are distinct ints
                if len(set(best_embeddings[i])) != i + 1:
                    best_embeddings[i] = None

    return best_embeddings

# EVOLVE-BLOCK-END
