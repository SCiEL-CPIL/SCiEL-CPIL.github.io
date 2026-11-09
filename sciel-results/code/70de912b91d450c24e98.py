import sys
import time
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

def construct():
    trees = instance_trees(20, 611)
    n = 20
    # Pre-compute all edges in K20
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add((u, v))
            
    # We want to pack trees T1..T20.
    # Total edges 190.
    # Strategy: Backtracking search.
    # Order of trees: Descending size (T20, T19, ..., T1).
    # This is usually better for packing as larger structures are harder to place.
    
    tree_indices = list(range(1, 21)) # 1 to 20
    # Sort by size descending
    tree_indices.sort(key=lambda k: len(trees[k-1]), reverse=True)
    
    def solve():
        used_edges = set()
        embeddings = [None] * 20
        
        # We will embed trees in the order of tree_indices
        # But we need to return in order 1..20.
        
        def embed_tree(k):
            edges_k = trees[k-1]
            if not edges_k:
                return [0] # T1
            if k == 2:
                # Find any unused edge
                for e in all_edges:
                    if e not in used_edges:
                        used_edges.add(e)
                        return [e[0], e[1]]
                return None
            
            # Build adjacency for the tree
            adj = [[] for _ in range(k)]
            for u, v in edges_k:
                adj[u].append(v)
                adj[v].append(u)
            
            # Order vertices by degree descending for better pruning?
            # Actually, we need to map tree vertices to host vertices.
            # Let's use a recursive DFS on the tree structure.
            # Root the tree at vertex 0.
            # We need to assign host vertices to tree vertices 0..k-1.
            
            # Heuristic: Try to place high-degree tree vertices first?
            # No, we must respect the tree structure.
            # Let's do a DFS from root 0.
            
            # To speed up, we can pre-compute a "good" order of neighbors.
            # But the host graph is dense (K20).
            
            # Let's try a randomized DFS with backtracking.
            # Since K20 is dense, conflicts are rare until the end.
            # The main constraint is edge-disjointness.
            
            # We'll use a stack for DFS.
            # State: current tree vertex, parent tree vertex, assigned hosts so far.
            
            # Optimization: If we fail, we need to backtrack.
            # Pure Python recursion with backtracking can be slow if not careful.
            # But k <= 20.
            
            # Let's try a simple backtracking solver.
            # We'll pick the next tree vertex to place based on the current partial mapping.
            
            # Actually, a better approach for dense graphs:
            # Randomly permute the host vertices for the root, then greedily place children.
            # If greedy fails, backtrack.
            
            # Let's implement a recursive function.
            
            def dfs(tree_node, parent_host, assigned_hosts, mapping):
                # tree_node: index in tree
                # parent_host: host index of parent in tree
                # assigned_hosts: set of host indices used
                # mapping: list of length k, mapping[i] is host index or -1
                
                # All tree vertices assigned?
                if len(assigned_hosts) == k:
                    # Check if all edges are valid (they are by construction)
                    # And distinct (ensured by assigned_hosts)
                    return True
                
                # Get neighbors of tree_node in the tree
                neighbors = adj[tree_node]
                
                # Filter out parent
                children = [n for n in neighbors if n != parent_host]
                
                # If we are at a leaf, we just need to assign a host.
                # If we are at an internal node, we need to assign a host and then recurse.
                
                # Which host to try?
                # Candidates: hosts h in 0..19 such that h not in assigned_hosts
                # and edge (parent_host, h) is not in used_edges (if parent exists)
                
                # Wait, the edge (parent_tree_node, tree_node) maps to (parent_host, h).
                # So we must ensure (parent_host, h) is not in used_edges.
                
                # Also, we must ensure that the subtree rooted at tree_node can be embedded.
                # This is the hard part (forward checking).
                
                # For now, simple backtracking.
                
                # Heuristic: Try hosts that have many available edges to other unplaced nodes?
                # Or just random order with a fixed seed for determinism in a restart?
                
                # Let's try all available hosts in a random order.
                candidates = []
                for h in range(n):
                    if h in assigned_hosts:
                        continue
                    if parent_host is not None:
                        edge = (min(parent_host, h), max(parent_host, h))
                        if edge in used_edges:
                            continue
                    candidates.append(h)
                    
                # Shuffle candidates for randomization
                # But we need to be fast.
                # Let's sort candidates by some heuristic.
                # Heuristic: Prefer hosts with higher degree in the *remaining* graph?
                # Since K20 is regular, this is not helpful.
                # Let's just shuffle.
                
                # To make it deterministic per restart, we pass a rng.
                # But this is inside a nested function, so we need to capture rng.
                
                # Let's restructure:
                # We will call a top-level solver that manages restarts.
                
                return False # Placeholder
                
            # This simple DFS might be too slow if it backtracks a lot.
            # However, for k=20, the branching factor is ~19, then ~18, etc.
            # 19! is huge. But constraints prune it.
            # In K20, the constraint is just edge availability.
            # Since we are packing into K20, and the total edges match, it's a tight packing.
            # Backtracking is necessary.
            
            return None # Placeholder for actual implementation

        # The above is a sketch. Let's implement a more robust solver.
        
        # We'll use a "coloring" approach or just smart backtracking.
        # Given the time limit, I will implement a backtracking solver with:
        # 1. Vertex ordering by degree (in the tree) to place high-degree nodes early.
        # 2. Host selection heuristic: prefer hosts with more available edges to unplaced neighbors.
        
        # Actually, the simplest robust method for this specific problem (packing into K_n)
        # is often to use a SAT solver or ILP, but we can't use external libs.
        
        # Let's try a different angle:
        # Since the trees are small (max 20 vertices), and K20 is dense,
        # a randomized greedy approach with a few restarts usually works.
        # The current program does this.
        
        # To improve, I will use a **deterministic** backtracking solver with a good heuristic.
        # If it fails, I will try a few random restarts.
        
        def solve_deterministic():
            used_edges = set()
            embeddings = [None] * 20
            
            # Process trees in descending order of size
            for k in sorted(range(1, 21), key=lambda x: len(trees[x-1]), reverse=True):
                edges_k = trees[k-1]
                if not edges_k:
                    embeddings[k-1] = [0]
                    continue
                if k == 2:
                    for u in range(n):
                        for v in range(u+1, n):
                            if (u, v) not in used_edges:
                                embeddings[k-1] = [u, v]
                                used_edges.add((u, v))
                                break
                        if embeddings[k-1]:
                            break
                    if not embeddings[k-1]:
                        return None
                    continue
                
                # Build tree adjacency
                adj = [[] for _ in range(k)]
                for u, v in edges_k:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Find a mapping
                # We will use a recursive backtracking.
                # To optimize, we order the tree vertices by degree descending.
                # Let's create a list of vertices sorted by degree.
                # But we must respect the tree structure for backtracking.
                # So we will do a DFS, but when choosing the next vertex to place,
                # we pick the one with the highest degree among the "frontier" (neighbors of placed vertices).
                
                # State:
                # placed: set of tree vertices placed
                # mapping: dict tree_vertex -> host_vertex
                # used_hosts: set of host vertices used
                
                # We start with a random root? Or vertex 0?
                # Let's try all possible roots? No, too slow.
                # Let's pick the highest degree vertex in the tree as root.
                root = max(range(k), key=lambda v: len(adj[v]))
                
                # Try all host vertices for the root
                success = False
                for root_host in range(n):
                    # If we fail, try next root_host
                    # This is O(n * (search space))
                    
                    # Initialize
                    mapping = {root: root_host}
                    placed = {root}
                    used_hosts = {root_host}
                    
                    # DFS
                    def dfs(placed, mapping, used_hosts):
                        if len(placed) == k:
                            return True
                        
                        # Find frontier vertices (neighbors of placed, not placed)
                        frontier = []
                        for v in placed:
                            for u in adj[v]:
                                if u not in placed:
                                    frontier.append(u)
                        
                        if not frontier:
                            return False # Should not happen if tree is connected and k>1
                        
                        # Pick the frontier vertex with highest degree
                        # Tie-breaker: smallest index
                        next_v = max(frontier, key=lambda x: (len(adj[x]), -x))
                        
                        # Find candidate hosts for next_v
                        # Must be connected to parent_host via unused edge
                        # Parent is the one in placed that is adjacent to next_v
                        parent_v = None
                        for p in placed:
                            if next_v in adj[p]:
                                parent_v = p
                                break
                        parent_host = mapping[parent_v]
                        
                        candidates = []
                        for h in range(n):
                            if h in used_hosts:
                                continue
                            edge = (min(parent_host, h), max(parent_host, h))
                            if edge in used_edges:
                                continue
                            candidates.append(h)
                            
                        # Heuristic: Sort candidates by "potential"
                        # Potential: number of unused edges from h to other unplaced neighbors?
                        # This is expensive.
                        # Let's just shuffle or use fixed order.
                        # To allow backtracking to find a solution, fixed order is deterministic.
                        # If deterministic fails, we need randomization.
                        
                        # For now, try in fixed order.
                        for h in candidates:
                            # Check if this choice allows completion? (Forward checking)
                            # Simple check: Are there enough unused edges left?
                            # We don't track that easily.
                            
                            # Try it
                            mapping[next_v] = h
                            placed.add(next_v)
                            used_hosts.add(h)
                            
                            # Mark edges used? No, we only mark them when we finalize?
                            # No, we must mark them to avoid conflicts in deeper levels.
                            # But we are backtracking, so we need to undo.
                            # So we should add edges to a local set or check against global `used_edges`
                            # and a local `local_used_edges`.
                            
                            # Let's pass `local_used_edges` as part of state.
                            # This is getting complex for a pure Python recursive function.
                            
                            # Alternative:
                            # Since the graph is dense, conflicts are rare.
                            # We can just check against `used_edges` (global) and `mapping` (local).
                            # But we need to ensure edges between *placed* vertices are not in `used_edges`.
                            # They are not, because we checked when placing them.
                            # We also need to ensure edges between *new* vertex and *placed* vertices are not in `used_edges`.
                            # We only connect to parent.
                            # What about edges to other placed vertices?
                            # The tree has no cycles, so next_v is only connected to parent in the tree.
                            # So only one edge is added.
                            
                            # So, we just need to ensure (parent_host, h) is not in `used_edges`.
                            # We already did that in candidate generation.
                            
                            # So, we can recurse.
                            if dfs(placed, mapping, used_hosts):
                                return True
                            
                            # Backtrack
                            del mapping[next_v]
                            placed.remove(next_v)
                            used_hosts.remove(h)
                            
                        return False
                        
                    # We need to modify dfs to handle local edge usage properly.
                    # The `used_edges` in the closure is global.
                    # We cannot add to it during search because we need to backtrack.
                    # So we must pass a set of locally used edges.
                    
                    # Let's re-implement dfs with `local_edges` parameter.
                    
                    def dfs2(placed, mapping, used_hosts, local_edges):
                        if len(placed) == k:
                            return True
                        
                        frontier = []
                        for v in placed:
                            for u in adj[v]:
                                if u not in placed:
                                    frontier.append(u)
                        
                        if not frontier:
                            return False
                        
                        next_v = max(frontier, key=lambda x: (len(adj[x]), -x))
                        parent_v = next((p for p in placed if next_v in adj[p]), None)
                        if parent_v is None:
                            return False
                        parent_host = mapping[parent_v]
                        
                        candidates = []
                        for h in range(n):
                            if h in used_hosts:
                                continue
                            edge = (min(parent_host, h), max(parent_host, h))
                            if edge in used_edges or edge in local_edges:
                                continue
                            candidates.append(h)
                            
                        for h in candidates:
                            new_local_edges = local_edges | {(min(parent_host, h), max(parent_host, h))}
                            mapping[next_v] = h
                            placed.add(next_v)
                            used_hosts.add(h)
                            
                            if dfs2(placed, mapping, used_hosts, new_local_edges):
                                return True
                            
                            del mapping[next_v]
                            placed.remove(next_v)
                            used_hosts.remove(h)
                            
                        return False

                    if dfs2({root}, {root: root_host}, {root_host}, set()):
                        # Success
                        for tv, hv in mapping.items():
                            embeddings[k-1][tv] = hv
                        # Update global used_edges
                        for u, v in edges_k:
                            used_edges.add((min(embeddings[k-1][u], embeddings[k-1][v]), max(embeddings[k-1][u], embeddings[k-1][v])))
                        success = True
                        break
                if not success:
                    return None
            return embeddings

        # Try deterministic first
        sol = solve_deterministic()
        if sol is not None:
            # Verify
            # ...
            return sol
            
        # If deterministic fails, try randomized restarts
        # ...
        
        return None

    # The above code is a draft. I need to clean it up and ensure it's correct.
    # One issue: `embeddings[k-1][tv] = hv` assumes `embeddings[k-1]` is a list of length k.
    # I should initialize it as `[0]*k` or similar.
    
    # Let's refine the deterministic solver.
    
    def solve():
        used_edges = set()
        embeddings = [None] * 20
        
        # Sort trees by size descending
        tree_order = sorted(range(1, 21), key=lambda k: len(trees[k-1]), reverse=True)
        
        for k in tree_order:
            edges_k = trees[k-1]
            if not edges_k:
                embeddings[k-1] = [0]
                continue
            if k == 2:
                placed = False
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) not in used_edges:
                            embeddings[k-1] = [u, v]
                            used_edges.add((u, v))
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    return None
                continue
                
            adj = [[] for _ in range(k)]
            for u, v in edges_k:
                adj[u].append(v)
                adj[v].append(u)
                
            root = max(range(k), key=lambda v: len(adj[v]))
            
            found = False
            for root_host in range(n):
                # Try to embed with this root
                # Use DFS
                mapping = {root: root_host}
                placed = {root}
                used_hosts = {root_host}
                local_edges = set()
                
                def dfs(placed, mapping, used_hosts, local_edges):
                    if len(placed) == k:
                        return True
                    
                    frontier = []
                    for v in placed:
                        for u in adj[v]:
                            if u not in placed:
                                frontier.append(u)
                    
                    if not frontier:
                        return False
                    
                    # Heuristic: pick highest degree
                    next_v = max(frontier, key=lambda x: (len(adj[x]), -x))
                    parent_v = next((p for p in placed if next_v in adj[p]), None)
                    if parent_v is None:
                        return False
                    parent_host = mapping[parent_v]
                    
                    candidates = []
                    for h in range(n):
                        if h in used_hosts:
                            continue
                        edge = (min(parent_host, h), max(parent_host, h))
                        if edge in used_edges or edge in local_edges:
                            continue
                        candidates.append(h)
                        
                    # Shuffle candidates to break symmetry?
                    # For deterministic, no.
                    for h in candidates:
                        new_local_edges = local_edges | {(min(parent_host, h), max(parent_host, h))}
                        mapping[next_v] = h
                        placed.add(next_v)
                        used_hosts.add(h)
                        
                        if dfs(placed, mapping, used_hosts, new_local_edges):
                            return True
                        
                        del mapping[next_v]
                        placed.remove(next_v)
                        used_hosts.remove(h)
                        
                    return False

                if dfs({root}, {root: root_host}, {root_host}, set()):
                    # Success
                    emb = [0] * k
                    for tv, hv in mapping.items():
                        emb[tv] = hv
                    embeddings[k-1] = emb
                    for u, v in edges_k:
                        used_edges.add((min(emb[u], emb[v]), max(emb[u], emb[v])))
                    found = True
                    break
                    
            if not found:
                return None
                
        return embeddings

    sol = solve()
    if sol is not None:
        return sol
        
    # Fallback: Randomized
    # ...
    
    return [None] * 20

# The above code might be slow due to recursion overhead in Python.
# But for k=20, it should be fine.

# Let's add a timeout guard.
