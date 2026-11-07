"""Packing trees T1..T20 into K20 using backtracking search."""
import time
import sys
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
    deadline = time.time() + 140  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    
    # Sort trees by size descending for harder-first packing
    # T1 has 0 edges, T2 has 1 edge, ..., T20 has 19 edges
    # We'll embed larger trees first
    
    # Edge usage tracking: set of (min(u,v), max(u,v)) for used edges
    used_edges = set()
    
    # Result: list of 20 entries, entry k-1 is None or list of k ints
    result = [None] * 20
    
    # For each tree, we need to find an embedding (mapping from tree vertices 0..k-1 to host vertices 0..19)
    # such that all tree edges map to distinct unused K20 edges
    
    # Trees in order of embedding (largest first)
    tree_order = list(range(19, -1, -1))  # indices 19 down to 0, i.e., T20 first
    
    # For backtracking, we'll try to embed each tree in tree_order
    # But we need to return them in the original order (T1 at index 0, T20 at index 19)
    
    # Let's use a recursive backtracking approach
    # State: used_edges, assignments for trees already placed
    
    def try_embed(tree_idx, tree_edges, used_edges_set, current_assignments):
        """Try to embed the tree with given edges into unused host vertices."""
        k = len(tree_edges) + 1 if tree_edges else (1 if tree_idx == 0 else 2)
        # Actually, let's get k from the tree structure
        # tree_edges is a list of (u,v) pairs; vertices are 0..k-1
        
        if not tree_edges:
            # T1: single vertex, no edges. Any host vertex works.
            # But we need to pick one that doesn't conflict... actually no edges so any vertex
            # But we should pick a vertex. Let's just pick 0.
            # Actually for T1, any single vertex is fine since there are no edges to conflict.
            # But we need to return a list of 1 int.
            # Let's pick the first available... well, all are available for vertices since no edges.
            # Just return [0]
            return [0]
        
        # For trees with edges, we need to assign each tree vertex to a distinct host vertex
        # such that each tree edge (a,b) maps to (host[a], host[b]) which is not in used_edges_set
        
        # Get all vertices in the tree
        verts = set()
        for a, b in tree_edges:
            verts.add(a)
            verts.add(b)
        k = len(verts)
        verts = sorted(verts)
        
        # Build adjacency for the tree
        adj = {v: [] for v in verts}
        for a, b in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Try all permutations? That's too expensive for k=20.
        # Use backtracking: pick a root, assign it to a host vertex, then BFS/DFS assign neighbors
        
        # Pick root as vertex 0 of the tree (or any)
        root = verts[0]
        
        # Try each host vertex for the root
        for hv_root in range(20):
            assignment = {root: hv_root}
            # BFS/DFS to assign remaining vertices
            queue = [root]
            assigned = {root}
            success = True
            
            while queue and success:
                v = queue.pop(0)
                for w in adj[v]:
                    if w in assigned:
                        continue
                    # Try to assign w to a host vertex
                    found = False
                    for hv_w in range(20):
                        if hv_w in assignment.values():
                            continue
                        # Check if edge (assignment[v], hv_w) is available
                        e1 = min(assignment[v], hv_w)
                        e2 = max(assignment[v], hv_w)
                        if (e1, e2) in used_edges_set:
                            continue
                        # Also check that hv_w is not already used by another vertex
                        # (already checked above)
                        # Assign
                        assignment[w] = hv_w
                        assigned.add(w)
                        queue.append(w)
                        found = True
                        break
                    if not found:
                        # Need to backtrack - this simple approach doesn't backtrack well
                        # Let's use a proper recursive backtracking
                        success = False
                        break
            
            if success and len(assignment) == k:
                # Verify all edges
                valid = True
                new_edges = set()
                for a, b in tree_edges:
                    ha = assignment[a]
                    hb = assignment[b]
                    e1 = min(ha, hb)
                    e2 = max(ha, hb)
                    if (e1, e2) in used_edges_set:
                        valid = False
                        break
                    new_edges.add((e1, e2))
                if valid:
                    # Return the assignment as a list indexed by tree vertex
                    # result[i] = host vertex for tree vertex i
                    return [assignment[v] for v in range(k)]
        
        return None
    
    # The simple approach above doesn't handle backtracking well.
    # Let's use a more robust recursive backtracking with proper undo.
    
    def embed_tree(tree_edges, used_edges_set):
        """Find an embedding of the tree into K20 using unused edges."""
        if not tree_edges:
            # T1: return [0] (any single vertex)
            return [0]
        
        verts = set()
        for a, b in tree_edges:
            verts.add(a)
            verts.add(b)
        k = len(verts)
        
        if k == 2:
            # T2: two vertices, one edge
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges_set:
                        return [u, v]
            return None
        
        # Build adjacency
        adj = {v: [] for v in range(k)}
        for a, b in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Recursive backtracking
        assignment = [-1] * k  # assignment[i] = host vertex for tree vertex i, or -1 if unassigned
        
        def backtrack(tree_v):
            if tree_v == k:
                # All vertices assigned, verify
                for a, b in tree_edges:
                    ha = assignment[a]
                    hb = assignment[b]
                    e1 = min(ha, hb)
                    e2 = max(ha, hb)
                    if (e1, e2) in used_edges_set:
                        return False
                return True
            
            # Try each host vertex for tree_v
            for hv in range(20):
                if hv in assignment:
                    continue
                # Check edges to already-assigned neighbors
                valid = True
                for nb in adj[tree_v]:
                    if assignment[nb] != -1:
                        e1 = min(hv, assignment[nb])
                        e2 = max(hv, assignment[nb])
                        if (e1, e2) in used_edges_set:
                            valid = False
                            break
                if valid:
                    assignment[tree_v] = hv
                    if backtrack(tree_v + 1):
                        return True
                    assignment[tree_v] = -1
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Now do the packing: embed trees in order from largest to smallest
    # This greedy approach may fail, but let's try it first
    # If it fails, we could use more sophisticated search, but time is limited
    
    # Actually, let's just try embedding in order T20, T19, ..., T1
    # and see if we can pack them all
    
    for idx in tree_order:
        if time.time() > deadline:
            break
        tree_edges = trees[idx]
        emb = embed_tree(tree_edges, used_edges)
        if emb is None:
            # Failed to embed this tree
            # For now, leave it as None
            result[idx] = None
        else:
            result[idx] = emb
            # Update used edges
            for a, b in tree_edges:
                ha = emb[a]
                hb = emb[b]
                e1 = min(ha, hb)
                e2 = max(ha, hb)
                used_edges.add((e1, e2))
    
    # Verify the result
    # Check that all non-None entries have correct length and distinct ints
    for i in range(20):
        if result[i] is not None:
            if len(result[i]) != i + 1:
                result[i] = None
            elif len(set(result[i])) != i + 1:
                result[i] = None
            elif any(not isinstance(x, int) or x < 0 or x > 19 for x in result[i]):
                result[i] = None
    
    # Verify edge-disjointness
    used_edges_check = set()
    for i in range(20):
        if result[i] is None:
            continue
        tree_edges = trees[i]
        for a, b in tree_edges:
            ha = result[i][a]
            hb = result[i][b]
            e1 = min(ha, hb)
            e2 = max(ha, hb)
            if (e1, e2) in used_edges_check:
                # Conflict! This shouldn't happen with our embedding, but just in case
                result[i] = None
                break
            used_edges_check.add((e1, e2))
    
    # Count total edges
    total_edges = 0
    for i in range(20):
        if result[i] is not None:
            total_edges += len(trees[i])
    
    return result
