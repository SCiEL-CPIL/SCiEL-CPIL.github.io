"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
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
    deadline = time.time() + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    # trees[k-1] is tree T_k with k vertices and k-1 edges
    
    # We need to embed each tree T_k into K20 (vertices 0..19)
    # Entry k-1 is None or a list of k distinct int host vertices in 0..19
    # Position i is the image of vertex i of tree T_k
    # Embedded trees must be edge-disjoint in K20
    
    # Total edges in K20 = 190
    # Total edges in all trees = sum(k-1 for k=1..20) = 190
    # So we need a perfect packing
    
    # Process largest first for better constraint propagation
    # Order: T20, T19, T18, ..., T1
    order = list(range(19, -1, -1))  # indices 19,18,...,0 corresponding to T20,T19,...,T1
    
    # Track used edges in K20
    used_edges = set()
    embeddings = [None] * 20  # embeddings[k-1] = list of 20 host vertices or None
    
    # For backtracking, we need to try all possible embeddings
    # An embedding of tree T_k is a bijection from {0..k-1} to k distinct vertices in 0..19
    # Such that for each edge (u,v) in T_k, the edge (host[u], host[v]) is not in used_edges
    
    def try_embed(tree_idx, depth, assignment, used_edges_local):
        """
        Try to embed tree at tree_idx.
        tree_idx is index into trees list (0-based, so tree_idx=19 is T20)
        depth: current vertex being assigned (0..k-1)
        assignment: list mapping tree vertex -> host vertex (partially filled)
        used_edges_local: set of edges used in K20 so far
        """
        if time.time() > deadline:
            return None
        
        k = tree_idx + 1  # tree T_k has k vertices
        if depth == k:
            return assignment[:]
        
        # Get available host vertices (not yet used in this embedding)
        used_hosts = set(assignment[:depth])
        available = [v for v in range(20) if v not in used_hosts]
        
        # Try each available host vertex
        for host in available:
            # Check if assigning tree vertex 'depth' to 'host' creates any conflict
            valid = True
            new_edges = []
            for i in range(depth):
                if assignment[i] is not None:
                    # Check if (i, depth) is an edge in the tree
                    # We need to check tree edges
                    pass
            
            # Better approach: check all edges from 'depth' to already-assigned vertices
            tree_edges = trees[tree_idx]
            for (u, v) in tree_edges:
                if u == depth:
                    other = v
                elif v == depth:
                    other = u
                else:
                    continue
                if other < depth and assignment[other] is not None:
                    e = frozenset((host, assignment[other]))
                    if e in used_edges_local:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Also check edges from 'depth' to unassigned vertices - can't check yet
            # But we need to make sure we don't create duplicate edges within this tree
            # Actually, the tree is a tree, so no cycles, but we need to ensure
            # the mapping is injective (already ensured by used_hosts)
            
            assignment[depth] = host
            result = try_embed(tree_idx, depth + 1, assignment, used_edges_local)
            if result is not None:
                return result
            assignment[depth] = None
        
        return None
    
    # Simpler approach: for each tree, try all permutations of k vertices from 20
    # But that's too slow for large k. Use backtracking with pruning.
    
    # Let me use a different strategy: process trees in order, and for each tree
    # try to find an embedding that doesn't conflict with already-used edges.
    
    # For efficiency, precompute the tree edges as adjacency
    tree_adj = []
    for t in trees:
        adj = [[] for _ in range(len(t) + 1)]
        for (u, v) in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    def embed_tree(tree_idx, used_edges_set):
        """Find an embedding of tree T_{tree_idx+1} into K20 avoiding used_edges_set."""
        if time.time() > deadline:
            return None
        
        k = tree_idx + 1
        if k == 1:
            # Single vertex, any host works
            return [0]
        
        # Backtracking: assign tree vertices 0..k-1 to host vertices
        # Order: assign vertex 0 first, then 1, etc.
        # But better: assign in an order that maximizes constraint propagation
        
        assignment = [None] * k
        assignment[0] = 0  # Fix vertex 0 to host 0 (WLOG by symmetry of K20)
        
        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == k:
                # Verify all edges are valid
                for (u, v) in trees[tree_idx]:
                    e = frozenset((assignment[u], assignment[v]))
                    if e in used_edges_set:
                        return False
                return True
            
            # Get neighbors of 'depth' that are already assigned
            assigned_neighbors = []
            for (u, v) in trees[tree_idx]:
                if u == depth:
                    if v < depth and assignment[v] is not None:
                        assigned_neighbors.append(assignment[v])
                elif v == depth:
                    if u < depth and assignment[u] is not None:
                        assigned_neighbors.append(assignment[u])
            
            used_hosts = set(h for h in assignment if h is not None)
            
            # Try each available host
            for host in range(20):
                if host in used_hosts:
                    continue
                
                # Check if edges to already-assigned neighbors are free
                valid = True
                for nh in assigned_neighbors:
                    e = frozenset((host, nh))
                    if e in used_edges_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                assignment[depth] = host
                if backtrack(depth + 1):
                    return True
                assignment[depth] = None
            
            return False
        
        if backtrack(1):
            return assignment[:]
        return None
    
    # Process trees largest first
    # But we need to be careful: the embedding of a tree depends on which edges are used
    # Let's try a greedy approach first, then backtracking if needed
    
    # Actually, let me just do full backtracking over all trees
    # This could be expensive, so let's use the deadline guard
    
    best_result = [None] * 20
    
    # Try to embed trees in order T20, T19, ..., T1
    current_used = set()
    current_embeddings = [None] * 20
    
    for tree_idx in order:
        k = tree_idx + 1
        emb = embed_tree(tree_idx, current_used)
        if emb is None:
            break
        current_embeddings[tree_idx] = emb
        # Add edges to used set
        for (u, v) in trees[tree_idx]:
            e = frozenset((emb[u], emb[v]))
            current_used.add(e)
    
    # Count how many trees we embedded
    count = sum(1 for x in current_embeddings if x is not None)
    
    # If we didn't embed all, try a more sophisticated backtracking
    if count < 20:
        # Full backtracking over all trees
        def full_backtrack(order_idx, used_edges, embeddings):
            if time.time() > deadline:
                return False
            
            if order_idx == len(order):
                return True
            
            tree_idx = order[order_idx]
            k = tree_idx + 1
            
            emb = embed_tree(tree_idx, used_edges)
            if emb is None:
                return False
            
            embeddings[tree_idx] = emb
            new_edges = []
            for (u, v) in trees[tree_idx]:
                e = frozenset((emb[u], emb[v]))
                used_edges.add(e)
                new_edges.append(e)
            
            if full_backtrack(order_idx + 1, used_edges, embeddings):
                return True
            
            # Undo
            for e in new_edges:
                used_edges.discard(e)
            embeddings[tree_idx] = None
            return False
        
        emb2 = [None] * 20
        if full_backtrack(0, set(), emb2):
            current_embeddings = emb2
    
    return current_embeddings
# EVOLVE-BLOCK-END
