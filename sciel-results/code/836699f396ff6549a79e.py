"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit, keep margin
    
    trees = instance_trees(20, 611)
    n = 20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # We'll embed trees in order of decreasing size (most constrained first)
    # Tree k has k vertices and k-1 edges (for k>=2), T1 has 0 edges
    # Order: T20, T19, ..., T2, T1
    
    # For each tree, we need to find a mapping of its k vertices to n host vertices
    # such that no edge is used twice
    
    # Use backtracking with constraint propagation
    
    # Precompute tree structures
    # tree[i] is list of (u,v) edges for tree i (0-indexed, tree 0 = T1)
    
    def get_tree_adj(tree_edges, k):
        """Build adjacency list for a tree with k vertices."""
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    # Order of embedding: largest trees first
    # Tree index i (0-based) corresponds to T_{i+1} with i+1 vertices
    # T1: 1 vertex, 0 edges; T2: 2 vertices, 1 edge; ...; T20: 20 vertices, 19 edges
    embed_order = list(range(19, -1, -1))  # T20, T19, ..., T1
    
    # We'll try to embed trees in this order
    # For each tree, try all possible injective mappings of its vertices to host vertices
    # This is expensive, so use backtracking within each tree embedding
    
    used_edges = set()
    result = [None] * 20  # result[i] will be the embedding of T_{i+1} or None
    
    def try_embed_tree(tree_idx, used_edges_set, host_used_vertices=None):
        """Try to embed tree tree_idx (0-indexed) into K20.
        Returns a list of host vertices for the tree's vertices, or None if impossible.
        tree_idx 0 = T1 (1 vertex), tree_idx 1 = T2 (2 vertices), etc.
        """
        k = tree_idx + 1  # number of vertices in this tree
        tree_edges = trees[tree_idx]
        
        if k == 1:
            # T1: just pick any free vertex
            for v in range(n):
                if host_used_vertices is None or v not in host_used_vertices:
                    return [v]
            return None
        
        if k == 2:
            # T2: pick any edge not used
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges_set:
                        return [u, v]
            return None
        
        # For larger trees, use backtracking
        # Build adjacency
        adj = get_tree_adj(tree_edges, k)
        
        # Find a root (vertex 0 for simplicity, or a leaf)
        # Use DFS-style backtracking to assign host vertices to tree vertices
        
        # Order tree vertices for assignment: use a leaf-to-root order
        # Find leaves
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            return None
        
        # Build an ordering: start from a leaf, do BFS/DFS
        # Actually, let's use a simple approach: assign vertices in order 0..k-1
        # but use backtracking with constraint checking
        
        # Better: use a recursive backtracking that assigns host vertices to tree vertices
        # in an order that maximizes constraint propagation
        
        # Let's use a simple recursive approach
        assignment = [None] * k  # assignment[tree_v] = host_v
        host_to_tree = {}  # host_v -> tree_v (for quick lookup)
        
        def backtrack(tree_v):
            if tree_v == k:
                return True
            
            # Try all possible host vertices for tree_v
            for host_v in range(n):
                if host_v in host_to_tree:
                    continue
                
                # Check if all edges from tree_v to already-assigned neighbors are valid
                valid = True
                for nb in adj[tree_v]:
                    if assignment[nb] is not None:
                        edge = (min(host_v, assignment[nb]), max(host_v, assignment[nb]))
                        if edge in used_edges_set:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Assign
                assignment[tree_v] = host_v
                host_to_tree[host_v] = tree_v
                
                if backtrack(tree_v + 1):
                    return True
                
                # Unassign
                assignment[tree_v] = None
                del host_to_tree[host_v]
            
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Greedy approach: embed trees in order, using the first valid embedding found
    # This may not be optimal, but let's try
    
    host_used = set()  # vertices used by any embedded tree (but vertices can be shared!)
    # Wait, vertices CAN be shared between trees! Only edges must be disjoint.
    # So host_used_vertices is not a constraint.
    
    for tree_idx in embed_order:
        if time.time() > deadline - 5:
            break
        
        k = tree_idx + 1
        if k == 1:
            result[tree_idx] = [0]  # T1: any single vertex
            continue
        
        embedding = try_embed_tree(tree_idx, used_edges)
        if embedding is not None:
            result[tree_idx] = embedding
            # Mark edges as used
            tree_edges = trees[tree_idx]
            for u, v in tree_edges:
                edge = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
                used_edges.add(edge)
        else:
            result[tree_idx] = None
    
    # Count total embedded edges
    total_edges = 0
    for i in range(20):
        if result[i] is not None:
            k = i + 1
            if k >= 2:
                total_edges += k - 1
    
    return result

# EVOLVE-BLOCK-END
