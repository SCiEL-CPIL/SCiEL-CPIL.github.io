# EVOLVE-BLOCK-START
import itertools

def _get_cycle_lengths(adj):
    n = len(adj)
    lengths = set()
    for start in range(n):
        stack = [(start, [start], frozenset([start]))]
        while stack:
            node, path, visited = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 3:
                    lengths.add(len(path))
                elif nb > start and nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))
    return lengths

def _build_adj(edges):
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _count_cycle_lengths(adj):
    lengths = _get_cycle_lengths(adj)
    return len(lengths) if lengths else float('inf')

def construct():
    # Systematic search for 3-regular bipartite graphs on 6+6 vertices
    # Left: 0-5, Right: 6-11
    # Symmetry breaking: L[0] connects to R[0], R[1], R[2] (indices 6,7,8)
    # We represent right neighbors as indices 0..5
    
    best_edges = None
    best_count = float('inf')
    
    # Pre-compute all 3-subsets of {0..5}
    subsets = list(itertools.combinations(range(6), 3))
    
    # Backtracking state:
    # left_neighbors: list of 3-subsets for L[0]..L[5]
    # right_degree: count of how many left nodes connect to each right node (0..5)
    
    def backtrack(idx, left_neighbors, right_degree):
        nonlocal best_edges, best_count
        
        if idx == 6:
            # Check if all right degrees are 3
            if all(d == 3 for d in right_degree):
                # Build edges
                edges = []
                for i, neigh in enumerate(left_neighbors):
                    for r in neigh:
                        edges.append((i, 6 + r))
                
                # Check validity (should be valid by construction)
                # Check triangle-free (bipartite is always triangle-free)
                # Count cycle lengths
                adj = _build_adj(edges)
                count = _count_cycle_lengths(adj)
                
                if count < best_count:
                    best_count = count
                    best_edges = edges
            return
        
        # Symmetry breaking for idx == 0
        if idx == 0:
            # Fix L[0] to connect to {0, 1, 2}
            # Note: This assumes WLOG that we can permute right nodes.
            # Since we are looking for the minimum count, and all permutations
            # of right nodes yield isomorphic graphs with same cycle lengths,
            # this is a valid reduction.
            candidates = [(0, 1, 2)]
        else:
            candidates = subsets
            
        for neigh in candidates:
            # Check if adding these neighbors exceeds degree 3 for any right node
            valid = True
            new_right_degree = right_degree[:]
            for r in neigh:
                if new_right_degree[r] >= 3:
                    valid = False
                    break
                new_right_degree[r] += 1
            
            if valid:
                backtrack(idx + 1, left_neighbors + [neigh], new_right_degree)
                
        # Pruning: if current partial assignment already forces a high count?
        # Hard to prune cycle count easily. But search space is small.

    backtrack(0, [], [0]*6)
    
    if best_edges is None:
        # Fallback: K3,3 + K3,3
        edges = []
        for i in range(6):
            for j in range(6):
                if i < 3 and j < 3:
                    edges.append((i, 6+j))
                elif i >= 3 and j >= 3:
                    edges.append((i, 6+j))
        # Wait, K3,3 is complete bipartite between 3 and 3.
        # My loop above is not K3,3.
        # K3,3: L{0,1,2} connected to R{0,1,2}. L{3,4,5} connected to R{3,4,5}.
        # This is two disjoint K3,3.
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append((i, 6+j))
        for i in range(3, 6):
            for j in range(3, 6):
                edges.append((i, 6+j))
        best_edges = edges
        
    sorted_edges = sorted(tuple(sorted(e)) for e in best_edges)
    return [list(e) for e in sorted_edges]

# EVOLVE-BLOCK-END
