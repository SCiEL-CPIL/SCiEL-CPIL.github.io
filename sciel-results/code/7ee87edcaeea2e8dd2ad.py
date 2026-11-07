# EVOLVE-BLOCK-START
import random
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
    # Randomized Local Search for 3-regular bipartite graphs
    # Vertices 0-5 (Left), 6-11 (Right)
    # Represented as 6x6 matrix M where M[i][j] = 1 if i connected to 6+j
    
    best_val = float('inf')
    best_matrix = None
    
    # Initialize with a known 3-regular bipartite graph: Two K3,3
    # M = [[1,1,1,0,0,0],
    #      [1,1,1,0,0,0],
    #      [1,1,1,0,0,0],
    #      [0,0,0,1,1,1],
    #      [0,0,0,1,1,1],
    #      [0,0,0,1,1,1]]
    
    def random_3_reg_bipartite():
        # Generate a random 3-regular bipartite graph
        # Using random permutation composition
        # A 3-regular bipartite graph can be represented by 3 permutations p1, p2, p3
        # where edges are (i, 6+p1[i]), (i, 6+p2[i]), (i, 6+p3[i])
        # We need p1, p2, p3 to be permutations of 0..5
        # And for each i, p1[i], p2[i], p3[i] must be distinct.
        # This is equivalent to a Latin square of order 6 with symbols 0..5?
        # No, just 3 disjoint perfect matchings.
        
        # Simple method: Randomly shuffle columns for each of 3 layers?
        # Ensure no duplicate edges.
        
        # Let's use the configuration model or random restarts of 2-switches from a base.
        # Base:
        m = [
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [0,0,0,1,1,1],
            [0,0,0,1,1,1],
            [0,0,0,1,1,1]
        ]
        # Apply random 2-switches
        for _ in range(100):
            r1, r2 = random.sample(range(6), 2)
            c1, c2 = random.sample(range(6), 2)
            if m[r1][c1] == 1 and m[r2][c2] == 1 and m[r1][c2] == 0 and m[r2][c1] == 0:
                m[r1][c1] = 0; m[r2][c2] = 0
                m[r1][c2] = 1; m[r2][c1] = 1
            elif m[r1][c1] == 0 and m[r2][c2] == 0 and m[r1][c2] == 1 and m[r2][c1] == 1:
                m[r1][c1] = 1; m[r2][c2] = 1
                m[r1][c2] = 0; m[r2][c1] = 0
        return m

    def matrix_to_edges(m):
        edges = []
        for i in range(6):
            for j in range(6):
                if m[i][j]:
                    edges.append((i, 6+j))
        return edges

    def evaluate(m):
        edges = matrix_to_edges(m)
        adj = _build_adj(edges)
        return _count_cycle_lengths(adj)

    # Run multiple restarts
    for _ in range(50):
        m = random_3_reg_bipartite()
        val = evaluate(m)
        if val < best_val:
            best_val = val
            best_matrix = [row[:] for row in m]
        
        # Local search
        for _ in range(200):
            r1, r2 = random.sample(range(6), 2)
            c1, c2 = random.sample(range(6), 2)
            new_m = [row[:] for row in m]
            
            if new_m[r1][c1] == 1 and new_m[r2][c2] == 1 and new_m[r1][c2] == 0 and new_m[r2][c1] == 0:
                new_m[r1][c1] = 0; new_m[r2][c2] = 0
                new_m[r1][c2] = 1; new_m[r2][c1] = 1
                new_val = evaluate(new_m)
                if new_val <= val: # Accept if better or equal (to allow movement)
                    m = new_m
                    val = new_val
                    if val < best_val:
                        best_val = val
                        best_matrix = [row[:] for row in m]
            elif new_m[r1][c1] == 0 and new_m[r2][c2] == 0 and new_m[r1][c2] == 1 and new_m[r2][c1] == 1:
                new_m[r1][c1] = 1; new_m[r2][c2] = 1
                new_m[r1][c2] = 0; new_m[r2][c1] = 0
                new_val = evaluate(new_m)
                if new_val <= val:
                    m = new_m
                    val = new_val
                    if val < best_val:
                        best_val = val
                        best_matrix = [row[:] for row in m]
                        
    if best_matrix is None:
        # Fallback
        best_matrix = [
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [0,0,0,1,1,1],
            [0,0,0,1,1,1],
            [0,0,0,1,1,1]
        ]

    edges = matrix_to_edges(best_matrix)
    sorted_edges = sorted(tuple(sorted(e)) for e in edges)
    return [list(e) for e in sorted_edges]

# EVOLVE-BLOCK-END
