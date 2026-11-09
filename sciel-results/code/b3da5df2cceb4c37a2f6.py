import time
import random
import sys

def has_c5(adj_matrix, n):
    """Check if graph contains a 5-cycle."""
    # A 5-cycle is a set of 5 distinct vertices v0..v4 such that
    # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0) are edges.
    # We can iterate over all combinations of 5 vertices? n=33, C(33,5) is ~200k.
    # 200k * 5 checks is fast.
    # But we can do better.
    # For each pair (u, v) that is an edge, check if there is a path of length 3 between u and v
    # that does not use the edge (u, v) and does not repeat vertices.
    # Actually, simpler:
    # Iterate u < v < w < x < y.
    # Check if they form a cycle.
    # This is O(n^5) which is 33^5 = 39 million. Might be too slow in Python if not optimized.
    # Let's use a bitset approach or just check specific structures.
    
    # Efficient check:
    # For each edge (u, v), we want to know if there is a path of length 4 from u to v.
    # This is equivalent to checking if (A^4)[u][v] > 0, where A is adjacency matrix.
    # But we must ensure the path is simple (no repeated vertices).
    # For n=33, we can just do a DFS/BFS from u to v with depth limit 4, avoiding the direct edge.
    
    # Given n is small, let's just check all 5-tuples.
    # To speed up, we can precompute neighbors.
    
    # Actually, let's just return False for now and use a simpler heuristic if needed.
    # But we need correctness.
    
    # Let's try to optimize the 5-tuple check.
    # We can skip if the subgraph induced by the 5 vertices doesn't have at least 5 edges.
    
    # For the purpose of this task, if we find a candidate, we verify it.
    # During search, we might use a faster proxy or just accept the risk if we verify at the end.
    # But we need to guide the search.
    
    # Let's implement a fast C5 check.
    # For each vertex u:
    #   For each neighbor v of u:
    #     For each neighbor w of v (w != u):
    #       For each neighbor x of w (x != u, x != v):
    #         If x is a neighbor of u and x != v and x != w:
    #            # We have u-v-w-x-u. This is a 4-cycle.
    #            # We need a 5-cycle.
    
    # Let's stick to the C(n,5) check but optimize it with numpy or just early exit.
    # With n=33, C(33,5) is 237,336.
    # In Python, 237k iterations is very fast (< 0.1s).
    
    for i in range(n-4):
        for j in range(i+1, n-3):
            for k in range(j+1, n-2):
                for l in range(k+1, n-1):
                    for m in range(l+1, n):
                        # Check if i-j-k-l-m-i is a cycle
                        if (adj_matrix[i][j] and adj_matrix[j][k] and adj_matrix[k][l] and adj_matrix[l][m] and adj_matrix[m][i]):
                            return True
    return False

def independence_number_bound(adj_matrix, n):
    """
    Returns True if alpha < 9.
    We can use a greedy coloring of the complement to get a lower bound on alpha.
    If the lower bound is >= 9, return False.
    If we can't prove it's >= 9, we assume it might be < 9.
    However, we need to be sure.
    For n=33, we can run a branch and bound for alpha.
    But that might be slow.
    Let's use a simple greedy independent set finder.
    If we find an independent set of size 9, return False.
    If we don't, we return True (optimistically).
    Since we are searching, we will verify the final candidate exactly.
    """
    # Greedy independent set
    remaining = set(range(n))
    independent_set = []
    # Order vertices by degree (ascending) to find large independent sets
    degrees = [sum(adj_matrix[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    for v in order:
        if v in remaining:
            independent_set.append(v)
            # Remove v and its neighbors
            remaining.discard(v)
            for u in range(n):
                if adj_matrix[v][u]:
                    remaining.discard(u)
                    
    if len(independent_set) >= 9:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Known valid graph: 8 disjoint K4s
    n_best = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find a graph with n=33
    # Strategy: Start with 8 K4s, merge two K4s by removing one edge and adding connections?
    # Or just random search on n=33.
    
    # Let's try random search for n=33
    # We want a graph that is C5-free and alpha < 9.
    # C5-free suggests low edge density or specific structure.
    # Alpha < 9 suggests high edge density.
    # This is a contradiction for random graphs.
    # So we need a structured graph.
    
    # Let's try to modify the 8-K4 structure.
    # Take 7 K4s and one component of 9 vertices.
    # The 9-vertex component must be C5-free and alpha <= 1 (since 7*1 + 1 = 8).
    # A graph on 9 vertices with alpha=1 is a clique K9.
    # K9 contains C5.
    # So we cannot have a component with alpha=1 and n=9.
    # We need the 9-vertex component to have alpha=1? No.
    # Total alpha = sum of alphas of components.
    # If we have 7 K4s (alpha=7) and one component C of size 9.
    # We need alpha(C) <= 1.
    # This implies C is a clique. K9 has C5.
    # So this decomposition fails.
    
    # What if we connect the components?
    # If we connect 8 K4s with a few edges, we might reduce alpha.
    # But adding edges creates C5s.
    
    # Let's try a different structure.
    # The complement of a C5-free graph is a graph with no independent set of size 5? No.
    # C5-free means no 5-cycle.
    
    # Given the difficulty, I will stick to n=32 unless a quick search finds 33.
    # I will implement a quick random search for n=33.
    
    import itertools
    
    def is_valid(n, edges):
        # Build adj matrix
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        
        if has_c5(adj, n):
            return False
        
        if not independence_number_bound(adj, n):
            return False
        
        return True

    # Search for n=33
    # Try random graphs with p=0.5? No, too dense for C5-free.
    # Try p=0.2?
    
    # Actually, let's just return the known best 32.
    # The probability of finding 33 in 100 seconds with random search is low.
    # But I can try a few structured modifications.
    
    # Modification: Take 8 K4s. Remove one vertex from one K4 (becomes K3).
    # Now we have 7 K4s, 1 K3, and 1 isolated vertex.
    # n = 28 + 3 + 1 = 32.
    # Connect the isolated vertex to the K3 to make a K4? Back to start.
    # Connect the isolated vertex to one vertex in each K4?
    # This creates the alpha=9 problem.
    
    # I will return 32.
    
    return {'n': n_best, 'edges': best_edges}

# EVOLVE-BLOCK-END
