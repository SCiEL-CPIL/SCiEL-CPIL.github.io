import itertools
import numpy as np

def construct():
    n = 16
    vertices = list(range(n))
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
            
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
        
    def has_q3_in_subset(subset, adj):
        s = list(subset)
        for A in itertools.combinations(s, 4):
            B = tuple(x for x in s if x not in A)
            # Count edges between A and B
            # M.sum() is the number of edges
            # We can count without building full matrix if we want, 
            # but building 4x4 is cheap.
            M = np.zeros((4, 4), dtype=int)
            count = 0
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    if adj[a, b]:
                        M[i, j] = 1
                        count += 1
                        
            # Optimization: If count > 12, no Q3 possible in this partition
            if count > 12:
                continue
                
            # If count <= 12, check if complement has perfect matching
            C = 1 - M
            
            # Simple check for perfect matching in 4x4
            # We can hardcode the permutations?
            # There are 24 permutations.
            # Check if any permutation p satisfies C[i][p[i]] == 1 for all i
            found_match = False
            for p in itertools.permutations(range(4)):
                if all(C[i][p[i]] == 1 for i in range(4)):
                    found_match = True
                    # Reconstruct one edge to remove
                    # The matching edges are (A[i], B[p[i]]) which are in C (missing in M)
                    # The Q3 edges are the other 12.
                    # Pick (A[0], B[1]) if p[0] != 1, else (A[0], B[2])...
                    for i in range(4):
                        for j in range(4):
                            if j != p[i]:
                                return True, (A[i], B[j])
                    break
            if found_match:
                # The loop above returns if found
                pass
        return False, None

    while True:
        found = False
        for subset in itertools.combinations(vertices, 8):
            exists, edge_to_remove = has_q3_in_subset(subset, adj)
            if exists:
                u, v = edge_to_remove
                # Remove edge
                if (u, v) in edges:
                    edges.remove((u, v))
                else:
                    edges.remove((v, u))
                adj[u, v] = False
                adj[v, u] = False
                found = True
                break
        if not found:
            break
            
    return [list(e) for e in edges]
