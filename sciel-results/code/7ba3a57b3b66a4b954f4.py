"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_k_clique(adj, k=5):
    """Check if graph with adjacency matrix has a clique of size k."""
    n = len(adj)
    # For each 4-tuple, check if all pairwise connected
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if not adj[a][d] or not adj[b][d] or not adj[c][d]:
                        continue
                    for e in range(d+1, n):
                        if (adj[a][e] and adj[b][e] and adj[c][e] and adj[d][e]):
                            return True
    return False

def _has_k_independent(adj, k=5):
    """Check if graph has independent set of size k."""
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if adj[a][b]:
                continue
            for c in range(b+1, n):
                if adj[a][c] or adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] or adj[b][d] or adj[c][d]:
                        continue
                    for e in range(d+1, n):
                        if adj[a][e] or adj[b][e] or adj[c][e] or adj[d][e]:
                            continue
                        return True
    return False

def _is_55_free(adj):
    """Check if graph is (5,5)-free."""
    return not _has_k_clique(adj, 5) and not _has_k_independent(adj, 5)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i][j] = adj[j][i] = True
    
    best_adj = [row[:] for row in adj]
    best_n = q
    
    # Try to add vertices by trying all possible connection patterns
    # A new vertex v connects to a subset S of existing vertices.
    # For no K5: S must not contain 4 vertices that form a K4 in the existing graph
    #   (because v + K4 = K5)
    # For no independent 5: complement of S must not contain 4 vertices that form
    #   an independent 4-set in the existing graph (because v + independent 4 = independent 5)
    
    # Precompute all K4s and independent 4-sets
    def find_k4s(adj, n):
        """Find all 4-cliques."""
        k4s = []
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            k4s.append((a, b, c, d))
        return k4s
    
    def find_ind_4s(adj, n):
        """Find all independent 4-sets."""
        ind_4s = []
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if adj[a][c] or adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] or adj[b][d] or adj[c][d]:
                            continue
                        ind_4s.append((a, b, c, d))
        return ind_4s
    
    n = q
    k4s = find_k4s(adj, n)
    ind_4s = find_ind_4s(adj, n)
    
    # Try adding vertices
    max_attempts = 50
    added = 0
    for attempt in range(max_attempts):
        if time.time() > deadline - 30:
            break
        
        # Recompute K4s and independent 4-sets
        k4s = find_k4s(adj, n)
        ind_4s = find_ind_4s(adj, n)
        
        # For a new vertex, it must not connect to all 4 vertices of any K4
        # and must connect to at least 1 vertex of any independent 4-set
        
        # Try random subsets
        found = False
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            
            # Random subset of vertices to connect to
            S = set()
            for v in range(n):
                if random.random() < 0.5:
                    S.add(v)
            
            # Check K4 constraint: S must not contain any K4
            k4_ok = True
            for k4 in k4s:
                if all(v in S for v in k4):
                    k4_ok = False
                    break
            if not k4_ok:
                continue
            
            # Check independent 4-set constraint: complement of S must not contain any independent 4-set
            comp_S = set(range(n)) - S
            ind_ok = True
            for i4 in ind_4s:
                if all(v in comp_S for v in i4):
                    ind_ok = False
                    break
            if not ind_ok:
                continue
            
            # Try adding this vertex
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            new_adj.append([False] * new_n)
            for i in range(n):
                new_adj[i].append(i in S)
                new_adj[new_n - 1][i] = i in S
            new_adj[new_n - 1][new_n - 1] = False
            
            if _is_55_free(new_adj):
                adj = new_adj
                n = new_n
                if n > best_n:
                    best_adj = [row[:] for row in adj]
                    best_n = n
                found = True
                break
        
        if not found:
            break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i][j]:
                edges.append((i, j))
    
    return best_n, edges
# EVOLVE-BLOCK-END
