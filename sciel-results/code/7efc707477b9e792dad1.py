import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """
    Check if graph is NOT 3-colorable.
    Returns True if not 3-colorable (i.e., chromatic number >= 4).
    Uses bitmask backtracking with ordering by degree.
    """
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending to prune faster
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Map vertex to position in order
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
        
    # Precompute earlier neighbors mask for each vertex in the order
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        nm = neigh[v]
        while nm:
            lsb = nm & (-nm)
            u = lsb.bit_length() - 1
            if pos_of[u] < idx:
                mask |= (1 << pos_of[u])
            nm ^= lsb
        earlier_neigh[v] = mask
        
    # color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        em = earlier_neigh[v]
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        
        available = 7 & ~forbidden
        if available == 0:
            return False
            
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)

def _count_edges(adj, n):
    return sum(sum(row) for row in adj) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """
    Create adjacency matrix for Grötzsch graph on vertices 0-10.
    Vertices 11-15 are initially isolated.
    """
    adj = [[0]*n for _ in range(n)]
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _construct_blowup_groetzsch(n=16):
    """
    Construct a graph by blowing up an edge in the Grötzsch graph.
    Replaces edge (0,1) with a bipartite graph U x V.
    Uses vertices 0..6 for U and V, and 7..15 for the rest of the Grötzsch graph.
    """
    adj = [[0]*n for _ in range(n)]
    
    # Mapping of old Grötzsch vertices to new indices
    # Old 0, 1 are replaced by U, V.
    # Old 2->7, 3->8, 4->9, 5->10, 6->11, 7->12, 8->13, 9->14, 10->15.
    # U = {0, 1, 2}
    # V = {3, 4, 5, 6}
    
    U = [0, 1, 2]
    V = [3, 4, 5, 6]
    
    # Edges within U x V
    for u in U:
        for v in V:
            adj[u][v] = 1
            adj[v][u] = 1
            
    # Neighbors of old 0 (excluding 1) in Grötzsch: 4, 6, 9
    # Mapped: 4->9, 6->11, 9->14
    N0 = [9, 11, 14]
    
    # Neighbors of old 1 (excluding 0) in Grötzsch: 2, 5, 7
    # Mapped: 2->7, 5->10, 7->12
    N1 = [7, 10, 12]
    
    # Connect N0 to V
    for u in N0:
        for v in V:
            adj[u][v] = 1
            adj[v][u] = 1
            
    # Connect N1 to U
    for u in U:
        for v in N1:
            adj[u][v] = 1
            adj[v][u] = 1
            
    # Edges among the rest of the Grötzsch graph (mapped)
    # Old edges:
    # Cycle: (0,1) removed. (1,2), (2,3), (3,4), (4,0).
    # (1,2) -> N1[0] (7) connected to ... wait.
    # Let's just re-add the edges of the subgraph induced by {2,3,4,5,6,7,8,9,10} in Grötzsch.
    
    # Old vertices: 2,3,4,5,6,7,8,9,10
    # New vertices: 7,8,9,10,11,12,13,14,15
    
    # Edges in Grötzsch among these:
    # (2,3): 7-8
    # (3,4): 8-9
    # (2,6): 7-11
    # (3,7): 8-12
    # (4,8): 9-13
    # (5,10): 10-15
    # (6,10): 11-15
    # (7,10): 12-15
    # (8,10): 13-15
    # (9,10): 14-15
    # (4,5): 9-10
    # (1,5) removed (1 is old 1).
    # (0,9) removed (0 is old 0).
    # (0,6) removed.
    # (1,7) removed.
    
    # Let's list all edges in Grötzsch and map them.
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # (1,2) -> (old 1, old 2). old 1 is removed.
    # (2,3) -> (7, 8)
    # (3,4) -> (8, 9)
    # (4,0) -> (9, old 0). old 0 is removed.
    
    # Spokes:
    # (0,6) -> removed.
    # (1,7) -> removed.
    # (2,6) -> (7, 11)
    # (3,7) -> (8, 12)
    # (4,8) -> (9, 13)
    
    # Cross:
    # (1,5) -> removed.
    # (0,9) -> removed.
    # (2,6) already listed.
    # (3,7) already listed.
    # (4,8) already listed.
    # (4,5) -> (9, 10)
    # (5,4) same.
    
    # Wait, my Grötzsch construction:
    # i=0: 0-6, 1-5
    # i=1: 1-7, 2-6
    # i=2: 2-8, 3-7
    # i=3: 3-9, 4-8
    # i=4: 4-5, 0-9
    
    # Edges to keep (both endpoints in {2..10}):
    # (2,6) -> 7-11
    # (3,7) -> 8-12
    # (4,8) -> 9-13
    # (4,5) -> 9-10
    # (2,3) -> 7-8
    # (3,4) -> 8-9
    # (5,10) -> 10-15
    # (6,10) -> 11-15
    # (7,10) -> 12-15
    # (8,10) -> 13-15
    # (9,10) -> 14-15
    
    edges_to_add = [
        (7, 11), (8, 12), (9, 13), (9, 10),
        (7, 8), (8, 9),
        (10, 15), (11, 15), (12, 15), (13, 15), (14, 15)
    ]
    
    for u, v in edges_to_add:
        adj[u][v] = 1
        adj[v][u] = 1
        
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    adj = _construct_blowup_groetzsch(n)
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    def compute_cn(adj, n):
        cn = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        mask |= (1 << k)
                cn[i][j] = mask
                cn[j][i] = mask
        return cn
        
    cn = compute_cn(adj, n)
    
    def try_add_edge(i, j):
        if adj[i][j]:
            return False
        if cn[i][j] != 0:
            return False
        
        adj[i][j] = 1
        adj[j][i] = 1
        
        old_cn_i = [cn[i][k] for k in range(n)]
        old_cn_j = [cn[j][k] for k in range(n)]
        for k in range(n):
            if k != i and k != j:
                if adj[i][k]:
                    cn[i][k] |= (1 << j)
                    cn[k][i] |= (1 << j)
                if adj[j][k]:
                    cn[j][k] |= (1 << i)
                    cn[k][j] |= (1 << i)
        cn[i][j] = 0
        cn[j][i] = 0
        
        if _is_not_3_colorable(adj, n):
            return True
        else:
            adj[i][j] = 0
            adj[j][i] = 0
            for k in range(n):
                if k != i and k != j:
                    if adj[i][k]:
                        cn[i][k] = old_cn_i[k]
                        cn[k][i] = old_cn_i[k]
                    if adj[j][k]:
                        cn[j][k] = old_cn_j[k]
                        cn[k][j] = old_cn_j[k]
            return False
            
    for iteration in range(10000):
        if time.time() > deadline:
            break
            
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        improved = False
        
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            if try_add_edge(i, j):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True

    return _get_edges(best_adj, n)
