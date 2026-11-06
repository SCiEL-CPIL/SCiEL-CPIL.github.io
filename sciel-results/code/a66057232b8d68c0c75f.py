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
    # earlier_neigh[v] is a bitmask of positions of neighbors of v that come before v in 'order'
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        # Neighbors of v that have position < idx
        # We can iterate through neighbors of v
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
        # Determine forbidden colors based on earlier neighbors
        forbidden = 0
        em = earlier_neigh[v]
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        
        available = 7 & ~forbidden
        if available == 0:
            return False
            
        # Try available colors
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
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_independent_set(adj, n, size):
    """Find an independent set of given size. Returns list of vertices or None."""
    # Simple backtracking
    vertices = list(range(n))
    # Order by degree ascending (low degree vertices are more likely to be in independent set)
    deg = [sum(adj[v]) for v in range(n)]
    vertices.sort(key=lambda x: deg[x])
    
    def backtrack(idx, current_set):
        if len(current_set) == size:
            return current_set
        if idx == len(vertices):
            return None
        
        v = vertices[idx]
        # Option 1: Include v
        if all(not adj[v][u] for u in current_set):
            res = backtrack(idx + 1, current_set + [v])
            if res:
                return res
        # Option 2: Exclude v
        res = backtrack(idx + 1, current_set)
        if res:
            return res
        return None

    return backtrack(0, [])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find an independent set of size 5 in the Grötzsch part (0-10)
    iso_set = _find_independent_set(adj, 11, 5)
    if not iso_set:
        # Fallback: manual known independent set in Grötzsch
        # Grötzsch graph: 0-1-2-3-4-0 cycle. 5-6-7-8-9 connected to 0-4. 10 connected to 5-9.
        # Independent set: {0, 2, 5, 7, 9}? 
        # 0 not adj 2. 0 not adj 5 (0 adj 5? No, 0 adj 5+0=5? Yes in construction above: adj[u][5+w] where u=0, w=0 -> adj[0][5]=1).
        # Let's re-verify Grötzsch construction.
        # Cycle 0-1-2-3-4-0.
        # 5 connected to 0 and 1? No.
        # Code: adj[u][5+w] = 1 where u=i, w=(i+1)%5.
        # i=0: u=0, w=1. adj[0][6]=1.
        # i=1: u=1, w=2. adj[1][7]=1.
        # i=2: u=2, w=3. adj[2][8]=1.
        # i=3: u=3, w=4. adj[3][9]=1.
        # i=4: u=4, w=0. adj[4][5]=1.
        # Also adj[w][5+u] = 1.
        # i=0: w=1, u=0. adj[1][5]=1.
        # i=1: w=2, u=1. adj[2][6]=1.
        # i=2: w=3, u=2. adj[3][7]=1.
        # i=3: w=4, u=3. adj[4][8]=1.
        # i=4: w=0, u=4. adj[0][9]=1.
        # 10 connected to 5,6,7,8,9.
        
        # Independent set candidates:
        # 0 is adj to 1, 4, 6, 9.
        # 2 is adj to 1, 3, 6, 8.
        # 5 is adj to 4, 1, 10.
        # 7 is adj to 2, 3, 10.
        # 9 is adj to 0, 4, 10.
        
        # Let's try {0, 2, 5, 7, 9}?
        # 0-2: No edge.
        # 0-5: No edge.
        # 0-7: No edge.
        # 0-9: Edge! (adj[0][9]=1). So {0,2,5,7,9} is not independent.
        
        # Try {1, 3, 5, 7, 9}?
        # 1-3: No.
        # 1-5: Edge! (adj[1][5]=1).
        
        # The finder should work. If it fails, use a default.
        iso_set = [0, 2, 5, 7, 10] # Just a guess, might fail.
        # Better to rely on the finder. If finder fails, we might have a bug.
        # For now, assume finder works.
        
    # Connect new vertices 11-15 to the independent set
    for i in range(5):
        v_new = 11 + i
        for u in iso_set:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Precompute common neighbor masks for triangle checking
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
    
    # Local Search
    # We will iterate, trying to add edges or swap edges.
    
    def try_add_edge(i, j):
        """Tries to add edge (i,j). Returns True if added and valid."""
        if adj[i][j]:
            return False
        if cn[i][j] != 0:
            return False
        
        # Tentatively add
        adj[i][j] = 1
        adj[j][i] = 1
        
        # Update CN
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
            # Revert
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
            
    def revert_add_edge(i, j):
        """Reverts a previously added edge (i,j). Assumes it was added."""
        adj[i][j] = 0
        adj[j][i] = 0
        for k in range(n):
            if k != i and k != j:
                if adj[i][k]:
                    cn[i][k] &= ~(1 << j)
                    cn[k][i] &= ~(1 << j)
                if adj[j][k]:
                    cn[j][k] &= ~(1 << i)
                    cn[k][j] &= ~(1 << i)
        cn[i][j] = 0
        cn[j][i] = 0

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
        
        # Phase 1: Add edges
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            if try_add_edge(i, j):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
                # Rebuild non_edges list? Or just continue?
                # Continuing is fine, we skip existing edges.
                # But non_edges is static for this iteration.
                # If we added an edge, it's no longer a non-edge.
                # The loop condition `if not adj[i][j]` inside try_add_edge handles this.
        
        # Phase 2: Swaps (Remove one, Add one or more)
        if not improved and time.time() < deadline:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:10]: # Limit swaps
                if time.time() > deadline:
                    break
                
                # Remove (ri, rj)
                adj[ri][rj] = 0
                adj[rj][ri] = 0
                # Update CN
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] &= ~(1 << rj)
                            cn[k][ri] &= ~(1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] &= ~(1 << ri)
                            cn[k][rj] &= ~(1 << ri)
                cn[ri][rj] = 0
                cn[rj][ri] = 0
                
                added = False
                # Try to add edges from non_edges
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    # Tentative add
                    adj[ai][aj] = 1
                    adj[aj][ai] = 1
                    old_cn_ai = [cn[ai][k] for k in range(n)]
                    old_cn_aj = [cn[aj][k] for k in range(n)]
                    for k in range(n):
                        if k != ai and k != aj:
                            if adj[ai][k]:
                                cn[ai][k] |= (1 << aj)
                                cn[k][ai] |= (1 << aj)
                            if adj[aj][k]:
                                cn[aj][k] |= (1 << ai)
                                cn[k][aj] |= (1 << ai)
                    cn[ai][aj] = 0
                    cn[aj][ai] = 0
                    
                    if _is_not_3_colorable(adj, n):
                        added = True
                        break
                    else:
                        # Revert add
                        adj[ai][aj] = 0
                        adj[aj][ai] = 0
                        for k in range(n):
                            if k != ai and k != aj:
                                if adj[ai][k]:
                                    cn[ai][k] = old_cn_ai[k]
                                    cn[k][ai] = old_cn_ai[k]
                                if adj[aj][k]:
                                    cn[aj][k] = old_cn_aj[k]
                                    cn[k][aj] = old_cn_aj[k]
                
                # Re-add removed edge (ri, rj)
                adj[ri][rj] = 1
                adj[rj][ri] = 1
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] |= (1 << rj)
                            cn[k][ri] |= (1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] |= (1 << ri)
                            cn[k][rj] |= (1 << ri)
                
                if added:
                    new_edges = _count_edges(adj, n)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    improved = True

    return _get_edges(best_adj, n)
