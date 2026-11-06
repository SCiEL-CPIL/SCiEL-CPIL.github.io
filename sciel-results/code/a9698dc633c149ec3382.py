import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using fast bitmask backtracking with vertex ordering."""
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending, then by vertex index for determinism
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: (-deg[x], x))
    
    # Map vertex -> position in order
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    # For each vertex v in order, compute bitmask of neighbors that come before v
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        nv = neigh[v]
        for u in range(n):
            if (nv >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        if em & color_masks[0]:
            forbidden |= 1
        if em & color_masks[1]:
            forbidden |= 2
        if em & color_masks[2]:
            forbidden |= 4
        available = 7 & ~forbidden
        if available == 0:
            return False
        # Try colors in order of most constrained first (fewer available colors)
        # Actually, try all available colors
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    # Symmetry breaking: first vertex gets color 0
    color_masks[0] = 1
    return not backtrack(1)

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9: each i connects to 5+(i+1)%5 and 5+(i-1)%5
    for i in range(5):
        adj[i][5+(i+1)%5] = adj[5+(i+1)%5][i] = 1
        adj[i][5+(i-1)%5] = adj[5+(i-1)%5][i] = 1
    # Vertex 10 connects to all of 5-9
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_max_independent_sets(adj, n, count, size):
    """Find 'count' independent sets of size 'size' in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= count:
                break
    return sets

def _initial_graph(n=16):
    """Create a good initial 4-chromatic triangle-free graph."""
    adj = _groetzsch_adj(n)
    
    # Find independent sets of size 5 in the Grötzsch part (vertices 0-10)
    iso_sets = _find_max_independent_sets(adj, 11, 5, 5)
    
    if len(iso_sets) < 5:
        # Fallback: use different independent sets
        while len(iso_sets) < 5:
            base = iso_sets[0] if iso_sets else [0, 2, 5, 7, 10]
            offset = len(iso_sets) % 3
            new_set = [(x + offset) % 11 for x in base]
            if new_set not in iso_sets:
                iso_sets.append(new_set)
            else:
                iso_sets.append(base)
    
    # Connect new vertices 11-15 to the independent sets
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    return adj

def _compute_cn(adj, n):
    """Precompute common neighbor masks for triangle checking."""
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

def _add_edge(adj, cn, n, i, j):
    """Add edge (i,j) and update common neighbor masks. Returns old cn state for rollback."""
    old_cn_i = [cn[i][k] for k in range(n)]
    old_cn_j = [cn[j][k] for k in range(n)]
    
    adj[i][j] = 1
    adj[j][i] = 1
    
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
    
    return old_cn_i, old_cn_j

def _remove_edge(adj, cn, n, i, j):
    """Remove edge (i,j) and update common neighbor masks."""
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

def _rollback_edge(adj, cn, n, i, j, old_cn_i, old_cn_j):
    """Rollback edge addition."""
    adj[i][j] = 0
    adj[j][i] = 0
    for k in range(n):
        cn[i][k] = old_cn_i[k]
        cn[k][i] = old_cn_i[k]
        cn[j][k] = old_cn_j[k]
        cn[k][j] = old_cn_j[k]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n = 16
    adj = _initial_graph(n)
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    cn = _compute_cn(adj, n)
    
    # Local search with multiple phases
    no_improve_count = 0
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Build non-edges list
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        improved = False
        
        # Phase 1: Try adding edges (most promising)
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            # Check triangle using precomputed common neighbors
            if cn[i][j] != 0:
                continue
            
            # Tentatively add
            old_cn_i, old_cn_j = _add_edge(adj, cn, n, i, j)
            
            if _is_not_3_colorable(adj, n):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
                no_improve_count = 0
            else:
                # Revert
                _rollback_edge(adj, cn, n, i, j, old_cn_i, old_cn_j)
        
        # Phase 2: Try remove+add moves if no improvement
        if not improved and time.time() < deadline:
            no_improve_count += 1
            
            # If stuck for a while, do a restart from best
            if no_improve_count > 50:
                adj = [row[:] for row in best_adj]
                cn = _compute_cn(adj, n)
                no_improve_count = 0
                continue
            
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            # Try up to 20 remove+add combinations
            for (ri, rj) in edge_list[:20]:
                if time.time() > deadline:
                    break
                
                # Remove edge (ri, rj)
                _remove_edge(adj, cn, n, ri, rj)
                
                # Try adding non-edges
                added = False
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    old_cn_ai, old_cn_aj = _add_edge(adj, cn, n, ai, aj)
                    
                    if _is_not_3_colorable(adj, n):
                        added = True
                        new_edges = _count_edges(adj, n)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [row[:] for row in adj]
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        _rollback_edge(adj, cn, n, ai, aj, old_cn_ai, old_cn_aj)
                
                # Re-add removed edge
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
                    break
        
        # Phase 3: If still stuck, try a random perturbation
        if not improved and no_improve_count > 30 and time.time() < deadline:
            # Remove 2-3 random edges and re-optimize
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            
            if edge_list:
                remove_count = min(3, len(edge_list))
                to_remove = random.sample(edge_list, remove_count)
                for (ri, rj) in to_remove:
                    _remove_edge(adj, cn, n, ri, rj)
                
                # Now try to add edges back
                non_edges2 = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not adj[i][j]:
                            non_edges2.append((i, j))
                random.shuffle(non_edges2)
                
                for (i, j) in non_edges2:
                    if time.time() > deadline:
                        break
                    if cn[i][j] != 0:
                        continue
                    
                    old_cn_i, old_cn_j = _add_edge(adj, cn, n, i, j)
                    
                    if _is_not_3_colorable(adj, n):
                        new_edges = _count_edges(adj, n)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [row[:] for row in adj]
                        improved = True
                        no_improve_count = 0
                    else:
                        _rollback_edge(adj, cn, n, i, j, old_cn_i, old_cn_j)
    
    return _get_edges(best_adj, n)
