"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _build_bitmasks(n, adj):
    """Build bitmask representation of adjacency for fast clique checking."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks


def _has_k5_fast(n, adj, masks):
    """Check if graph has a clique of size 5 using bitmask operations."""
    for v in range(n):
        neigh_mask = masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (masks[ni] & (1 << nj)):
                    continue
                for k in range(j+1, len(neigh)):
                    nk = neigh[k]
                    if not (masks[ni] & (1 << nk)) or not (masks[nj] & (1 << nk)):
                        continue
                    common = neigh_mask & masks[ni] & masks[nj] & masks[nk]
                    if common:
                        return True
    return False


def _has_ind5_fast(n, adj, masks):
    """Check if graph has an independent set of size 5."""
    all_mask = (1 << n) - 1
    comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
    
    for v in range(n):
        neigh_mask = comp_masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (comp_masks[ni] & (1 << nj)):
                    continue
                for k in range(j+1, len(neigh)):
                    nk = neigh[k]
                    if not (comp_masks[ni] & (1 << nk)) or not (comp_masks[nj] & (1 << nk)):
                        continue
                    common = neigh_mask & comp_masks[ni] & comp_masks[nj] & comp_masks[nk]
                    if common:
                        return True
    return False


def _is_valid_fast(n, adj, masks):
    """Check if graph is (5,5)-free using fast bitmask methods."""
    if _has_k5_fast(n, adj, masks):
        return False
    if _has_ind5_fast(n, adj, masks):
        return False
    return True


def _find_k4s(n, adj, masks):
    """Find all K4s in the graph. Returns list of 4-tuples."""
    k4s = []
    for v in range(n):
        neigh_mask = masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 3:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (masks[ni] & (1 << nj)):
                    continue
                common = neigh_mask & masks[ni] & masks[nj]
                if common & (1 << ni) == 0:  # ensure we don't count v itself
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        idx = lsb.bit_length() - 1
                        if idx > nj and (masks[ni] & (1 << idx)) and (masks[nj] & (1 << idx)):
                            k4s.append((v, ni, nj, idx))
                        cm ^= lsb
    return k4s


def _find_ind4s(n, adj, masks):
    """Find all independent 4-sets in the graph. Returns list of 4-tuples."""
    all_mask = (1 << n) - 1
    comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
    ind4s = []
    for v in range(n):
        neigh_mask = comp_masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 3:
            continue
        
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i+1, len(neigh)):
                nj = neigh[j]
                if not (comp_masks[ni] & (1 << nj)):
                    continue
                common = neigh_mask & comp_masks[ni] & comp_masks[nj]
                if common & (1 << ni) == 0:
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        idx = lsb.bit_length() - 1
                        if idx > nj and (comp_masks[ni] & (1 << idx)) and (comp_masks[nj] & (1 << idx)):
                            ind4s.append((v, ni, nj, idx))
                        cm ^= lsb
    return ind4s


def _get_safe_connections(n, adj, masks, k4s, ind4s):
    """
    For each vertex, determine which subsets of other vertices it can safely connect to.
    Returns a dict mapping vertex -> set of forbidden connection patterns.
    
    Actually, let's think differently: for a new vertex v_new, it can connect to a subset S of existing vertices
    as long as:
    1. S does not contain a K4 (otherwise v_new + K4 = K5)
    2. The complement of S (among existing vertices) does not contain an independent 4-set
       (otherwise v_new + ind4 = ind5)
    
    So we need to find all subsets S such that:
    - S contains no K4
    - V\S contains no independent 4-set
    """
    pass


def _try_add_vertex_constrained(current_n, current_adj, current_masks, k4s, ind4s, deadline):
    """
    Try to add a vertex using constraint-based approach.
    The new vertex's neighbors must not form a K4, and its non-neighbors must not contain an ind4.
    """
    new_vertex = current_n
    all_mask = (1 << current_n) - 1
    
    # Pre-compute: for each subset of size up to 4, check if it's a K4 or ind4
    # But that's too many subsets. Instead, use the known K4s and ind4s.
    
    # Strategy: try random subsets and check constraints
    # A subset S is valid if:
    # 1. No K4 is entirely contained in S
    # 2. No ind4 is entirely contained in (V \ S)
    
    # For efficiency, we'll use bitmask operations
    k4_masks = [0] * len(k4s)
    for idx, k4 in enumerate(k4s):
        m = 0
        for v in k4:
            m |= (1 << v)
        k4_masks[idx] = m
    
    ind4_masks = [0] * len(ind4s)
    for idx, ind4 in enumerate(ind4s):
        m = 0
        for v in ind4:
            m |= (1 << v)
        ind4_masks[idx] = m
    
    # Try different densities and random samples
    densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for density in densities:
        for _ in range(30):
            if time.time() > deadline:
                return False, current_n, current_adj, current_masks
            
            # Generate random connection pattern
            conn_mask = 0
            for i in range(current_n):
                if random.random() < density:
                    conn_mask |= (1 << i)
            
            # Check constraint 1: no K4 entirely in conn_mask
            valid = True
            for km in k4_masks:
                if (km & conn_mask) == km:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check constraint 2: no ind4 entirely in (~conn_mask)
            non_conn_mask = all_mask ^ conn_mask
            for im in ind4_masks:
                if (im & non_conn_mask) == im:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid! Build the new graph
            new_adj = [s.copy() for s in current_adj]
            new_adj.append(set())
            
            m = conn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                new_adj[new_vertex].add(idx)
                new_adj[idx].add(new_vertex)
                m ^= lsb
            
            new_masks = _build_bitmasks(current_n + 1, new_adj)
            # Final verification
            if _is_valid_fast(current_n + 1, new_adj, new_masks):
                return True, current_n + 1, new_adj, new_masks
    
    return False, current_n, current_adj, current_masks


def _try_flip_edge(current_n, current_adj, current_masks, deadline):
    """
    Try flipping a random edge. Returns (success, new_adj, new_masks).
    """
    i = random.randint(0, current_n - 1)
    j = random.randint(0, current_n - 1)
    if i == j:
        return False, current_adj, current_masks
    
    new_adj = [s.copy() for s in current_adj]
    
    if j in new_adj[i]:
        new_adj[i].remove(j)
        new_adj[j].remove(i)
    else:
        new_adj[i].add(j)
        new_adj[j].add(i)
    
    new_masks = _build_bitmasks(current_n, new_adj)
    if _is_valid_fast(current_n, new_adj, new_masks):
        return True, new_adj, new_masks
    
    return False, current_adj, current_masks


def _try_expand(n, adj, max_n, deadline):
    """
    Try to expand the graph by adding vertices and flipping edges.
    """
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    current_n = n
    current_adj = [s.copy() for s in adj]
    current_masks = _build_bitmasks(n, adj)
    
    # Pre-compute K4s and ind4s for constraint-based vertex addition
    k4s = _find_k4s(current_n, current_adj, current_masks)
    ind4s = _find_ind4s(current_n, current_adj, current_masks)
    
    add_streak = 0
    flip_streak = 0
    max_streak = 15
    
    while current_n < max_n:
        if time.time() > deadline:
            break
        
        # Decide whether to try adding or flipping
        if add_streak < max_streak and (flip_streak > 8 or random.random() < 0.65):
            # Try to add a vertex using constraint-based approach
            success, new_n, new_adj, new_masks = _try_add_vertex_constrained(
                current_n, current_adj, current_masks, k4s, ind4s, deadline
            )
            if success:
                current_n = new_n
                current_adj = new_adj
                current_masks = new_masks
                # Recompute K4s and ind4s
                k4s = _find_k4s(current_n, current_adj, current_masks)
                ind4s = _find_ind4s(current_n, current_adj, current_masks)
                if current_n > best_n:
                    best_n = current_n
                    best_adj = [s.copy() for s in current_adj]
                add_streak += 1
                flip_streak = 0
            else:
                add_streak += 1
                if add_streak > max_streak:
                    add_streak = 0
                    flip_streak = 0
        else:
            # Try edge flipping
            success, new_adj, new_masks = _try_flip_edge(
                current_n, current_adj, current_masks, deadline
            )
            if success:
                current_adj = new_adj
                current_masks = new_masks
                # Recompute K4s and ind4s
                k4s = _find_k4s(current_n, current_adj, current_masks)
                ind4s = _find_ind4s(current_n, current_adj, current_masks)
                flip_streak += 1
                add_streak = 0
            else:
                flip_streak += 1
                if flip_streak > max_streak:
                    flip_streak = 0
                    add_streak = 0
        
        # Occasionally reset streaks to explore
        if (add_streak + flip_streak) % 25 == 0:
            add_streak = 0
            flip_streak = 0
    
    return best_n, best_adj


def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_37_edges()
    adj = _build_adjacency(n, edges)
    
    # Verify it's valid
    masks = _build_bitmasks(n, adj)
    if not _is_valid_fast(n, adj, masks):
        return n, edges
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to expand
    expanded_n, expanded_adj = _try_expand(n, adj, 46, deadline)
    if expanded_n > best_n:
        best_n = expanded_n
        best_adj = expanded_adj
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if j in best_adj[i]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
