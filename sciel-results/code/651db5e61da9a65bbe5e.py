import time
import random
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free.
    Returns True if it IS K3,3-free (valid), False otherwise.
    Uses bit_count() for speed.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        for j in range(i + 1, n):
            common_ij = adj_bitmasks[i] & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Fast version: only check the new common neighbor introduced by the edge.
    When we add (u,v):
      - For any triple containing u: v is a new common neighbor only if v is also adjacent to the other two vertices in the triple.
      - For any triple containing v: u is a new common neighbor only if u is also adjacent to the other two vertices in the triple.
    So we only need to check: for each pair (i,j) where both are adjacent to v, check if |N(u)∩N(i)∩N(j)| >= 2 (since v would make it 3).
    Similarly for pairs adjacent to u.
    """
    n = len(adj_bitmasks)
    
    # Check triples containing u: for each pair (i,j) both adjacent to v,
    # v becomes a common neighbor of (u,i,j). Need |N(u)∩N(i)∩N(j)| <= 1 (since v adds 1)
    v_neighbors = adj_bitmasks[v] & ~(1 << u)
    
    v_neigh_list = []
    m = v_neighbors
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        v_neigh_list.append(idx)
        m -= lsb
    
    for a_idx in range(len(v_neigh_list)):
        i = v_neigh_list[a_idx]
        if i == u:
            continue
        for b_idx in range(a_idx + 1, len(v_neigh_list)):
            j = v_neigh_list[b_idx]
            if j == u:
                continue
            common = adj_bitmasks[u] & adj_bitmasks[i] & adj_bitmasks[j]
            if common.bit_count() >= 2:
                return False
    
    # Check triples containing v: for each pair (i,j) both adjacent to u,
    # u becomes a new common neighbor of (v,i,j). Need |N(v)∩N(i)∩N(j)| <= 1
    u_neighbors = adj_bitmasks[u] & ~(1 << v)
    
    u_neigh_list = []
    m = u_neighbors
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        u_neigh_list.append(idx)
        m -= lsb
    
    for a_idx in range(len(u_neigh_list)):
        i = u_neigh_list[a_idx]
        if i == v:
            continue
        for b_idx in range(a_idx + 1, len(u_neigh_list)):
            j = u_neigh_list[b_idx]
            if j == v:
                continue
            common = adj_bitmasks[v] & adj_bitmasks[i] & adj_bitmasks[j]
            if common.bit_count() >= 2:
                return False
    
    return True

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = 0
    best_adj_bitmasks = None
    
    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def edges_to_bitmasks(edges):
        bitmasks = [0] * n
        for u, v in edges:
            bitmasks[u] |= (1 << v)
            bitmasks[v] |= (1 << u)
        return bitmasks

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (bitmasks[i] & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def local_search(initial_bitmasks):
        bitmasks = initial_bitmasks[:]
        current_edges = sum(b.bit_count() for b in bitmasks) // 2
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]
        
        stuck_count = 0
        while time.time() < deadline:
            moved = False
            
            # Try to add edges - keep adding until no more can be added
            while True:
                non_edges = get_non_edges(bitmasks)
                if not non_edges:
                    break
                random.shuffle(non_edges)
                added_this_round = False
                for u, v in non_edges:
                    if time.time() > deadline:
                        break
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        added_this_round = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]
                if not added_this_round:
                    break
            
            if current_edges > best_local_edges:
                moved = True
            
            if not moved:
                # Try swap: remove edge (a,b), add edge (c,d)
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    # Try several random swaps
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        if a == c or a == d or b == c or b == d:
                            continue
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if has_k33_fast(bm2):
                                bitmasks = bm2
                                current_edges = sum(b.bit_count() for b in bitmasks) // 2
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_bitmasks = bitmasks[:]
                                break
                
                if not moved:
                    # Try removing a random edge to escape local optimum
                    if all_edges:
                        u, v = random.choice(all_edges)
                        bitmasks[u] &= ~(1 << v)
                        bitmasks[v] &= ~(1 << u)
                        current_edges -= 1
                        moved = True
            
            if not moved:
                stuck_count += 1
                if stuck_count > 50:
                    break
            else:
                stuck_count = 0
                
        return best_local_bitmasks, best_local_edges

    # Run multiple restarts with varied strategies
    num_restarts = 40
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        strategy = r % 4
        init_bitmasks = [0] * n
        
        if strategy == 0:
            p = 0.10 + 0.02 * (r // 4)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
        elif strategy == 1:
            init_bitmasks = [0] * n
        elif strategy == 2:
            p = 0.18 + 0.01 * (r // 4)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
        else:
            if best_adj_bitmasks is not None:
                init_bitmasks = best_adj_bitmasks[:]
                all_e = get_all_edges(init_bitmasks)
                for _ in range(min(5, len(all_e))):
                    if all_e:
                        u, v = random.choice(all_e)
                        init_bitmasks[u] &= ~(1 << v)
                        init_bitmasks[v] &= ~(1 << u)
                        all_e.remove((u, v))
            else:
                p = 0.15
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            init_bitmasks[i] |= (1 << j)
                            init_bitmasks[j] |= (1 << i)
        
        # Repair: Remove random edges until valid
        valid = has_k33_fast(init_bitmasks)
        attempts = 0
        while not valid and attempts < 300:
            all_edges = get_all_edges(init_bitmasks)
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            init_bitmasks[u] &= ~(1 << v)
            init_bitmasks[v] &= ~(1 << u)
            valid = has_k33_fast(init_bitmasks)
            attempts += 1
        
        if not valid:
            continue
            
        # Run local search
        local_best_bitmasks, local_best_edges = local_search(init_bitmasks)
        
        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks

    # If no valid graph found (should not happen), return empty
    if best_adj_bitmasks is None:
        return []
        
    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Verify K3,3 free
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
