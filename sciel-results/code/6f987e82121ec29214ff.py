import random
import time
import itertools

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    """
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute common neighbors
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
            
    def dfs(available, current_mask, current_size):
        nonlocal best
        if current_size + available.bit_count() <= best:
            return
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        best_v_bit = 0
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj_bitmasks[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                best_v_bit = v_bit
            temp -= v_bit
        
        if best_v < 0:
            return
        
        v = best_v
        v_bit = best_v_bit
        
        # Check if v can be added
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            temp_check = N_v_in_cur
            while temp_check:
                u_bit = temp_check & -temp_check
                u = u_bit.bit_length() - 1
                if adj_bitmasks[u] & N_v_in_cur:
                    safe_to_add = False
                    break
                temp_check -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            N_v_in_new = adj_bitmasks[v] & new_mask
            forbidden = 0
            temp_w = N_v_in_new
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                forbidden |= adj_bitmasks[w]
                temp_w -= w_bit
            forbidden &= adj_bitmasks[v]
            forbidden &= available
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
        
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _is_k4_free(bm, n):
    """Check K4-freeness using bitmasks."""
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                common = bm[i] & bm[j]
                while common:
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        return False
                    common -= k_bit
    return True

def _turan_base(n, parts):
    """Create complete multipartite graph with given part sizes."""
    bm = [0] * n
    idx = 0
    for p in parts:
        for i in range(idx, idx + p):
            for j in range(idx + p, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
        idx += p
    return bm

def _edge_flip_k4_check(bm, n, i, j, adding):
    """Check if flipping edge (i,j) maintains K4-freeness."""
    if adding:
        # Check if adding (i,j) creates a K4
        common = bm[i] & bm[j]
        while common:
            k_bit = common & -common
            k = k_bit.bit_length() - 1
            if bm[i] & bm[j] & bm[k]:
                return False
            common -= k_bit
        return True
    else:
        # Removing an edge always maintains K4-freeness
        return True

def _count_tf_subsets_of_size(bm, n, size):
    """Count how many subsets of given size are triangle-free."""
    count = 0
    for subset in itertools.combinations(range(n), size):
        subset_mask = 0
        for v in subset:
            subset_mask |= (1 << v)
        # Check if subset is triangle-free
        is_tf = True
        for a in subset:
            for b in subset:
                if a < b and (bm[a] & (1 << b)):
                    common = bm[a] & bm[b] & subset_mask
                    if common:
                        is_tf = False
                        break
            if not is_tf:
                break
        if is_tf:
            count += 1
    return count

def _local_search_guided(bm, n, max_iter, seed, deadline, target_size=11):
    """
    Clause-guided local search: try edge flips that reduce the number of 
    triangle-free subsets of size target_size+1.
    """
    rng = random.Random(seed)
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    
    # Precompute all (target_size+1)-subsets
    all_subsets = list(itertools.combinations(range(n), target_size + 1))
    
    # For each subset, track which edges are "critical" (removing them makes it TF)
    # and which are "protective" (adding them makes it non-TF)
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random edge to try flipping
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        has_edge = bool(bm[i] & (1 << j))
        
        if has_edge:
            # Try removing edge (i, j)
            bm[i] &= ~(1 << j)
            bm[j] &= ~(1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                # Revert
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
        else:
            # Try adding edge (i, j)
            if _edge_flip_k4_check(bm, n, i, j, True):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    # Revert
                    bm[i] &= ~(1 << j)
                    bm[j] &= ~(1 << i)
            # If adding would create K4, skip
    
    # Convert best_bm to edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _systematic_perturbation(bm, n, deadline):
    """
    Systematically try all single-edge perturbations around the base graph,
    then double-edge perturbations for promising ones.
    """
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Try all single edge additions (that maintain K4-freeness)
    for i, j in all_edges:
        if time.time() > deadline - 10:
            break
        if not (bm[i] & (1 << j)):
            if _edge_flip_k4_check(bm, n, i, j, True):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                score = _max_tf_size_fast(bm, n)
                if score < best_score:
                    best_score = score
                    best_bm = bm[:]
                bm[i] &= ~(1 << j)
                bm[j] &= ~(1 << i)
    
    # Try all single edge removals
    for i, j in all_edges:
        if time.time() > deadline - 10:
            break
        if bm[i] & (1 << j):
            bm[i] &= ~(1 << j)
            bm[j] &= ~(1 << i)
            score = _max_tf_size_fast(bm, n)
            if score < best_score:
                best_score = score
                best_bm = bm[:]
            bm[i] |= (1 << j)
            bm[j] |= (1 << i)
    
    # For the best single perturbation, try double perturbations
    if best_score < 15:  # Only if we improved
        best_bm_copy = best_bm[:]
        for i, j in all_edges:
            if time.time() > deadline - 5:
                break
            if not (best_bm_copy[i] & (1 << j)):
                if _edge_flip_k4_check(best_bm_copy, n, i, j, True):
                    best_bm_copy[i] |= (1 << j)
                    best_bm_copy[j] |= (1 << i)
                    score = _max_tf_size_fast(best_bm_copy, n)
                    if score < best_score:
                        best_score = score
                        best_bm = best_bm_copy[:]
                    best_bm_copy[i] &= ~(1 << j)
                    best_bm_copy[j] &= ~(1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18
    
    # Start with Turán graph T(18,3): three parts of size 6
    parts = [6, 6, 6]
    bm = _turan_base(n, parts)
    
    baseline_score = _max_tf_size_fast(bm, n)
    best_score = baseline_score
    best_bm = bm[:]
    
    # Systematic perturbation around Turán base
    if time.time() < deadline - 30:
        edges, score = _systematic_perturbation(bm, n, deadline)
        if score < best_score:
            best_score = score
            # Reconstruct bm from edges
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Guided local search from the best found so far
    if time.time() < deadline - 20:
        edges, score = _local_search_guided(best_bm, n, max_iter=500, seed=42, deadline=deadline)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Try different Turán-like bases: unbalanced partitions
    alt_parts = [
        [5, 6, 7],
        [4, 7, 7],
        [5, 5, 8],
        [6, 5, 7],
        [7, 5, 6],
    ]
    for alt in alt_parts:
        if time.time() > deadline - 15:
            break
        alt_bm = _turan_base(n, alt)
        alt_score = _max_tf_size_fast(alt_bm, n)
        if alt_score < best_score:
            best_score = alt_score
            best_bm = alt_bm[:]
            # Quick local search from this base
            if time.time() < deadline - 10:
                edges, ls_score = _local_search_guided(best_bm, n, max_iter=200, seed=123, deadline=deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_bm = [0] * n
                    for e in edges:
                        u, v = e
                        best_bm[u] |= (1 << v)
                        best_bm[v] |= (1 << u)
    
    # Final local search polish
    if time.time() < deadline - 5:
        edges, score = _local_search_guided(best_bm, n, max_iter=300, seed=777, deadline=deadline)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Verify K4-freeness
    if not _is_k4_free(best_bm, n):
        # Fallback to Turán base
        best_bm = _turan_base(n, [6, 6, 6])
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    
    return edges
