import time
import random
import itertools

def _fast_validity(orientation, n):
    """Fast validity check using bitsets. Returns True if every 4-subset has a dominator."""
    P = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                P[j] |= (1 << i)
            else:
                P[i] |= (1 << j)
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    target = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    found = False
                    for v in range(n):
                        if (P[v] & target) == target:
                            found = True
                            break
                    if not found:
                        return False
    return True

def _pale_tournament(p):
    """Construct Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j iff (j - i) % p is a square
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _random_tournament(n):
    """Construct a random tournament on n vertices."""
    orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    return orientation

def _orientation_to_bitsets(orientation, n):
    """
    Convert orientation list to list of bitsets for predecessors.
    P[v] is a bitset where bit u is 1 if u -> v.
    """
    P = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j, so i is a predecessor of j
                P[j] |= (1 << i)
            else:
                # j -> i, so j is a predecessor of i
                P[i] |= (1 << j)
    return P

def _count_unique_dominators(P, n, active_mask):
    """
    Count for each vertex v how many 4-subsets S (subset of active vertices)
    have v as their unique dominator.
    Optimized with early termination and bit operations.
    """
    counts = [0] * n
    active_vertices = [i for i in range(n) if (active_mask >> i) & 1]
    m = len(active_vertices)
    
    if m < 4:
        return counts
    
    # Precompute P for active vertices only for faster access
    P_active = [P[v] for v in active_vertices]
    
    # Iterate over all 4-subsets of active vertices
    for i in range(m - 3):
        p1 = P_active[i]
        for j in range(i + 1, m - 2):
            p12 = p1 & P_active[j]
            if p12 == 0:
                continue
            for k in range(j + 1, m - 1):
                p123 = p12 & P_active[k]
                if p123 == 0:
                    continue
                for l in range(k + 1, m):
                    doms = p123 & P_active[l] & active_mask
                    c = doms.bit_count()
                    if c == 1:
                        d = doms.bit_length() - 1
                        counts[d] += 1
    return counts

def _reconstruct_orientation(P, active_mask, n):
    """Reconstruct orientation list from P and active_mask."""
    active_vertices = [i for i in range(n) if (active_mask >> i) & 1]
    new_n = len(active_vertices)
    new_orientation = []
    for i in range(new_n):
        old_i = active_vertices[i]
        for j in range(i + 1, new_n):
            old_j = active_vertices[j]
            # Check direction in original P
            # old_i -> old_j if bit old_i is set in P[old_j]
            if (P[old_j] >> old_i) & 1:
                new_orientation.append(0) # old_i -> old_j
            else:
                new_orientation.append(1) # old_j -> old_i
    return new_n, new_orientation

def _is_valid_tournament(orientation, n):
    """Verify that every 4-subset has a common dominator."""
    P = _orientation_to_bitsets(orientation, n)
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    # Check if any vertex dominates {i,j,k,l}
                    # A vertex v dominates {i,j,k,l} if v->i, v->j, v->k, v->l
                    # i.e., bits i,j,k,l are all set in P[v]
                    target = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    valid = False
                    for v in range(n):
                        if v in (i, j, k, l):
                            continue
                        if (P[v] & target) == target:
                            valid = True
                            break
                    if not valid:
                        return False
    return True

def _prune_tournament(orientation, n, max_restarts=5, time_limit=100.0):
    """
    Attempt to prune the tournament multiple times with different random choices.
    Returns the best (smallest n) orientation found.
    """
    start_time = time.time()
    best_n = n
    best_orientation = orientation
    
    P = _orientation_to_bitsets(orientation, n)
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        active_mask = (1 << n) - 1
        current_n = n
        
        # Pruning loop
        while time.time() < start_time + time_limit:
            counts = _count_unique_dominators(P, n, active_mask)
            
            # Find removable vertices (count == 0)
            removable = [i for i in range(n) if (active_mask >> i) & 1 and counts[i] == 0]
            
            if not removable:
                break
            
            # Greedy heuristic: among removable vertices, pick the one that
            # is the unique dominator of the FEWEST 4-subsets. This is the
            # "safest" removal because it's least critical for coverage.
            # Since all removable have count==0, we use a secondary criterion:
            # prefer vertices with lower degree (fewer edges incident), as they
            # are less likely to be needed as dominators for other subsets.
            if len(removable) == 1:
                v_to_remove = removable[0]
            else:
                # Compute a score for each removable vertex: 
                # number of active vertices it points to (out-degree in active set)
                # Lower out-degree means less "useful" as a dominator
                best_v = removable[0]
                best_score = float('inf')
                for v in removable:
                    # Count how many active vertices v points to
                    out_deg = (P[v] & active_mask).bit_count() if False else 0
                    # Actually P[v] has bits for predecessors of v, not successors
                    # Let's count in-degree instead: how many point to v
                    in_deg = P[v].bit_count()
                    # We want to remove the one that is least useful
                    # Use in-degree as proxy: lower in-degree = less useful as dominator
                    score = in_deg
                    if score < best_score:
                        best_score = score
                        best_v = v
                v_to_remove = best_v
            
            # Update active mask
            active_mask &= ~(1 << v_to_remove)
            current_n -= 1
            
            # Update best if improved
            if current_n < best_n:
                best_n = current_n
                # Defer reconstruction to end of restart
        
        # End of pruning loop for this restart
        if current_n < best_n:
            best_n = current_n
            new_n, new_orientation = _reconstruct_orientation(P, active_mask, n)
            best_orientation = new_orientation

    return best_n, best_orientation

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_n = 90
    best_orientation = []
    
    # Strategy 1: Paley(67) - known to satisfy the property
    p = 67
    paley_orientation = _pale_tournament(p)
    n_p, o_p = _prune_tournament(paley_orientation, p, max_restarts=15, time_limit=50.0)
    if n_p < best_n:
        best_n = n_p
        best_orientation = o_p
        
    if time.time() > deadline:
        return {'n': best_n, 'orientation': best_orientation}
        
    # Strategy 2: Try other prime Paley tournaments
    for p2 in [71, 73, 79]:
        if time.time() > deadline:
            break
        paley_orientation2 = _pale_tournament(p2)
        n_p2, o_p2 = _prune_tournament(paley_orientation2, p2, max_restarts=5, time_limit=15.0)
        if n_p2 < best_n:
            best_n = n_p2
            best_orientation = o_p2
    
    if time.time() > deadline:
        return {'n': best_n, 'orientation': best_orientation}
        
    # Strategy 3: Random(70) - smaller random tournament
    if time.time() < deadline - 10:
        n_r = 70
        random_orientation = _random_tournament(n_r)
        n_r2, o_r = _prune_tournament(random_orientation, n_r, max_restarts=5, time_limit=20.0)
        if n_r2 < best_n:
            best_n = n_r2
            best_orientation = o_r
    
    # Final validation: ensure the best orientation is actually valid
    if best_orientation and best_n > 0:
        if not _fast_validity(best_orientation, best_n):
            # Fallback to unpruned Paley(67)
            best_n = 67
            best_orientation = _pale_tournament(67)
    
    return {'n': best_n, 'orientation': best_orientation}
