import time
import random

def _make_paley(p):
    """Generate Paley tournament on p vertices (p must be prime)."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _is_valid_fast(orientation, n):
    """Check if tournament is valid using bitmask in-neighborhood intersection."""
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
    
    for a in range(n):
        for b in range(a + 1, n):
            ab = in_masks[a] & in_masks[b]
            if ab == 0:
                continue
            for c in range(b + 1, n):
                abc = ab & in_masks[c]
                if abc == 0:
                    continue
                for d in range(c + 1, n):
                    if (abc & in_masks[d]) == 0:
                        return False
    return True

def _count_violations(orientation, n):
    """Count violating 4-sets using bitmask in-neighborhood intersection."""
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
    
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            ab = in_masks[a] & in_masks[b]
            if ab == 0:
                continue
            for c in range(b + 1, n):
                abc = ab & in_masks[c]
                if abc == 0:
                    continue
                for d in range(c + 1, n):
                    if (abc & in_masks[d]) == 0:
                        count += 1
    return count

def _local_search(orientation, n, time_limit=10.0):
    """
    Local search with incremental violation counting.
    Uses in-neighborhood bitmask intersection for validity checks.
    """
    bits = list(orientation)
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = bits[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
            
    # Build edge map
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    
    # Precompute 4-sets and their initial status
    sets_info = []
    bad_count = 0
    for a in range(n):
        for b in range(a + 1, n):
            ab = in_masks[a] & in_masks[b]
            if ab == 0:
                continue
            for c in range(b + 1, n):
                abc = ab & in_masks[c]
                if abc == 0:
                    continue
                for d in range(c + 1, n):
                    is_bad = (abc & in_masks[d]) == 0
                    if is_bad:
                        bad_count += 1
                    sets_info.append((a, b, c, d, is_bad))
                    
    if bad_count == 0:
        return bits, True

    # Map vertices to 4-sets they belong to
    vertex_to_sets = [[] for _ in range(n)]
    for k, (a, b, c, d, _) in enumerate(sets_info):
        vertex_to_sets[a].append(k)
        vertex_to_sets[b].append(k)
        vertex_to_sets[c].append(k)
        vertex_to_sets[d].append(k)

    start_time = time.time()
    T = float(max(bad_count, 1))
    T_min = 0.1
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if bad_count == 0:
            return bits, True
        
        # Pick a random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx_edge = edge_map[(i, j)]
        
        # Identify affected 4-sets (those containing i or j)
        affected_indices = set(vertex_to_sets[i])
        affected_indices.update(vertex_to_sets[j])
        
        if not affected_indices:
            continue
        
        # Check current status of affected sets
        old_bad_count = 0
        for k in affected_indices:
            if sets_info[k][4]:
                old_bad_count += 1
                
        # Flip edge
        bits[idx_edge] = 1 - bits[idx_edge]
        
        # Update in_masks
        if bits[idx_edge] == 0:
            # i -> j
            in_masks[i] &= ~(1 << j)
            in_masks[j] |= (1 << i)
        else:
            # j -> i
            in_masks[j] &= ~(1 << i)
            in_masks[i] |= (1 << j)
            
        # Check new status of affected sets
        new_bad_count = 0
        for k in affected_indices:
            a, b, c, d, _ = sets_info[k]
            ab = in_masks[a] & in_masks[b]
            if ab == 0:
                is_bad = False
            else:
                abc = ab & in_masks[c]
                if abc == 0:
                    is_bad = False
                else:
                    is_bad = (abc & in_masks[d]) == 0
            sets_info[k] = (a, b, c, d, is_bad)
            if is_bad:
                new_bad_count += 1
                
        delta = new_bad_count - old_bad_count
        bad_count += delta
        
        if bad_count == 0:
            return bits, True
            
        # Accept or reject
        if delta < 0:
            pass  # Accept
        else:
            # Simulated annealing acceptance
            if random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
                pass  # Accept
            else:
                # Revert
                bits[idx_edge] = 1 - bits[idx_edge]
                if bits[idx_edge] == 0:
                    in_masks[i] &= ~(1 << j)
                    in_masks[j] |= (1 << i)
                else:
                    in_masks[j] &= ~(1 << i)
                    in_masks[i] |= (1 << j)
                bad_count -= delta
                # Revert sets_info
                for k in affected_indices:
                    a, b, c, d, _ = sets_info[k]
                    ab = in_masks[a] & in_masks[b]
                    if ab == 0:
                        is_bad = False
                    else:
                        abc = ab & in_masks[c]
                        if abc == 0:
                            is_bad = False
                        else:
                            is_bad = (abc & in_masks[d]) == 0
                    sets_info[k] = (a, b, c, d, is_bad)
                
        T *= cooling
        if T < T_min:
            T = T_min
        
    return bits, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with the known valid Paley tournament on 67 vertices
    orient67 = _make_paley(67)
    
    # Verify n=67
    if _is_valid_fast(orient67, 67):
        best_result = {'n': 67, 'orientation': orient67}
    else:
        # Should not happen, but fallback
        best_result = {'n': 67, 'orientation': orient67}
    
    current_n = 67
    current_orient = orient67
    
    # Try to reduce n by removing vertices and repairing
    for target_n in range(66, 34, -1):
        if time.time() > deadline - 20:
            break
        
        # Initialize A for target_n
        # If we have a valid tournament for target_n + 1, use it.
        if target_n + 1 == current_n:
            # We have a valid tournament for current_n.
            # Remove a random vertex.
            n_curr = current_n
            orient_curr = current_orient
            
            # Convert to adjacency matrix
            A_curr = [[0] * n_curr for _ in range(n_curr)]
            idx = 0
            for i in range(n_curr):
                for j in range(i + 1, n_curr):
                    bit = orient_curr[idx]
                    if bit == 0:
                        A_curr[i][j] = 1
                        A_curr[j][i] = 0
                    else:
                        A_curr[i][j] = 0
                        A_curr[j][i] = 1
                    idx += 1
            
            # Remove a random vertex
            rem_idx = random.randint(0, n_curr - 1)
            mask = [i for i in range(n_curr) if i != rem_idx]
            
            # Build new orientation
            orient_new = []
            for ii in range(target_n):
                for jj in range(ii + 1, target_n):
                    i_orig = mask[ii]
                    j_orig = mask[jj]
                    if A_curr[i_orig][j_orig] == 1:
                        orient_new.append(0)
                    else:
                        orient_new.append(1)
            
            # Verify
            if _is_valid_fast(orient_new, target_n):
                best_result = {'n': target_n, 'orientation': orient_new}
                current_n = target_n
                current_orient = orient_new
                continue
            
            # Repair
            time_left = deadline - time.time() - 5
            if time_left > 2:
                repaired, success = _local_search(orient_new, target_n, time_limit=min(time_left, 15.0))
                if success:
                    best_result = {'n': target_n, 'orientation': repaired}
                    current_n = target_n
                    current_orient = repaired
                    continue
        
        # If we don't have a valid tournament for target_n + 1, or removal/repair failed:
        # Try a fresh start using Paley(target_n)
        if time.time() > deadline - 10:
            break
        
        # Find smallest prime >= target_n
        p = target_n
        while True:
            if p < 2:
                p = 2
            is_prime = True
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime:
                break
            p += 1
        
        # Generate Paley tournament on p vertices
        orient_p = _make_paley(p)
        
        # If p == target_n, use directly
        if p == target_n:
            if _is_valid_fast(orient_p, target_n):
                best_result = {'n': target_n, 'orientation': orient_p}
                current_n = target_n
                current_orient = orient_p
                continue
            
            time_left = deadline - time.time() - 5
            if time_left > 2:
                repaired, success = _local_search(orient_p, target_n, time_limit=min(time_left, 15.0))
                if success:
                    best_result = {'n': target_n, 'orientation': repaired}
                    current_n = target_n
                    current_orient = repaired
                    continue
        else:
            # p > target_n, so we need to remove p - target_n vertices
            # Remove random vertices
            mask = list(range(p))
            random.shuffle(mask)
            mask = mask[:target_n]
            
            # Build new orientation
            orient_new = []
            for ii in range(target_n):
                for jj in range(ii + 1, target_n):
                    i_orig = mask[ii]
                    j_orig = mask[jj]
                    # Find index in orient_p
                    if i_orig < j_orig:
                        idx = i_orig * (p - 1) - (i_orig * (i_orig + 1) // 2) + (j_orig - i_orig - 1)
                    else:
                        idx = j_orig * (p - 1) - (j_orig * (j_orig + 1) // 2) + (i_orig - j_orig - 1)
                    bit = orient_p[idx]
                    # bit 0 means i_orig -> j_orig
                    if bit == 0:
                        orient_new.append(0)
                    else:
                        orient_new.append(1)
            
            # Verify
            if _is_valid_fast(orient_new, target_n):
                best_result = {'n': target_n, 'orientation': orient_new}
                current_n = target_n
                current_orient = orient_new
                continue
            
            # Repair
            time_left = deadline - time.time() - 5
            if time_left > 2:
                repaired, success = _local_search(orient_new, target_n, time_limit=min(time_left, 15.0))
                if success:
                    best_result = {'n': target_n, 'orientation': repaired}
                    current_n = target_n
                    current_orient = repaired
                    continue
        
        # If we still didn't find a valid tournament for this n, 
        # try a few random starts
        for attempt in range(2):
            if time.time() > deadline - 5:
                break
            
            orient_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
            
            time_left = deadline - time.time() - 2
            if time_left > 1:
                repaired, success = _local_search(orient_rand, target_n, time_limit=min(time_left, 5.0))
                if success:
                    best_result = {'n': target_n, 'orientation': repaired}
                    current_n = target_n
                    current_orient = repaired
                    break
        
        # If we still didn't find a valid tournament for target_n, 
        # break because smaller n is harder without a parent
        break
    
    return best_result
