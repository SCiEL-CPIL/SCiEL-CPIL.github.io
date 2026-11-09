import time
import random
import itertools

def popcount(x):
    return bin(x).count("1")

def _is_regular_subset(neigh, mask, verts):
    """Check if the induced subgraph on verts (given by mask) is regular."""
    if not verts:
        return True
    first_deg = popcount(neigh[verts[0]] & mask)
    for v in verts[1:]:
        if popcount(neigh[v] & mask) != first_deg:
            return False
    return True

def _find_violation(neigh, n):
    """Find a subset of size >= 6 that induces a regular subgraph. Returns (k, verts) or None."""
    # Check size 6 first (most likely to exist if any)
    for verts in itertools.combinations(range(n), 6):
        mask = 0
        for v in verts:
            mask |= (1 << v)
        if _is_regular_subset(neigh, mask, verts):
            return 6, verts
    
    # Check larger sizes (less likely, but must be checked for validity)
    # We can stop early if we find one
    for k in range(7, n + 1):
        for verts in itertools.combinations(range(n), k):
            mask = 0
            for v in verts:
                mask |= (1 << v)
            if _is_regular_subset(neigh, mask, verts):
                return k, verts
    return None

def _local_search(neigh, n, timeout, start_time):
    """
    Local search to minimize the number of regular induced subgraphs of size 6.
    If we reach 0, we check for larger violations and try to fix them.
    Returns (neigh, edges, valid)
    """
    # Precompute subsets of size 6 for fast evaluation
    subsets_6 = list(itertools.combinations(range(n), 6))
    subset_masks = [0] * len(subsets_6)
    for idx, s in enumerate(subsets_6):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subset_masks[idx] = mask
    
    # Map pair (u, v) to indices of subsets containing both
    pair_to_subsets = {}
    for idx, s in enumerate(subsets_6):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                key = (s[i], s[j])
                if key not in pair_to_subsets:
                    pair_to_subsets[key] = []
                pair_to_subsets[key].append(idx)
    
    def count_bad_6():
        count = 0
        for idx, s in enumerate(subsets_6):
            if _is_regular_subset(neigh, subset_masks[idx], s):
                count += 1
        return count
    
    bad_6 = count_bad_6()
    
    # Edge set for tracking
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if neigh[i] & (1 << j):
                edge_set.add((i, j))
    
    best_bad_6 = bad_6
    best_neigh = neigh[:]
    best_edge_set = set(edge_set)
    
    steps = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout - 2:
        if bad_6 == 0:
            # Check if graph is fully valid
            viol = _find_violation(neigh, n)
            if viol is None:
                edges = [[u, v] for u, v in edge_set]
                return neigh, edges, True
            
            # If not valid, try to break the specific violation
            # Pick a random edge within the violating set and flip it?
            # Or just continue search.
            # Let's try to flip an edge inside the violating set to break regularity.
            k, verts = viol
            # Pick a random pair in verts
            pair = random.sample(verts, 2)
            u, v = pair[0], pair[1]
            key = (min(u, v), max(u, v))
            
            # Flip this specific edge
            if key in edge_set:
                neigh[u] ^= (1 << v)
                neigh[v] ^= (1 << u)
                edge_set.remove(key)
            else:
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
                edge_set.add(key)
            
            # Recalculate bad_6 (might have introduced new ones)
            # This is expensive, but bad_6 was 0, so we expect it to stay low or become small
            # Actually, let's just do a full recount if we are in this state
            old_bad_6 = bad_6
            bad_6 = count_bad_6()
            
            if bad_6 < best_bad_6:
                best_bad_6 = bad_6
                best_neigh = neigh[:]
                best_edge_set = set(edge_set)
                no_improve = 0
            else:
                # Revert if it didn't help reduce bad_6? 
                # No, we want to break the larger violation. 
                # If bad_6 increased, we might be stuck. 
                # Let's accept the move if it breaks the large violation, even if bad_6 increases slightly?
                # But we are minimizing bad_6. 
                # Strategy: If bad_6 > 0, minimize it. If bad_6 == 0, try to break large violations.
                # If flipping an edge in the violation increases bad_6, revert?
                # Let's revert if bad_6 > old_bad_6
                if bad_6 > old_bad_6:
                    # Revert
                    if key in edge_set:
                        neigh[u] ^= (1 << v)
                        neigh[v] ^= (1 << u)
                        edge_set.remove(key)
                    else:
                        neigh[u] |= (1 << v)
                        neigh[v] |= (1 << u)
                        edge_set.add(key)
                    bad_6 = old_bad_6
                    no_improve += 1
                    if no_improve > 20:
                        # Restart local search dynamics
                        break
                else:
                    no_improve = 0
            
            steps += 1
            continue
        
        # Normal move: Flip random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        key = (min(u, v), max(u, v))
        affected = pair_to_subsets.get(key, [])
        
        old_bad = 0
        for idx in affected:
            if _is_regular_subset(neigh, subset_masks[idx], subsets_6[idx]):
                old_bad += 1
        
        # Flip
        if key in edge_set:
            neigh[u] ^= (1 << v)
            neigh[v] ^= (1 << u)
            edge_set.remove(key)
        else:
            neigh[u] |= (1 << v)
            neigh[v] |= (1 << u)
            edge_set.add(key)
        
        new_bad = 0
        for idx in affected:
            if _is_regular_subset(neigh, subset_masks[idx], subsets_6[idx]):
                new_bad += 1
        
        delta = new_bad - old_bad
        
        # Accept if improvement, or with small prob if neutral
        if delta < 0 or (delta == 0 and random.random() < 0.1):
            bad_6 += delta
            if bad_6 < best_bad_6:
                best_bad_6 = bad_6
                best_neigh = neigh[:]
                best_edge_set = set(edge_set)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert
            if key in edge_set:
                neigh[u] ^= (1 << v)
                neigh[v] ^= (1 << u)
                edge_set.remove(key)
            else:
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
                edge_set.add(key)
            no_improve += 1
        
        steps += 1
        
        # Restart if stuck
        if no_improve > max_no_improve:
            # Simple restart: randomize some edges? Or full restart?
            # Full restart is safer for finding new basins
            break
    
    # Return best found
    # Check if best is valid
    viol = _find_violation(best_neigh, n)
    if viol is None:
        edges = [[u, v] for u, v in best_edge_set]
        return best_neigh, edges, True
    
    return best_neigh, [[u, v] for u, v in best_edge_set], False

def construct():
    start_time = time.time()
    timeout = 145  # Leave 5s margin
    
    # We aim for n=20. If that fails, try 19, etc.
    # Known: n=20 exists.
    
    # Strategy:
    # 1. Try to construct n=20 using local search with multiple restarts.
    # 2. If successful, return.
    # 3. If time runs out, return best valid graph found (maybe n=19, 18...).
    
    best_valid_n = 5
    best_valid_edges = []
    
    # Fallback valid graph for n=5 (empty graph is not regular? 
    # Wait, empty graph on 5 vertices: induced subgraph on any 6 vertices? 
    # No, n=5 means no subset of size 6 exists. So n=5 is trivially valid.
    # But we want to maximize n.
    # n=5 is always valid.
    # n=6: Empty graph on 6 vertices is regular (degree 0). So n=6 empty is INVALID.
    # We need a specific graph.
    
    # Let's try to find valid graphs for n=20, 19, ... down to 5.
    # We will spend most time on n=20.
    
    time_budget_per_n = [
        (20, 100),  # 100s for n=20
        (19, 20),   # 20s for n=19
        (18, 10),   # 10s for n=18
        (17, 5),    # 5s for n=17
        (16, 2),    # 2s for n=16
        (15, 1),    # 1s for n=15
    ]
    
    for n, budget in time_budget_per_n:
        if time.time() - start_time > timeout - 5:
            break
        
        # Try multiple restarts for this n
        found_valid = False
        edges = []
        
        # Number of restarts depends on budget
        # Estimate time per restart ~ 1s? 
        # Let's just loop until time runs out for this n
        restart_start = time.time()
        
        while time.time() - restart_start < budget:
            # Initialize random graph
            neigh = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        neigh[i] |= (1 << j)
                        neigh[j] |= (1 << i)
            
            # Run local search
            neigh, edges, valid = _local_search(neigh, n, timeout, start_time)
            
            if valid:
                best_valid_n = n
                best_valid_edges = edges
                found_valid = True
                break
            
            # If not valid, check if we improved best_valid_n? No, only valid counts.
            # But we can keep the best "almost valid" to maybe fix later? 
            # For now, just restart.
            
            # Check time
            if time.time() - start_time > timeout - 5:
                break
        
        if found_valid:
            # Found a valid graph for this n. 
            # Since we iterate n from 20 down, this is the max n found so far.
            # We can return immediately if n=20.
            # If n<20, we should try smaller n? No, we want MAX n.
            # So if we found valid n=19, we should try n=20 first.
            # The loop starts at 20. So if we found valid 20, we return.
            # If we failed 20, we try 19. If we find valid 19, we store it.
            # Then we try 18? No, 19 is better than 18.
            # So we should stop trying smaller n once we find a valid one?
            # Yes.
            return {'n': best_valid_n, 'edges': best_valid_edges}
    
    # If no valid graph found for n >= 6, return n=5
    return {'n': best_valid_n, 'edges': best_valid_edges}

# Quick test / main
if __name__ == "__main__":
    # Just to ensure it runs
    res = construct()
    print(res['n'])
