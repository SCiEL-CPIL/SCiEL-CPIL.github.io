import itertools
import time
import random

def _is_regular_induced(subset_mask, adj_masks):
    """
    Check if the induced subgraph on vertices in subset_mask is regular.
    Returns True if regular, False otherwise.
    """
    # Get list of vertices in the subset
    vertices = []
    m = subset_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(v)
        m ^= lsb
        
    k = len(vertices)
    if k < 6:
        return False
        
    # Calculate degrees within the induced subgraph
    degs = []
    for v in vertices:
        common = adj_masks[v] & subset_mask
        d = common.bit_count()
        degs.append(d)
        
    # Check if all degrees are equal
    first_deg = degs[0]
    for d in degs[1:]:
        if d != first_deg:
            return False
    return True

def _find_violation_size_k(n, adj_masks, k):
    """
    Find a regular induced subgraph of size k.
    Returns the mask if found, else 0.
    """
    vertices = list(range(n))
    for subset in itertools.combinations(vertices, k):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_masks):
            return mask
    return 0

def _check_graph_validity(n, adj_masks):
    """
    Check if the graph has no regular induced subgraph of size >= 6.
    Returns True if valid.
    """
    for k in range(6, n + 1):
        if _find_violation_size_k(n, adj_masks, k) != 0:
            return False
    return True

def _count_violations_size_k(n, adj_masks, k):
    """
    Count regular induced subgraphs of size k.
    """
    count = 0
    vertices = list(range(n))
    for subset in itertools.combinations(vertices, k):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_regular_induced(mask, adj_masks):
            count += 1
    return count

def _local_search(n, adj_masks, deadline):
    """
    Attempt to repair the graph using local search.
    Returns True if a valid graph is found (modifies adj_masks in place).
    """
    # Initial check
    if _check_graph_validity(n, adj_masks):
        return True
        
    # Focus on size 6 violations primarily, as they are the most common constraint
    # If we eliminate all size 6, we check larger ones.
    
    for step in range(5000):
        if time.time() > deadline:
            return False
            
        # Find a violation of size 6
        viol_mask = _find_violation_size_k(n, adj_masks, 6)
        
        if viol_mask == 0:
            # No size 6 violations, check larger sizes
            if _check_graph_validity(n, adj_masks):
                return True
            # If there are larger violations, we need to handle them.
            # Try to find a violation of size 7, 8, etc.
            found_larger = False
            for k in range(7, n + 1):
                viol_mask = _find_violation_size_k(n, adj_masks, k)
                if viol_mask != 0:
                    found_larger = True
                    break
            if not found_larger:
                return True # Should have been caught by _check_graph_validity
                
            # If we found a larger violation, break it by flipping an edge inside it
            # This might create size 6 violations, but we'll handle them in the next iteration
        else:
            # We have a size 6 violation
            pass
        
        # Flip a random edge within the violation mask
        # Get vertices in viol_mask
        vertices = []
        m = viol_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
            
        if len(vertices) < 2:
            continue
            
        # Pick two random distinct vertices from the violation
        i, j = random.sample(vertices, 2)
        
        # Flip edge (i, j)
        mask_bit_i = 1 << j
        mask_bit_j = 1 << i
        adj_masks[i] ^= mask_bit_i
        adj_masks[j] ^= mask_bit_j

    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 5
    best_edges = []
    
    # We prioritize finding n=20, then n=21
    # But n=21 is harder. We should probably try n=20 first to secure the base score.
    # Then use remaining time for n=21.
    
    # Strategy:
    # 1. Try to find n=20.
    # 2. If found, store it.
    # 3. Try to find n=21.
    # 4. Return best.
    
    found_20 = False
    graph_20 = None
    
    # Time budget allocation: 60s for n=20, 55s for n=21?
    # Or just loop through n=20, 21?
    # Since n=20 is known to exist, we should be able to find it quickly.
    
    for target_n in [20, 21]:
        if time.time() > deadline:
            break
            
        # If we already found 20, we can give 21 less time?
        # Actually, if we found 20, we can spend the rest of the time on 21.
        
        # Random restarts with local search
        attempts = 0
        max_attempts = 100000 # High number, limited by time
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random graph
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Local search
            if _local_search(target_n, adj_masks, deadline):
                # Valid graph found
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                
                if target_n == 20:
                    graph_20 = {'n': 20, 'edges': edges}
                    found_20 = True
                    # Don't break, we want to try 21?
                    # The loop structure will move to 21 next.
                    # But we should probably break out of the while loop for 20
                    # and move to the next target_n in the for loop.
                    break
                elif target_n == 21:
                    return {'n': 21, 'edges': edges}
        
        # If we are in the 20 loop and didn't break, we continue to 21?
        # No, if we didn't find 20, we should probably keep trying 20 or fallback.
        # But 20 is known to exist. If we fail to find it in reasonable time, something is wrong.
        # Let's add a fallback: if we fail 20, try 19?
        # But 20 is the target.
        
    # If we found 20, return it.
    if found_20:
        return graph_20
        
    # If we didn't find 20, maybe we found nothing?
    # Return a small valid graph?
    # n=5 is always valid.
    return {'n': 5, 'edges': []}
