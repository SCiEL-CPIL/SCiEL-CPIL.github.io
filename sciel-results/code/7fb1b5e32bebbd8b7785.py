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

def _local_search(n, adj_masks, deadline):
    """
    Attempt to repair the graph using local search.
    Returns True if a valid graph is found (modifies adj_masks in place).
    """
    # Initial check
    if _check_graph_validity(n, adj_masks):
        return True
        
    for step in range(10000):
        if time.time() > deadline:
            return False
            
        # Find a violation of size 6
        viol_mask = _find_violation_size_k(n, adj_masks, 6)
        
        if viol_mask == 0:
            # No size 6 violations, check larger sizes
            if _check_graph_validity(n, adj_masks):
                return True
            # If there are larger violations, we need to handle them.
            found_larger = False
            for k in range(7, n + 1):
                viol_mask = _find_violation_size_k(n, adj_masks, k)
                if viol_mask != 0:
                    found_larger = True
                    break
            if not found_larger:
                return True
                
            # If we found a larger violation, break it by flipping an edge inside it
        else:
            pass
        
        # Flip a random edge within the violation mask
        vertices = []
        m = viol_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
            
        if len(vertices) < 2:
            continue
            
        i, j = random.sample(vertices, 2)
        
        mask_bit_i = 1 << j
        mask_bit_j = 1 << i
        adj_masks[i] ^= mask_bit_i
        adj_masks[j] ^= mask_bit_j

    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try 22, then 21, then 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline:
            break
            
        attempts = 0
        while attempts < 5000 and time.time() < deadline:
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
                
                return {'n': target_n, 'edges': edges}
        
        # If we didn't find it for this n, try smaller?
        # The loop structure will move to the next target_n.
        
    # Fallback: return a small valid graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}
