import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def check_regular(mask, adj_masks, n):
        """Check if the induced subgraph on 'mask' is regular."""
        # Get vertices in mask
        vertices = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if len(vertices) < 6:
            return False
            
        degs = []
        for v in vertices:
            degs.append((adj_masks[v] & mask).bit_count())
            
        # Check if all degrees are equal
        first_deg = degs[0]
        for d in degs[1:]:
            if d != first_deg:
                return False
        return True

    def find_violation_size(adj_masks, n, target_size):
        """Find a regular induced subgraph of exactly 'target_size' vertices.
        Returns the mask if found, None otherwise."""
        vertices = list(range(n))
        for subset in itertools.combinations(vertices, target_size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if check_regular(mask, adj_masks, n):
                return mask
        return None

    def full_check(adj_masks, n):
        """Check all subsets of size >= 6. Returns (is_valid, violating_mask, violating_size)"""
        vertices = list(range(n))
        # Check sizes from 6 to n
        for k in range(6, n + 1):
            for subset in itertools.combinations(vertices, k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if check_regular(mask, adj_masks, n):
                    return False, mask, k
        return True, None, None

    def verify_edges(edges, n):
        if n < 6:
            return True, None, None
        
        adj_masks = [0] * n
        for e in edges:
            u, v = e
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        return full_check(adj_masks, n)

    best_result = {'n': 5, 'edges': []}
    
    # Try to find a valid graph for n=20, then 19, etc.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
            
        # Local Search
        # Start with a random graph
        p = 0.5
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    edges.append([i, j])
        
        adj_masks = [0] * n
        for e in edges:
            u, v = e
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Track which sizes to check in local search
        # Start with size 6
        sizes_to_check = [6]
        
        max_iterations = 1000
        for it in range(max_iterations):
            if time.time() > deadline:
                break
                
            violation_found = False
            # Check each size in sizes_to_check
            for k in sizes_to_check:
                mask = find_violation_size(adj_masks, n, k)
                if mask is not None:
                    violation_found = True
                    # Flip a random edge within the violating subgraph
                    # Get vertices in mask
                    verts_in_mask = []
                    m = mask
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        verts_in_mask.append(v)
                        m ^= lsb
                    
                    # Pick two random vertices from the subset
                    v1, v2 = random.sample(verts_in_mask, 2)
                    
                    # Check if edge exists
                    if adj_masks[v1] & (1 << v2):
                        # Remove edge
                        adj_masks[v1] ^= (1 << v2)
                        adj_masks[v2] ^= (1 << v1)
                        # Update edges list (remove)
                        edges = [e for e in edges if not (e[0]==v1 and e[1]==v2) and not (e[0]==v2 and e[1]==v1)]
                    else:
                        # Add edge
                        adj_masks[v1] |= (1 << v2)
                        adj_masks[v2] |= (1 << v1)
                        edges.append([v1, v2])
                    
                    break # Restart checking from size 6
                    
            if not violation_found:
                # No violations found for the current sizes_to_check
                # Do a full check to see if we are truly valid
                # But full check is expensive. 
                # We can do it every few iterations or once at the end.
                # Let's do it once at the end of the local search loop.
                pass
                
        # After local search, do a full check
        if time.time() < deadline - 10: # Leave 10s for full check
            is_valid, viol_mask, viol_size = full_check(adj_masks, n)
            if is_valid:
                best_result = {'n': n, 'edges': edges}
                break
            else:
                # If invalid, we can try to add viol_size to sizes_to_check and restart local search?
                # Or just accept and try next n?
                # Let's try one more round of local search with the new size
                if viol_size not in sizes_to_check:
                    sizes_to_check.append(viol_size)
                    # Restart local search with updated sizes
                    # For simplicity, we just break and try next n if we don't have time
                    # Or we can loop again.
                    # Given time constraints, let's just break and try smaller n if this one fails
                    pass
        else:
            # Time tight, skip full check? No, must be valid.
            # If we don't have time for full check, we can't claim validity.
            # But we can return the graph if we are confident? No, risk of 0 score.
            # Better to return smaller n if we can verify it.
            pass

    # Fallback: return a known small valid graph if n=20 fails
    # A 5-vertex graph is always valid (no subset of size 6).
    # But we want to maximize n.
    # If the loop finished without finding a valid n>=6, best_result is n=5.
    
    return best_result
