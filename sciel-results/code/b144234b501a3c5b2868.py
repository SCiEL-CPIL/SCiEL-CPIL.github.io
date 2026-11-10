import itertools
import random
import time

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Returns a dictionary with 'n' and 'edges'.
    """
    
    def is_regular_subgraph(vertices, adj_mask):
        """
        Checks if the induced subgraph on 'vertices' is regular.
        'vertices' is a tuple of vertex indices.
        'adj_mask' is a list of integers where adj_mask[i] is the bitmask of neighbors of i.
        """
        size = len(vertices)
        if size < 6:
            return False
        
        # Compute degrees of vertices in the induced subgraph
        # Use bitmask for speed
        s_mask = 0
        for v in vertices:
            s_mask |= (1 << v)
            
        degs = []
        for v in vertices:
            degs.append((adj_mask[v] & s_mask).bit_count())
            
        return len(set(degs)) == 1

    def find_bad_subset(n, adj_mask, min_size=6):
        """
        Finds a bad subset (regular induced subgraph) of size >= min_size.
        Returns the subset or None.
        """
        # Check size 6 first, then 7, etc.
        # For speed, we only check size 6 and 7 in the search loop.
        # Final verification checks all.
        
        # Check size 6
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subgraph(subset, adj_mask):
                return subset
        # Check size 7
        for subset in itertools.combinations(range(n), 7):
            if is_regular_subgraph(subset, adj_mask):
                return subset
        return None

    def try_solve(n, time_limit, seed):
        random.seed(seed)
        start_time = time.time()
        
        # Initialize random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Local Search / Greedy Repair
        while time.time() - start_time < time_limit:
            bad_subset = find_bad_subset(n, adj_mask, 6)
            if bad_subset is None:
                # No bad subset of size 6 or 7 found.
                # We must verify ALL sizes >= 6 to be sure.
                # This is expensive, so we only do it if 6 and 7 are clean.
                valid = True
                for size in range(8, n + 1):
                    for subset in itertools.combinations(range(n), size):
                        if is_regular_subgraph(subset, adj_mask):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return adj_mask
                else:
                    # Found a larger bad subset, need to fix it
                    # Re-find a bad subset (the loop above found one, but we didn't store it)
                    # Let's just break and restart or fix the specific one?
                    # For simplicity, we'll break and try to fix the specific large one in next iteration?
                    # Actually, find_bad_subset only checks 6 and 7.
                    # If 6 and 7 are clean, but 8 is bad, we need to find it.
                    # Let's modify find_bad_subset to check all? No, too slow.
                    # Let's assume if 6 and 7 are clean, 8+ is rare.
                    # If we find a bad 8+, we flip an edge in it.
                    # Let's re-scan for any bad subset of size >= 6 for the fix step.
                    # This is O(2^n) worst case.
                    
                    # Optimization: If 6 and 7 are clean, we can try flipping random edges to break potential larger regularities?
                    # Or just restart.
                    pass
            
            # If a bad subset (6 or 7) was found, fix it
            if bad_subset is not None:
                # Pick a random vertex in the bad subset and flip a random edge
                u = random.choice(bad_subset)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u, v)
                if (adj_mask[u] >> v) & 1:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            
            # If no bad 6/7 was found, but we didn't return, it means we timed out or 
            # the "valid" check failed due to size > 7.
            # To handle size > 7, we can just continue the loop. 
            # But find_bad_subset won't find it.
            # So we need a way to find larger bad subsets if 6/7 are clean.
            # Let's add a "deep check" mode if 6/7 are clean for a while.
            # For now, if 6/7 are clean, we break and do a full check.
            # If full check fails, we return None (failed).
            
            # Actually, the logic above:
            # 1. Find bad 6/7.
            # 2. If found, flip.
            # 3. If not found, do full check.
            # 4. If full check passes, return.
            # 5. If full check fails, we have a bad subset of size > 7.
            #    We need to find it to fix it.
            #    Let's find it.
            bad_large = None
            for size in range(8, n + 1):
                for subset in itertools.combinations(range(n), size):
                    if is_regular_subgraph(subset, adj_mask):
                        bad_large = subset
                        break
                if bad_large:
                    break
            
            if bad_large:
                u = random.choice(bad_large)
                v = random.randint(0, n - 1)
                if u != v:
                    if (adj_mask[u] >> v) & 1:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                    else:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
            else:
                # Should not happen if valid check failed
                break

        return None

    # Try n=20
    best_adj = None
    # Use a few seeds to ensure reliability
    for seed in [1, 2, 3, 42, 123]:
        adj = try_solve(20, 20, seed)
        if adj:
            best_adj = adj
            break
            
    if best_adj:
        n = 20
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback to n=17 (Paley)
    n = 17
    adj_mask = [0] * n
    # Paley graph construction
    # Quadratic residues mod 17: 1, 4, 9, 16, 8, 2, 15, 13
    residues = set([1, 4, 9, 16, 8, 2, 15, 13])
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Verify n=17
    valid = True
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if is_regular_subgraph(subset, adj_mask):
                valid = False
                break
        if not valid:
            break
            
    if valid:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback to n=10
    n = 10
    random.seed(1)
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Verify n=10
    valid = True
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if is_regular_subgraph(subset, adj_mask):
                valid = False
                break
        if not valid:
            break
            
    if valid:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Last resort
    return {'n': 5, 'edges': []}
