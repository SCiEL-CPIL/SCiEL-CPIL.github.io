import random
import time
import itertools

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Returns a dictionary with 'n' and 'edges'.
    """
    
    def count_regular_6(n, adj):
        """
        Counts the number of induced subgraphs of size 6 that are regular.
        Uses bitmask iteration for speed.
        """
        count = 0
        # Precompute list of vertices for faster iteration? 
        # Iterating over combinations of indices is standard.
        # We can precompute the 6-sets as tuples of indices.
        # But generating them takes time. 
        # Let's just use itertools.combinations but optimize the check.
        
        # Actually, for n=20, C(20,6) is 38760.
        # Generating combinations is the bottleneck? No, the check is.
        # Let's try to generate masks.
        
        # A faster way: iterate all masks from 0 to 2^n - 1? No, too many.
        # Just use combinations.
        
        for subset in itertools.combinations(range(n), 6):
            # Build mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check degrees
            deg0 = (adj[subset[0]] & mask).bit_count()
            if deg0 == 0 or deg0 == 5:
                # 0-regular or 5-regular (complete)
                # Check if all others match
                is_reg = True
                for v in subset[1:]:
                    if (adj[v] & mask).bit_count() != deg0:
                        is_reg = False
                        break
                if is_reg:
                    count += 1
            else:
                # 1-regular, 2-regular, 3-regular, 4-regular
                is_reg = True
                for v in subset[1:]:
                    if (adj[v] & mask).bit_count() != deg0:
                        is_reg = False
                        break
                if is_reg:
                    count += 1
        return count

    def is_valid(n, adj):
        """
        Checks if the graph has NO induced regular subgraph of size >= 6.
        """
        # First check size 6
        if count_regular_6(n, adj) > 0:
            return False
        
        # Check size 7
        # C(20,7) = 77520
        for subset in itertools.combinations(range(n), 7):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = [(adj[v] & mask).bit_count() for v in subset]
            if len(set(degs)) == 1:
                return False
                
        # Check size 8
        # C(20,8) = 125970
        for subset in itertools.combinations(range(n), 8):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = [(adj[v] & mask).bit_count() for v in subset]
            if len(set(degs)) == 1:
                return False
                
        # Check size 9
        # C(20,9) = 167960
        for subset in itertools.combinations(range(n), 9):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = [(adj[v] & mask).bit_count() for v in subset]
            if len(set(degs)) == 1:
                return False
                
        # Check size 10
        # C(20,10) = 184756
        for subset in itertools.combinations(range(n), 10):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = [(adj[v] & mask).bit_count() for v in subset]
            if len(set(degs)) == 1:
                return False
                
        # For n=20, sizes 11-20 are less likely to be regular if 6-10 are not.
        # But we must check all.
        # To save time, we might skip larger sizes in the search phase and only check them in the final verification.
        # However, the task requires validity.
        # Let's assume if 6-10 are clear, it's very likely valid.
        # We will do a full check on the final candidate.
        return True

    def full_verify(n, adj):
        """
        Full verification for sizes 6 to n.
        """
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    return False
        return True

    n = 20
    start_time = time.time()
    time_limit = 110 # Leave 10s for final checks
    
    best_adj = None
    best_score = float('inf')
    
    # Initial random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    current_score = count_regular_6(n, adj)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    steps = 0
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_score = count_regular_6(n, adj)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                if best_score == 0:
                    # Check if it's actually valid (size 7+)
                    # We can do a quick check for size 7
                    # If size 7 is also 0, we are in a good spot
                    # Let's verify fully if we have time
                    if time.time() - start_time < time_limit - 5:
                        if full_verify(n, adj):
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if (adj[i] >> j) & 1:
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        T *= cooling
        steps += 1
        
    # If we didn't find a valid one during search, try to repair the best candidate
    if best_adj is not None:
        adj = best_adj[:]
        # Try a few local repairs
        for _ in range(500):
            if time.time() - start_time > time_limit - 2:
                break
            score = count_regular_6(n, adj)
            if score == 0:
                if full_verify(n, adj):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Find a bad 6-set and break it
            # Sample random 6-sets
            bad_found = False
            for _ in range(20):
                subset = random.sample(range(n), 6)
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    # Break it by flipping a random edge in the subset
                    u, v = random.sample(subset, 2)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    bad_found = True
                    break
            
            if not bad_found:
                # No bad 6-set found in samples, might be 0
                # Check size 7
                bad_7 = False
                for _ in range(20):
                    subset = random.sample(range(n), 7)
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    degs = [(adj[v] & mask).bit_count() for v in subset]
                    if len(set(degs)) == 1:
                        u, v = random.sample(subset, 2)
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        bad_7 = True
                        break
                if not bad_7 and not bad_found:
                    # Verify fully
                    if full_verify(n, adj):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}

    # Fallback to smaller n
    for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() - start_time > time_limit - 2:
            break
        n = target_n
        for _ in range(100):
            if time.time() - start_time > time_limit - 1:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if full_verify(n, adj):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # Last resort: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
