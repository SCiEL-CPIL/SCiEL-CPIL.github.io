import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit

    def has_indep_6(adj_mask, n):
        """Check if there is an independent set of size 6 using bitmask backtracking."""
        # adj_mask[i] is a bitmask of neighbors of i
        
        def bt(cand, k):
            if k == 0:
                return True
            if cand == 0:
                return False
            
            # Pick the lowest set bit
            v = (cand & -cand).bit_length() - 1
            
            # Option 1: Include v
            # Remove v and its neighbors from candidates
            new_cand = cand & ~(1 << v) & ~adj_mask[v]
            if bt(new_cand, k - 1):
                return True
            
            # Option 2: Exclude v
            new_cand = cand & ~(1 << v)
            if bt(new_cand, k):
                return True
            
            return False

        initial_cand = (1 << n) - 1
        return bt(initial_cand, 6)

    def count_k4(adj_mask, n):
        """Count the number of K4s in the graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    for k in range(j + 1, n):
                        if (adj_mask[i] >> k) & 1 and (adj_mask[j] >> k) & 1:
                            # i, j, k form a triangle
                            common = adj_mask[i] & adj_mask[j] & adj_mask[k]
                            # Check if any common neighbor forms a K4
                            # Actually, any common neighbor l > k would form a K4 (i,j,k,l)
                            # But we just need to know if there is ANY common neighbor.
                            # The count of K4s containing triangle (i,j,k) is popcount(common).
                            # We can just sum popcounts.
                            count += bin(common).count('1')
        return count

    def get_score(adj_mask, n):
        """Calculate the score: 100 * K4_count + 1 if indep_6 exists else 0."""
        k4_count = count_k4(adj_mask, n)
        if k4_count > 0:
            return 100 * k4_count
        if has_indep_6(adj_mask, n):
            return 1
        return 0

    def edges_from_mask(adj_mask, n):
        edges = []
        for i in range(n):
            mask = adj_mask[i]
            # Only consider edges to j > i to avoid duplicates
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def try_find_valid(n, max_iters=50000, time_limit=10):
        """Try to find a valid graph on n vertices using local search."""
        # Initialize with random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        current_score = get_score(adj_mask, n)
        if current_score == 0:
            return adj_mask
        
        T = 1.0
        cooling = 0.999
        best_adj = adj_mask[:]
        best_score = current_score
        
        start_loop = time.time()
        for it in range(max_iters):
            if time.time() - start_loop > time_limit:
                break
            
            # Pick a random edge to toggle
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Toggle edge (u, v)
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            
            new_score = get_score(adj_mask, n)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj_mask[:]
                    if best_score == 0:
                        return best_adj
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta < 0:
                    # Should not happen in else
                    pass
                elif random.random() < pow(2.71828, -delta / T):
                    current_score = new_score
            
            T *= cooling
        
        if best_score == 0:
            return best_adj
        return None

    best_g = None
    best_n = 0
    
    # Try to find a valid graph for n=36, then 35, etc.
    # We prioritize larger n.
    for n in range(36, 30, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        found = None
        restarts = 0
        while time.time() < deadline - 10 and found is None and restarts < 5:
            found = try_find_valid(n, max_iters=20000, time_limit=15)
            restarts += 1
        
        if found is not None:
            edges = edges_from_mask(found, n)
            # Final verification
            # Rebuild adj_mask from edges to be safe? No, found is the mask.
            # But we need to return edges.
            # Verify the returned graph is valid.
            adj_check = [0] * n
            for u, v in edges:
                adj_check[u] |= (1 << v)
                adj_check[v] |= (1 << u)
            
            if get_score(adj_check, n) == 0:
                best_g = {'n': n, 'edges': edges}
                best_n = n
                # If we found 36, we can stop or try 37?
                # 37 is very unlikely. Let's try 37 quickly if we have time.
                if n == 36:
                    if time.time() < deadline - 20:
                        found_37 = try_find_valid(37, max_iters=10000, time_limit=10)
                        if found_37 is not None:
                            edges_37 = edges_from_mask(found_37, 37)
                            adj_check_37 = [0] * 37
                            for u, v in edges_37:
                                adj_check_37[u] |= (1 << v)
                                adj_check_37[v] |= (1 << u)
                            if get_score(adj_check_37, 37) == 0:
                                best_g = {'n': 37, 'edges': edges_37}
                                best_n = 37
                    break # Found 36 or 37, stop searching lower n
                else:
                    break # Found 35, stop
    
    if best_g is None:
        # Fallback: return a small valid graph
        # Paley(5) is C5, valid.
        return {'n': 5, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]}
    
    return best_g
