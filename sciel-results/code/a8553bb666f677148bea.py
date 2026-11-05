"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _count_bad_sets(n, orientation):
    """
    Count the number of 4-sets with no common dominator.
    orientation: list of bits, index = i*n - i*(i+1)/2 + (j-i-1) for i<j?
    Let's use a 2D matrix for speed.
    """
    # Build adjacency matrix
    # orientation is in lexicographic order (i, j) with i < j.
    # Index calculation: for fixed i, j ranges from i+1 to n-1.
    # Total elements before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
    # Offset in row i: j - (i+1)
    idx = 0
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            # 0 means i -> j, 1 means j -> i
            if bit == 0:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
            idx += 1
            
    bad_count = 0
    # Iterate over all 4-sets
    # This is O(n^4). For n=40, 40^4 = 2.5M. Python loops are slow.
    # We need to be careful with time.
    # Optimization: Precompute out-neighbors?
    # Or just count and hope n is small enough or we use numpy.
    
    # Let's use a simpler check:
    # A 4-set is bad if no vertex has out-degree 3 within the set.
    # We can iterate and break early?
    
    # For performance, let's limit the search to n <= 50 initially.
    # If n is too large, O(n^4) is too slow for pure Python in 120s.
    # n=40 -> 1M sets. 1M iterations in Python is ~0.1-0.5s.
    # n=50 -> 6M sets. ~1-3s.
    # n=60 -> 13M sets. ~5-10s.
    # So we can try up to n=60.
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if any of a,b,c,d dominates the other 3
                    # a dominates b,c,d iff adj[a][b]==1 and adj[a][c]==1 and adj[a][d]==1
                    if (adj[a][b] and adj[a][c] and adj[a][d]):
                        continue
                    if (adj[b][a] and adj[b][c] and adj[b][d]):
                        continue
                    if (adj[c][a] and adj[c][b] and adj[c][d]):
                        continue
                    if (adj[d][a] and adj[d][b] and adj[d][c]):
                        continue
                    bad_count += 1
    return bad_count

def _local_search(n, max_iters=20000):
    """
    Attempt to find a valid tournament of size n using local search.
    Returns orientation list if found, else None.
    """
    # Initialize random orientation
    m = n * (n - 1) // 2
    orientation = [random.randint(0, 1) for _ in range(m)]
    
    # Helper to get/set edge
    # idx(i, j) for i < j
    # Precompute indices for speed?
    # idx = i*(2*n - i - 1)//2 + (j - i - 1)
    
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
    
    current_cost = _count_bad_sets(n, orientation)
    if current_cost == 0:
        return orientation
        
    best_orientation = orientation[:]
    best_cost = current_cost
    
    # Simple hill climbing with random restarts or just one run?
    # Let's do a few restarts.
    
    for restart in range(5):
        if restart > 0:
            orientation = [random.randint(0, 1) for _ in range(m)]
            current_cost = _count_bad_sets(n, orientation)
            if current_cost < best_cost:
                best_cost = current_cost
                best_orientation = orientation[:]
            if current_cost == 0:
                return orientation
                
        for it in range(max_iters):
            if current_cost == 0:
                return orientation
                
            # Pick a random edge to flip
            idx = random.randint(0, m - 1)
            # Determine i, j from idx?
            # This is expensive to reverse.
            # Better: Pick random i, j.
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j)
            
            # Flip
            old_bit = orientation[idx]
            orientation[idx] = 1 - old_bit
            
            new_cost = _count_bad_sets(n, orientation)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_orientation = orientation[:]
            else:
                # Revert
                orientation[idx] = old_bit
                
            # Early exit if stuck for a long time?
            if it % 1000 == 0 and it > 0:
                if current_cost == best_cost:
                    break
                    
    if best_cost == 0:
        return best_orientation
    return None

def construct():
    # Try to find a smaller tournament
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try n from 35 to 66
    # Start from smaller to larger? Or larger to smaller?
    # Smaller is harder. Start from 40?
    # If we find one, we are done.
    
    # Heuristic: n=67 is known.
    # Try n in range(40, 67):
    for n in range(40, 67):
        if time.time() - start_time > time_limit:
            break
        # Try local search
        # Adjust max_iters based on remaining time
        res = _local_search(n, max_iters=5000)
        if res is not None:
            return {'n': n, 'orientation': res}
            
    # Fallback to Paley tournament on 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
# EVOLVE-BLOCK-END
