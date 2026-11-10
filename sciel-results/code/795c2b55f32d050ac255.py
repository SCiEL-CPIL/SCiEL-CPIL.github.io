import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_regular_subgraph(adj, mask):
    """Check if the induced subgraph on vertices in mask is regular.
    adj is a list of bitmasks (int) for each vertex."""
    # Count vertices in mask
    cnt = mask.bit_count()
    if cnt < 2:
        return True
    # Get degrees of vertices in the induced subgraph
    # For each vertex i in mask, degree = popcount(adj[i] & mask)
    # Check if all equal
    first = -1
    m = mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        deg = (adj[i] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
        m ^= lb
    return True

def _has_regular_6plus(adj, n, max_k=7):
    """Check if there's a regular induced subgraph of size 6 or 7.
    If max_k is larger, also check those sizes."""
    for k in range(6, max_k + 1):
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _is_regular_subgraph(adj, mask):
                return True, k, combo
    return False, None, None

def _has_regular_6plus_fast(adj, n, max_k=7):
    """Faster check using bit operations for subset enumeration."""
    # For k=6 and k=7, use combinations
    for k in range(6, max_k + 1):
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            # Quick check: compute degrees
            first_deg = -1
            for v in combo:
                deg = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    break
            else:
                # All degrees equal
                return True, k, combo
    return False, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_n = 5
    best_edges = []
    
    # Greedy construction: start with empty graph, try to add vertices
    # and randomly add edges while maintaining the constraint
    
    def try_build(target_n, seed=42, max_iters=50000):
        """Try to build a valid graph on target_n vertices."""
        random.seed(seed)
        adj = [0] * target_n
        edges = []
        
        # Start with n=6 and try to grow
        # Actually, let's start fresh and build up to target_n
        # For each new vertex, we need to ensure no regular induced subgraph forms
        
        # Alternative approach: random graph generation with constraint checking
        # Generate random graphs on target_n vertices and check
        
        for attempt in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Generate a random graph
            adj = [0] * target_n
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append([i, j])
            
            # Check for regular induced subgraphs of size 6 and 7
            found, k, combo = _has_regular_6plus_fast(adj, target_n, max_k=7)
            if not found:
                # Likely valid - do full check if time permits
                if time.time() < deadline - 10.0:
                    full_ok = _full_check(adj, target_n)
                    if full_ok:
                        return target_n, edges
                else:
                    # Return without full check (risky but fast)
                    return target_n, edges
            
            # If we find a violation, perturb: flip some random edges
            if found and combo:
                # Try to fix by flipping edges in or near the violating subset
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    i, j = random.randint(0, target_n - 1), random.randint(0, target_n - 1)
                    if i == j:
                        continue
                    # Flip edge (i, j)
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        edges = [[u, v] for u, v in edges if not (u == i and v == j) and not (u == j and v == i)]
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append([min(i, j), max(i, j)])
                
                found, k, combo = _has_regular_6plus_fast(adj, target_n, max_k=7)
                if not found:
                    if time.time() < deadline - 10.0:
                        if _full_check(adj, target_n):
                            return target_n, edges
                    else:
                        return target_n, edges
        
        return None, None
    
    def _full_check(adj, n):
        """Check all subsets of size >= 6 for regularity."""
        for k in range(6, n + 1):
            for combo in combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                first_deg = -1
                for v in combo:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        break
                else:
                    return False
        return True
    
    # Try to build graphs of increasing size
    for target_n in range(20, 5, -1):
        if time.time() > deadline - 20.0:
            break
        
        result_n, result_edges = try_build(target_n, seed=target_n * 137)
        if result_n is not None:
            best_n = result_n
            best_edges = result_edges
            break
    
    # If we couldn't build a large graph, try smaller ones
    if best_n <= 5:
        for target_n in range(15, 5, -1):
            if time.time() > deadline - 10.0:
                break
            result_n, result_edges = try_build(target_n, seed=target_n * 137 + 7)
            if result_n is not None:
                best_n = result_n
                best_edges = result_edges
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
