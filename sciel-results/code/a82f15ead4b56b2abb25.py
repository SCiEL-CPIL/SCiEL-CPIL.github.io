import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_independent for all masks
    # is_clique[mask]: True if all vertices in mask form a clique
    # is_independent[mask]: True if all vertices in mask form an independent set
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    
    is_clique[0] = True
    is_independent[0] = True
    
    # Iterate through all masks
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be a clique
        # v is connected to all in rest if (rest & ~adj[v]) == 0
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        else:
            is_clique[mask] = False
            
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        # v is not connected to any in rest if (rest & adj[v]) == 0
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
        else:
            is_independent[mask] = False
            
    is_valid_part = [is_clique[m] or is_independent[m] for m in range(1 << n)]
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Backtracking search
    # We want to find min k such that V can be partitioned into k valid parts.
    # Symmetry breaking: The next part must contain the smallest vertex in remaining_mask.
    
    def search(remaining_mask, current_partition, k_limit):
        if remaining_mask == 0:
            return True
        
        # If we have used k_limit parts, we must have finished
        if len(current_partition) == k_limit:
            return False
        
        # Lower bound: each part can have at most max_part_size vertices
        # But we can be smarter: the remaining vertices must fit in (k_limit - len(current_partition)) parts
        remaining_count = popcount[remaining_mask]
        parts_left = k_limit - len(current_partition)
        
        # Quick check: if remaining_count > parts_left * max_possible_part_size, fail
        # But max possible part size depends on the graph structure.
        # A simpler bound: if remaining_count > parts_left * n, fail (trivially true)
        # Better: just proceed and let the search handle it.
        
        # Pick the smallest vertex in remaining_mask
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        # Collect all valid parts containing v, sorted by size descending
        rest = remaining_mask ^ lsb
        candidates = []
        sub = rest
        while True:
            S = sub | lsb
            if is_valid_part[S]:
                candidates.append((popcount[S], S))
            if sub == 0:
                break
            sub = (sub - 1) & rest
        
        # Sort by size descending - larger parts first to reduce branching
        candidates.sort(key=lambda x: -x[0])
        
        for _, S in candidates:
            current_partition.append(S)
            if search(remaining_mask ^ S, current_partition, k_limit):
                return True
            current_partition.pop()
        
        return False

    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Compute a better lower bound for k
    # Each part is either a clique or independent set
    # Lower bound: max(ceil(n / max_clique_size), ceil(n / max_independent_size))
    # But we need to be careful - different parts can be clique or independent
    
    # Find maximum clique size and maximum independent set size
    max_clique = 0
    max_indep = 0
    for m in range(1, 1 << n):
        if is_clique[m]:
            max_clique = max(max_clique, popcount[m])
        if is_independent[m]:
            max_indep = max(max_indep, popcount[m])
    
    # Lower bound: at least ceil(n / max(max_clique, max_indep)) parts needed
    import math
    lb = math.ceil(n / max(max_clique, max_indep))
    # Also, k >= 1
    lb = max(lb, 1)
    
    # Try k = lb, lb+1, ...
    for k in range(lb, n + 1):
        if time.time() - start_time > deadline:
            # Fallback to a greedy solution if exact search takes too long
            # This should not happen for n=14
            return _greedy_construct()
        
        partition = []
        if search(full_mask, partition, k):
            # Convert masks to lists of vertices
            result = []
            for mask in partition:
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                result.append(verts)
            
            # Ensure types are correct (list of lists of ints)
            # The problem requires list or tuple. We return list of lists.
            # Integers must be Python int.
            
            # Verify validity one last time
            edge_set = set()
            for u, v in P['edges']:
                edge_set.add((min(u,v), max(u,v)))
            
            all_verts = set()
            for part in result:
                for u in part:
                    if u in all_verts:
                        return _greedy_construct()
                    all_verts.add(u)
                if len(part) == 0:
                    return _greedy_construct()
            
            if all_verts != set(range(n)):
                return _greedy_construct()
            
            for part in result:
                is_cli = True
                is_ind = True
                for i in range(len(part)):
                    for j in range(i+1, len(part)):
                        u, v = part[i], part[j]
                        e = (min(u,v), max(u,v))
                        if e in edge_set:
                            is_ind = False
                        else:
                            is_cli = False
                if not (is_cli or is_ind):
                    return _greedy_construct()
            
            return result

    # If no solution found (should not happen), return greedy
    return _greedy_construct()

def _greedy_construct():
    """Fallback greedy constructor if exact search fails or times out."""
    n = P['n']
    es = {tuple(e) for e in P['edges']}
    es.add(tuple(reversed(e)) for e in P['edges']) # Make symmetric for quick check? No, just check both ways or use set of frozensets
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(frozenset((u, v)))
        
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part as clique or independent
            is_clique_so_far = True
            is_independent_so_far = True
            
            # Check against existing vertices
            for u in part:
                if frozenset((u, v)) in edge_set:
                    is_independent_so_far = False
                else:
                    is_clique_so_far = False
            
            if is_clique_so_far or is_independent_so_far:
                part.append(v)
                placed = True
                break
        
        if not placed:
            parts.append([v])
            
    return parts

# EVOLVE-BLOCK-END
