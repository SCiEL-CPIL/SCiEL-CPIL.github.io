import time
import random
import itertools

def count_violations(adj, n):
    """
    Returns (num_K4, num_I4)
    adj is a list of ints (bitmasks)
    """
    num_k4 = 0
    num_i4 = 0
    
    # Precompute edge existence for speed? 
    # adj[i] has bit j set if edge (i,j)
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                # Edge (i,j) exists
                # Find common neighbors k > j
                # common = adj[i] & adj[j] & mask_greater_than_j
                # But we need to count K4s involving (i,j) and two other vertices k,l
                # This approach is complex. 
                # Let's stick to 4-tuple iteration for simplicity and correctness, 
                # as n=35 is small.
                pass
    # Actually, 4-tuple iteration is simpler to write correctly.
    # Let's optimize the 4-tuple loop.
    
    # We can iterate i < j < k < l
    # Check edges:
    # K4: (i,j), (i,k), (i,l), (j,k), (j,l), (k,l) all exist
    # I4: None of these exist
    
    # To speed up, we can break early or use bit tricks.
    # But 50k iterations is small.
    
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    # Check K4
                    if (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[i] & (1 << l)) and \
                       (adj[j] & (1 << k)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        num_k4 += 1
                    # Check I4
                    elif not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[i] & (1 << l)) and \
                         not (adj[j] & (1 << k)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        num_i4 += 1
    return num_k4, num_i4

def has_independent_6(adj, n):
    """Check if there is an independent set of size 6."""
    for combo in itertools.combinations(range(n), 6):
        is_indep = True
        for i in range(6):
            for j in range(i + 1, 6):
                u = combo[i]
                v = combo[j]
                if adj[u] & (1 << v):
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_graph = None
    best_n = 0
    
    # Try from 35 down to 17
    for n in range(35, 16, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Local search for this n
        best_cost = float('inf')
        best_adj = None
        
        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        k4, i4 = count_violations(adj, n)
        cost = k4 + i4
        best_cost = cost
        best_adj = adj[:]
        
        if cost == 0:
            # Verify I6 just in case (though I4=0 implies I6=0)
            if not has_independent_6(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # Local Search
        while time.time() < deadline - 10.0:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_k4, new_i4 = count_violations(adj, n)
            new_cost = new_k4 + new_i4
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj[:]
                if new_cost == 0:
                    if not has_independent_6(adj, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Occasionally restart if stuck? 
            # For now, simple hill climbing.
        
        # If we finished the loop for this n, check best_adj
        if best_cost == 0 and best_adj:
            if not has_independent_6(best_adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # If we found a valid graph, we would have returned.
        # If not, we move to smaller n.
        # But we should keep track of the best valid graph found?
        # The task says "Maximize n". So if we don't find 35, we try 34.
        # If we don't find 34, we try 33...
        # If we find nothing, we return the seed (17).
        
    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
