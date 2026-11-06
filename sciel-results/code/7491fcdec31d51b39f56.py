import random
import itertools
import time

def is_3_colorable(adj_mask, n):
    """
    Check if graph represented by adjacency masks is 3-colorable.
    adj_mask[i] is an integer bitmask of neighbors of i.
    """
    color = [-1] * n
    
    # Order by degree (popcount)
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Get colors of colored neighbors
        used = set()
        mask = adj_mask[v]
        for w in range(n):
            if (mask >> w) & 1:
                if color[w] != -1:
                    used.add(color[w])
        
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def count_edges_mask(adj_mask, n):
    total = 0
    for i in range(n):
        total += bin(adj_mask[i]).count('1')
    return total // 2

def construct():
    n = 16
    best_edges = -1
    best_adj_mask = [0] * n
    
    # Grötzsch graph edges (on vertices 0..10)
    # Cycle 0-1-2-3-4-0
    # Mycielski:
    # 5-1, 5-4? No.
    # Standard Grötzsch:
    # Vertices 0-4 form C5.
    # Vertices 5-9: 5+i connected to i and i+1?
    # Let's use the explicit edge list from the current program's make_grotzsch to be sure.
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Mycielski connections:
    # For edge (u,v) in C5, add 5+u and 5+v?
    # The current program's make_grotzsch:
    # adj[u].add(5 + v); adj[5 + v].add(u)
    # adj[v].add(5 + u); adj[5 + u].add(v)
    # And 10 connected to 5..9.
    
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,6), (1,5), (1,6), (2,6), (2,7), (3,7), (3,8), (4,8), (4,9), (0,9),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    # Wait, let's verify the Mycielski edges in the current code.
    # cycle = [(i, (i + 1) % 5) for i in range(5)] -> (0,1), (1,2), (2,3), (3,4), (4,0)
    # for u, v in cycle:
    #   adj[u].add(5 + v)
    #   adj[5 + v].add(u)
    #   adj[v].add(5 + u)
    #   adj[5 + u].add(v)
    # (0,1): 0-6, 6-0, 1-5, 5-1
    # (1,2): 1-7, 7-1, 2-6, 6-2
    # (2,3): 2-8, 8-2, 3-7, 7-3
    # (3,4): 3-9, 9-3, 4-8, 8-4
    # (4,0): 4-5, 5-4, 0-9, 9-0
    # So edges are:
    # (0,6), (1,5), (1,7), (2,6), (2,8), (3,7), (3,9), (4,8), (4,5), (0,9)
    # Plus cycle edges.
    # Plus 10 connected to 5,6,7,8,9.
    
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,6), (1,5), (1,7), (2,6), (2,8), (3,7), (3,9), (4,8), (4,5), (0,9),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    # Iterate over all subsets of 11 vertices from 16
    # This ensures we don't miss optimal placements
    all_vertices = list(range(n))
    
    # To speed up, we can limit the number of restarts or use a heuristic for subset selection
    # But 4368 is small.
    
    for subset in itertools.combinations(all_vertices, 11):
        # Map Grötzsch vertices 0..10 to subset[0]..subset[10]
        mapping = {i: subset[i] for i in range(11)}
        
        adj_mask = [0] * n
        # Add Grötzsch edges
        valid_seed = True
        for u, v in grotzsch_edges:
            mu = mapping[u]
            mv = mapping[v]
            # Check triangle-free (should be true for Grötzsch)
            if adj_mask[mu] & (1 << mv): continue # already there?
            # Check if adding (mu, mv) creates triangle
            if (adj_mask[mu] & adj_mask[mv]) != 0:
                valid_seed = False
                break
            adj_mask[mu] |= (1 << mv)
            adj_mask[mv] |= (1 << mu)
        
        if not valid_seed:
            continue
            
        # Check if seed is not 3-colorable
        if is_3_colorable(adj_mask, n):
            # Grötzsch is not 3-colorable, so this should be False
            # But if mapping is weird? No, it's an isomorphism.
            continue

        # Local Search / Simulated Annealing
        current_edges = count_edges_mask(adj_mask, n)
        
        # Greedy Phase: Add all possible edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_mask[u] >> v) & 1:
                        continue
                    if (adj_mask[u] & adj_mask[v]) != 0:
                        continue
                    # Try adding
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    if not is_3_colorable(adj_mask, n):
                        current_edges += 1
                        changed = True
                    else:
                        adj_mask[u] ^= (1 << v)
                        adj_mask[v] ^= (1 << u)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_mask = adj_mask[:]
            
        # Simulated Annealing Phase
        # Only do SA if we have time / if the greedy result is promising?
        # Given 120s limit, 4368 subsets * SA might be too slow.
        # We should limit SA to a few iterations or only for the best few seeds.
        
        # Let's do a short local search for all, and SA only if we find a high score.
        # Or just do a random walk for 1000 steps.
        
        temp = 1.0
        for step in range(500):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or ((adj_mask[u] >> v) & 1):
                continue
            if (adj_mask[u] & adj_mask[v]) != 0:
                continue
            
            # Try add
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if not is_3_colorable(adj_mask, n):
                new_edges = current_edges + 1
                # Accept (improvement)
                current_edges = new_edges
            else:
                # Reject (revert)
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_mask = adj_mask[:]

    # Convert best_adj_mask to edge list
    edges = []
    for u in range(n):
        mask = best_adj_mask[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
            
    return edges
