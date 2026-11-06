"""Bootstrap spawn: DIRECT construction using Paley tournament on p=67 vertices."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Uses the Paley tournament on p=67 vertices, then attempts to prune vertices
    while maintaining the property.
    """
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit
    
    # The smallest known valid n is 67 (Paley tournament)
    p = 67
    
    # Precompute quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    # Build adjacency matrix for Paley 67
    # adj[i][j] = 1 if i -> j, 0 otherwise
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            diff = (j - i) % p
            if diff in squares:
                adj[i][j] = 1
            else:
                adj[i][j] = 0
                
    # Function to check if a tournament defined by adjacency matrix 'a' (size n)
    # satisfies the property: every 4-set has a common dominator.
    def is_valid(a):
        n = a.shape[0]
        # For each vertex u, determine the set of vertices it dominates (out-neighbors)
        # out[i] is a bitset or list of vertices dominated by i
        # To check if a 4-set {a,b,c,d} has a dominator u:
        # u must dominate a, b, c, d.
        # So u must be in the intersection of IN(a), IN(b), IN(c), IN(d).
        # Where IN(v) is the set of vertices that point to v.
        
        # Let's precompute in-neighbors for each vertex as a set or bitset.
        # Using Python sets for clarity, might be slow for large n, but n <= 67 is small.
        # Actually, using numpy integer operations or just loops is fine for n=67.
        
        in_neigh = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and a[i][j] == 1:
                    in_neigh[j].add(i)
                    
        # Iterate over all 4-sets
        # Optimization: If a vertex has in-degree < 4, it cannot be a dominator for any 4-set?
        # No, the dominator is OUTSIDE the 4-set.
        # The 4-set is {v1, v2, v3, v4}.
        # We need a vertex u not in {v1..v4} such that u -> v1, u -> v2, u -> v3, u -> v4.
        # This means u is in in_neigh[v1] INTERSECT in_neigh[v2] INTERSECT in_neigh[v3] INTERSECT in_neigh[v4].
        # And u not in {v1..v4}.
        
        # To speed up, we can check if the intersection is non-empty.
        # For n=67, number of 4-sets is ~715k.
        # Intersection of 4 sets of size ~33.
        # This is fast in Python if we use sets.
        
        for v1 in range(n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    for v4 in range(v3 + 1, n):
                        # Check intersection
                        # Start with smallest set for efficiency?
                        # All in-neigh sets are roughly same size.
                        inter = in_neigh[v1]
                        if not inter: continue
                        inter = inter.intersection(in_neigh[v2])
                        if not inter: continue
                        inter = inter.intersection(in_neigh[v3])
                        if not inter: continue
                        inter = inter.intersection(in_neigh[v4])
                        
                        # Check if there is a vertex in inter that is not in {v1, v2, v3, v4}
                        # Since v1..v4 are in the set, and u must dominate them, u cannot be v1..v4
                        # because there are no self-loops.
                        # So if inter is not empty, is it guaranteed to contain a vertex outside the 4-set?
                        # Yes, because if u is in in_neigh[v1], u != v1.
                        # So u is distinct from v1, v2, v3, v4.
                        # Thus, if inter is non-empty, we have a valid dominator.
                        
                        if len(inter) > 0:
                            continue
                        else:
                            return False
        return True

    # Try to prune vertices from the Paley tournament
    # We start with all 67 vertices.
    # We try to remove vertices one by one.
    # Since the tournament is highly symmetric, the order might not matter much,
    # but we can try removing 0, then 1, etc.
    
    current_adj = adj
    current_n = p
    removed = set()
    
    # We will try to remove vertices 0 to p-1 in order.
    # If removing a vertex breaks validity, we stop.
    # Note: Removing a vertex creates a subtournament.
    # We can just slice the matrix.
    
    valid_n = p
    valid_adj = adj
    
    # Check if full Paley 67 is valid (it should be)
    if is_valid(adj):
        valid_n = p
        valid_adj = adj
    
    # Try to remove vertices
    # We maintain a list of active vertices
    active = list(range(p))
    
    for v in range(p):
        if time.time() - start_time > deadline - 10:
            break
            
        # Try removing v from active
        # If v is already removed, skip? 
        # We are iterating v from 0 to p-1.
        # If we remove 0, active becomes [1..66].
        # Then we try to remove 1 from active [1..66] -> [2..66].
        
        if v not in active:
            continue
            
        new_active = [u for u in active if u != v]
        
        # Build new adjacency matrix
        n_new = len(new_active)
        if n_new < 5:
            break
            
        # Map old indices to new indices
        # We need to extract submatrix
        # indices = np.array(new_active)
        # sub_adj = current_adj[indices][:, indices]
        
        # Using numpy fancy indexing
        idx = np.array(new_active)
        sub_adj = current_adj[idx][:, idx]
        
        # Check validity
        # is_valid is O(n^4) with set intersections.
        # For n=66, it takes ~0.1s?
        # Let's assume it's fast enough.
        
        if is_valid(sub_adj):
            valid_n = n_new
            valid_adj = sub_adj
            active = new_active
            # current_adj = sub_adj # Not strictly needed if we rebuild from original, 
            # but sub_adj is the new base.
            current_adj = sub_adj
            # Note: The indices in sub_adj are 0..n_new-1 corresponding to new_active.
            # The next iteration will remove a vertex from the ORIGINAL set?
            # No, we are iterating `v` in range(p).
            # If we removed 0, active is [1..66].
            # Next v=1. 1 is in active.
            # We remove 1 from active.
            # This works.
            
    # Construct orientation bits from valid_adj
    orientation = []
    n = valid_n
    for i in range(n):
        for j in range(i + 1, n):
            if valid_adj[i][j] == 1:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
