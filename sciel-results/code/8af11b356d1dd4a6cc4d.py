import math
import itertools
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    deadline = time.time() + 140.0
    
    def build_paley_adj(p):
        """Build Paley tournament adjacency matrix on p vertices (p prime ≡ 3 mod 4)."""
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x*x) % p] = True
        # adj[i][j] = True if i -> j
        # i -> j iff (j-i) mod p is a quadratic residue
        idx = np.arange(p)
        diff = (idx[None, :] - idx[:, None]) % p
        adj = squares[diff]
        # Fix diagonal
        np.fill_diagonal(adj, False)
        return adj
    
    def check_property_fast(adj, n):
        """Check if every 4-set has a common dominator using NumPy."""
        # For each vertex v, compute the set of 4-sets it dominates
        # v dominates 4-set {a,b,c,d} iff adj[v][a] & adj[v][b] & adj[v][c] & adj[v][d]
        # We need every 4-set to be dominated by at least one vertex.
        
        # For each vertex v, the vertices it beats: out_v = adj[v, :]
        # A 4-set {a,b,c,d} is dominated by v iff all four are in out_v.
        # Equivalent: for the 4-set, the number of vertices in the set that v beats must be 4.
        
        # For each 4-set, check if any vertex beats all 4.
        # Vectorized approach: for each vertex v, form a bitmask of whom it beats.
        # Then for each 4-set, check if any v's mask covers all 4 bits.
        
        # Use bitmasks: for n up to 67, we can use Python ints as bitmasks.
        masks = np.zeros(n, dtype=np.int64) if n <= 63 else None
        if n <= 63:
            for v in range(n):
                m = 0
                for j in range(n):
                    if adj[v, j]:
                        m |= (1 << j)
                masks[v] = m
        
        if n <= 63:
            # For each 4-combination, check if any mask covers all 4 bits
            for quad in itertools.combinations(range(n), 4):
                a, b, c, d = quad
                target = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                # Check if any mask has all these bits set
                found = False
                for v in range(n):
                    if (masks[v] & target) == target:
                        found = True
                        break
                if not found:
                    return False
            return True
        else:
            # Fallback for larger n: use numpy
            for quad in itertools.combinations(range(n), 4):
                a, b, c, d = quad
                # For each v, check if v beats a, b, c, d
                ok = (adj[:, a] & adj[:, b] & adj[:, c] & adj[:, d]).any()
                if not ok:
                    return False
            return True
    
    def prune_tournament(adj, n, deadline):
        """Remove vertices that are not needed as dominators of any 4-set."""
        # A vertex v is "essential" if there exists a 4-set that v is the ONLY dominator of.
        # We can iteratively remove non-essential vertices.
        
        n_current = n
        adj_current = adj.copy()
        active = np.ones(n, dtype=bool)
        
        while time.time() < deadline:
            # For each active vertex, determine if it's the unique dominator of any 4-set
            # among the active vertices.
            # This is expensive, so let's use a heuristic:
            # A vertex v can be removed if for every 4-set containing v or dominated only by v,
            # there's another active vertex that also dominates it.
            
            # Simpler approach: try removing each vertex one at a time and check if property holds.
            # But that's O(n * cost_of_check). Let's be smarter.
            
            # For each vertex v, compute the set of 4-sets for which v is the unique dominator.
            # If this set is empty, v can be removed.
            
            # To check if v is the unique dominator of some 4-set {a,b,c,d}:
            # v beats a,b,c,d AND for all other active vertices u, u does NOT beat all of a,b,c,d.
            
            # This is still expensive. Let's use a greedy approach:
            # Try to remove vertices in order of their "redundancy score".
            
            # Compute out-degree for each active vertex
            active_indices = np.where(active)[0]
            n_active = len(active_indices)
            
            if n_active <= 5:
                break
            
            # For each active vertex, count how many 4-sets it uniquely dominates
            # Approximate: a vertex with low out-degree is more likely to be essential
            out_degrees = np.array([np.sum(adj_current[i, :]) for i in active_indices])
            
            # Try removing the vertex with highest out-degree (most likely redundant)
            # Actually, vertices with HIGH out-degree are more likely to be dominators,
            # so they might be essential. Let's try removing vertices with LOW out-degree first.
            # Wait, no: a vertex that beats many others is a strong dominator.
            # A vertex with low out-degree might not dominate any 4-set, so it's safe to remove.
            
            # Let's just try removing each vertex and checking.
            # To speed up, we'll check in order of increasing out-degree.
            order = np.argsort(out_degrees)
            
            removed_this_round = False
            for idx in order:
                if time.time() > deadline - 10:
                    break
                
                v = active_indices[idx]
                
                # Check if removing v preserves the property
                # Build new active set without v
                new_active = active.copy()
                new_active[v] = False
                new_active_indices = np.where(new_active)[0]
                n_new = len(new_active_indices)
                
                if n_new < 5:
                    continue
                
                # Check property on the induced sub-tournament
                # For every 4-set in new_active_indices, is there a vertex in new_active_indices
                # that beats all 4?
                valid = True
                # Sample check first for speed
                if n_new > 30:
                    # Quick check: sample some 4-sets
                    rng = np.random.default_rng(42)
                    for _ in range(200):
                        quad = rng.choice(n_new, size=4, replace=False)
                        quad = sorted(quad)
                        a, b, c, d = [new_active_indices[quad[i]] for i in range(4)]
                        # Check if any active vertex beats all 4
                        ok = False
                        for u in new_active_indices:
                            if adj_current[u, a] and adj_current[u, b] and adj_current[u, c] and adj_current[u, d]:
                                ok = True
                                break
                        if not ok:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Full check
                valid = True
                for quad in itertools.combinations(range(n_new), 4):
                    a, b, c, d = [new_active_indices[quad[i]] for i in range(4)]
                    ok = False
                    for u in new_active_indices:
                        if adj_current[u, a] and adj_current[u, b] and adj_current[u, c] and adj_current[u, d]:
                            ok = True
                            break
                    if not ok:
                        valid = False
                        break
                
                if valid:
                    active[v] = False
                    removed_this_round = True
                    # Don't break; try to remove more
                    # But update active_indices for next iteration would be needed
                    # For simplicity, break and restart
                    break
            
            if not removed_this_round:
                break
        
        # Build final orientation
        final_indices = np.where(active)[0]
        n_final = len(final_indices)
        
        orientation = []
        for i in range(n_final):
            for j in range(i+1, n_final):
                vi = final_indices[i]
                vj = final_indices[j]
                if adj[vi, vj]:
                    orientation.append(0)  # vi -> vj
                else:
                    orientation.append(1)  # vj -> vi
        
        return n_final, orientation
    
    # Build Paley tournament on 67 vertices
    p = 67
    adj = build_paley_adj(p)
    n = p
    
    # Verify the full tournament has the property
    if not check_property_fast(adj, n):
        # Shouldn't happen for Paley(67), but handle gracefully
        # Return the full tournament
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                orientation.append(0 if adj[i, j] else 1)
        return {'n': n, 'orientation': orientation}
    
    # Try to prune
    n_pruned, orientation_pruned = prune_tournament(adj, n, deadline)
    
    # Verify the pruned tournament
    if n_pruned < n:
        # Rebuild adjacency for pruned version to verify
        # Actually, let's just verify the orientation directly
        adj_check = np.zeros((n_pruned, n_pruned), dtype=bool)
        idx = 0
        for i in range(n_pruned):
            for j in range(i+1, n_pruned):
                if orientation_pruned[idx] == 0:
                    adj_check[i, j] = True
                else:
                    adj_check[j, i] = True
                idx += 1
        
        if check_property_fast(adj_check, n_pruned):
            return {'n': n_pruned, 'orientation': orientation_pruned}
    
    # Fallback: return the full 67-vertex tournament
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            orientation.append(0 if adj[i, j] else 1)
    return {'n': n, 'orientation': orientation}

# EVOLVE-BLOCK-END
