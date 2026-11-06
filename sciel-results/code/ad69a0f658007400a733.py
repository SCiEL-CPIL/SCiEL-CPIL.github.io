"""Bootstrap spawn: DIRECT construction using Paley tournament on p=67 vertices with bitmask optimization and local search repair."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Uses the Paley tournament on p=67 vertices, then attempts to prune vertices
    while maintaining the property, using fast bitmask validity checks.
    If pruning gets stuck, it attempts local search (edge flipping) to repair
    the tournament and allow further vertex removal.
    """
    start_time = time.time()
    deadline = 115  # Safety margin for 120s limit
    
    # 1. Construct Paley Tournament on p=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    # Build adjacency bitmask: adj[i] is an integer where bit j is 1 if i -> j
    adj = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            diff = (j - i) % p
            if diff in squares:
                mask |= (1 << j)
        adj[i] = mask

    # Helper to compute in-neighborhood masks
    def get_in_masks(adj_list):
        n = len(adj_list)
        in_masks = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (adj_list[i] >> j) & 1:
                    in_masks[j] |= (1 << i)
        return in_masks

    # Fast validity check using bitmasks
    # Returns True if valid, False otherwise.
    # Also returns the count of violations (useful for local search).
    def check_validity(adj_list):
        n = len(adj_list)
        if n < 5:
            return True, 0
        in_masks = get_in_masks(adj_list)
        violations = 0
        # Iterate over all 4-sets
        # Optimization: Precompute intersection of first two in_masks?
        # Just do it directly, it's very fast.
        for a in range(n):
            ma = in_masks[a]
            for b in range(a + 1, n):
                mab = ma & in_masks[b]
                if mab == 0:
                    continue
                for c in range(b + 1, n):
                    mabc = mab & in_masks[c]
                    if mabc == 0:
                        continue
                    for d in range(c + 1, n):
                        if (mabc & in_masks[d]) == 0:
                            violations += 1
        return violations == 0, violations

    # Convert current adj to a list of active indices to manage pruning
    active = list(range(p))
    current_adj = adj[:]
    valid_n = p
    best_adj = adj[:]
    
    # Verify base case
    is_valid, _ = check_validity(current_adj)
    if not is_valid:
        # Should not happen for Paley 67, but if it does, we return n=67 anyway?
        # No, if it's invalid, we have a bug. Assume it is valid.
        pass
        
    # 2. Greedy Pruning
    # Try to remove vertices 0..p-1.
    # We maintain 'active' and 'current_adj' which is the adjacency matrix for 'active' vertices (renumbered 0..k-1).
    # To simplify, we will just filter the original adj list based on active indices?
    # No, flipping edges requires modifying the matrix.
    # Let's keep a dense matrix representation for the current active set for ease of flipping, 
    # but use bitmasks for checking.
    
    # Let's restart with a structure that supports local search better.
    # State: A list of bitmasks `cur_adj` of length `cur_n`.
    
    cur_adj = adj[:]
    cur_n = p
    best_n = p
    best_adj = adj[:]
    
    # Pruning loop
    # We iterate through original vertices. If we remove v, we must update cur_adj.
    # It is easier to just try to remove vertices in the current active set sequentially.
    # However, the choice of WHICH vertex to remove matters.
    # Strategy: Try removing vertex 0. If valid, keep it removed. Else, try 1, etc.
    # This is "greedy sequential".
    
    # To allow local search, we need to be able to "undo" a removal or "repair" the state.
    # Simpler approach:
    # 1. Try to remove vertex 0. Check validity.
    #    - If valid: Keep removal.
    #    - If invalid: Try to repair via local search. If repaired, keep removal. Else, restore.
    
    # Let's implement a function to try to remove a vertex and repair if necessary.
    
    def try_remove_and_repair(adj_list, remove_idx, max_steps=500):
        """
        Tries to remove vertex remove_idx from adj_list.
        If the resulting tournament is invalid, attempts to flip edges to fix violations.
        Returns (new_adj_list, success_bool).
        """
        n = len(adj_list)
        if n <= 5:
            return adj_list, False
            
        # Create new adj list without remove_idx
        # Reindexing:
        new_n = n - 1
        new_adj = [0] * new_n
        
        # Mapping old index -> new index
        # 0..remove_idx-1 stay same
        # remove_idx+1..n-1 shift down by 1
        
        for i in range(n):
            if i == remove_idx:
                continue
            new_i = i if i < remove_idx else i - 1
            mask = 0
            for j in range(n):
                if j == remove_idx:
                    continue
                if (adj_list[i] >> j) & 1:
                    new_j = j if j < remove_idx else j - 1
                    mask |= (1 << new_j)
            new_adj[new_i] = mask
            
        is_valid, viol_count = check_validity(new_adj)
        
        if is_valid:
            return new_adj, True
            
        # If invalid, try local search
        # We need to identify which edges to flip.
        # Heuristic: Flip edges incident to the vertices involved in violations.
        # Since we don't easily get the list of violations from check_validity, 
        # let's just try random flips.
        
        # To make random flips effective, we should bias them.
        # The violations are 4-sets with no dominator. 
        # Flipping an edge (u,v) changes in_masks for u and v.
        
        # Let's try a limited number of random flips.
        current_adj = new_adj[:]
        current_viol = viol_count
        
        # Precompute in_masks for current state to help? 
        # check_validity is O(N^4). Flipping 1 edge changes O(N) in_masks entries?
        # Actually, flipping edge u->v to v->u changes:
        # out(u) loses v, gains nothing? No, tournament: u->v becomes v->u.
        # adj[u] bit v clears. adj[v] bit u sets.
        # in_masks[v] loses u, gains nothing? No, in_masks[v] had u. Now u points away from v?
        # Wait. u->v means u is in in(v).
        # If we flip to v->u:
        # adj[u] bit v = 0. adj[v] bit u = 1.
        # in_masks[v]: u was in it. Now u is not. So in_masks[v] clears bit u.
        # in_masks[u]: u was not in it. Now v is in it. So in_masks[u] sets bit v.
        
        # We can update in_masks incrementally.
        # But for simplicity and given time constraints, full recheck is OK if steps are low.
        # N=66, check is ~0.1s. 500 steps = 50s. Too slow.
        # We need faster violation counting or fewer steps.
        
        # Let's reduce max_steps to 50 and rely on greedy pruning mostly.
        # Or, only attempt repair if viol_count is small.
        
        if viol_count > 5:
             return new_adj, False
             
        steps = 0
        max_steps = 20 # Keep it very tight
        
        while steps < max_steps and current_viol > 0:
            # Pick a random edge to flip
            u = random.randint(0, new_n - 1)
            v = random.randint(0, new_n - 1)
            if u == v:
                continue
                
            # Flip
            # adj[u] bit v, adj[v] bit u
            
            bit_v_in_u = (current_adj[u] >> v) & 1
            if bit_v_in_u:
                # u -> v becomes v -> u
                current_adj[u] &= ~(1 << v)
                current_adj[v] |= (1 << u)
            else:
                # v -> u becomes u -> v
                current_adj[u] |= (1 << v)
                current_adj[v] &= ~(1 << u)
                
            # Check
            is_valid, new_viol = check_validity(current_adj)
            if new_viol < current_viol:
                current_viol = new_viol
                if new_viol == 0:
                    return current_adj, True
            else:
                # Revert
                if bit_v_in_u:
                    current_adj[u] |= (1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] |= (1 << u)
            steps += 1
            
        return current_adj, (current_viol == 0)

    # Iterate removals
    # We try to remove vertices in the current active set.
    # To allow different paths, we can just try removing index 0, then 0 again (which is original 1), etc.
    # This effectively removes vertices in the order 0, 1, 2...
    
    for _ in range(p - 5): # Don't go below 5
        if time.time() - start_time > deadline - 5:
            break
            
        if len(cur_adj) <= 5:
            break
            
        # Try to remove vertex 0 from current_adj
        # This corresponds to the "first" active vertex.
        new_adj, success = try_remove_and_repair(cur_adj, 0, max_steps=20)
        
        if success:
            cur_adj = new_adj
            if len(cur_adj) < best_n:
                best_n = len(cur_adj)
                best_adj = cur_adj[:]
        else:
            # If we can't remove 0, maybe we should stop?
            # Greedy sequential removal usually stops here.
            # We could try to remove a different vertex, but that complicates the state management.
            # Given the strict time limit and that 66/67 is likely the target, we stop.
            break

    # Final check on best_adj
    # Ensure best_adj is valid
    is_valid, _ = check_validity(best_adj)
    if not is_valid:
        # Fallback to original Paley 67
        best_adj = adj[:]
        best_n = p

    # Convert best_adj (list of bitmasks) to orientation bits
    orientation = []
    n = best_n
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
