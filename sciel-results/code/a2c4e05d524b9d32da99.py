import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    # 1. Precompute bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == L_ab and _lcm(b, c) == L_ab:
                    bad_triples.append((a, b, c))
                    
    # 2. Precompute conflicts for each element
    # triple_conflicts[x] = list of (u, v) such that {x, u, v} is a bad triple
    triple_conflicts = [[] for _ in range(n + 1)]
    for a, b, c in bad_triples:
        triple_conflicts[a].append((b, c))
        triple_conflicts[b].append((a, c))
        triple_conflicts[c].append((a, b))
        
    # Helper for popcount
    def popcount(mask):
        return bin(mask).count('1')
        
    # 3. Initial Heuristic: Greedy + Local Search
    # This provides a strong lower bound for the B&B
    
    def can_add(x, mask, active_edges):
        # Check against included
        for u, v in triple_conflicts[x]:
            if (mask >> (u - 1)) & 1 and (mask >> (v - 1)) & 1:
                return False
        # Check against active_edges (binary constraints)
        # If (x, y) is an active edge, and y is in mask, conflict
        # We don't have a direct list of edges incident to x in active_edges efficiently.
        # But in the heuristic, we might not use active_edges, just the hypergraph.
        # For the heuristic, let's just use the hypergraph constraint.
        return True

    def greedy_heuristic():
        mask = 0
        # Order by degree in bad triples (higher degree first)
        degrees = [len(triple_conflicts[i]) for i in range(n + 1)]
        order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
        
        for x in order:
            if can_add(x, mask, []):
                mask |= (1 << (x - 1))
        return mask

    best_mask = greedy_heuristic()
    best_size = popcount(best_mask)
    
    # Simple local search to improve heuristic
    def local_search(mask):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing one element and re-greedy
            s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
            if not s_list:
                break
            for _ in range(10):
                x = random.choice(s_list)
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                # Re-greedy
                for y in range(1, n + 1):
                    if not (new_mask >> (y - 1)) & 1:
                        if can_add(y, new_mask, []):
                            new_mask |= (1 << (y - 1))
                if popcount(new_mask) > popcount(mask):
                    mask = new_mask
                    improved = True
                    break
        return mask

    best_mask = local_search(best_mask)
    best_size = popcount(best_mask)
    
    # 4. Branch and Bound
    # State: available_mask, included_mask, active_edges (list of (u,v)), current_size
    # To speed up, we represent active_edges as a set of encoded ints (u*n + v) or similar.
    # But checking conflicts with active edges is O(E).
    # We can optimize by only adding edges when necessary.
    
    def check_conflict_with_included(x, included_mask):
        for u, v in triple_conflicts[x]:
            if (included_mask >> (u - 1)) & 1 and (included_mask >> (v - 1)) & 1:
                return True
        return False

    def check_conflict_with_active_edges(x, active_edges_set, available_mask):
        # active_edges_set contains encoded edges (u, v) where u < v
        # We need to check if there is an edge (x, y) in active_edges_set
        # Such that y is in available_mask (or included? No, if y was included, x would have been excluded earlier)
        # Actually, if (x, y) is an active edge, it means x and y cannot both be selected.
        # If we are trying to include x, and y is already in included_mask, conflict.
        # If y is in available_mask, it's a future constraint, not a current conflict.
        # Wait, active edges are constraints between *available* vertices.
        # So if we include x, we must ensure x is not in an active edge with an *included* vertex?
        # No, active edges are only between vertices that are still in the game.
        # If x is in available, and (x, y) is an active edge, y must also be in available.
        # So checking against active_edges for a *new* inclusion x:
        # We just need to know if x is part of any active edge?
        # No, the active edge means "if you pick x, you can't pick y".
        # It doesn't forbid picking x.
        # So `check_conflict_with_active_edges` for inclusion is FALSE.
        # The conflict arises when we *add* the edges.
        
        # However, we must ensure that we don't violate the edge if the other endpoint is *already included*.
        # But if the other endpoint was included, the edge would have been resolved when that endpoint was included.
        # So, if (x, y) is an active edge, y is in available.
        # Therefore, including x does not immediately conflict with active edges.
        # It just adds new constraints.
        return False

    # Recursive DFS
    # To avoid deep recursion limits, we can use a stack, but Python recursion is fine for N=128 if we prune well.
    # We will use a global variable for best_size to allow pruning.
    
    def dfs(available_mask, included_mask, active_edges_list, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        # Prune
        if current_size + popcount(available_mask) <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = included_mask
            return
            
        # Select pivot
        # Heuristic: Pick x with highest degree in active_edges + triple_conflicts
        # For simplicity, pick the first available x with high triple_conflict degree
        # Let's precompute a static order of preference
        # Or just pick the lowest index available?
        # Better: Pick x that maximizes the number of constraints generated.
        
        # Find available vertices
        avail_list = [i for i in range(1, n + 1) if (available_mask >> (i - 1)) & 1]
        if not avail_list:
            return
            
        # Sort by "impact": number of pairs in triple_conflicts[x] where both u,v are in available
        def impact(x):
            count = 0
            for u, v in triple_conflicts[x]:
                if (available_mask >> (u - 1)) & 1 and (available_mask >> (v - 1)) & 1:
                    count += 1
            return count
            
        pivot = max(avail_list, key=impact)
        
        # Branch 1: Exclude pivot
        dfs(available_mask & ~(1 << (pivot - 1)), included_mask, active_edges_list, current_size)
        
        # Branch 2: Include pivot
        if not check_conflict_with_included(pivot, included_mask):
            new_included = included_mask | (1 << (pivot - 1))
            new_available = available_mask & ~(1 << (pivot - 1))
            
            # Propagate constraints
            new_active_edges = active_edges_list[:] # Copy list
            valid = True
            
            for u, v in triple_conflicts[pivot]:
                u_in_inc = (included_mask >> (u - 1)) & 1
                v_in_inc = (included_mask >> (v - 1)) & 1
                u_in_avail = (new_available >> (u - 1)) & 1
                v_in_avail = (new_available >> (v - 1)) & 1
                
                if u_in_inc and v_in_inc:
                    valid = False
                    break
                elif u_in_inc and v_in_avail:
                    # v must be excluded
                    new_available &= ~(1 << (v - 1))
                    # Remove edges involving v from new_active_edges?
                    # Yes, because v is no longer available.
                    new_active_edges = [(a, b) for a, b in new_active_edges if a != v and b != v]
                elif v_in_inc and u_in_avail:
                    # u must be excluded
                    new_available &= ~(1 << (u - 1))
                    new_active_edges = [(a, b) for a, b in new_active_edges if a != u and b != u]
                elif u_in_avail and v_in_avail:
                    # Add edge (u, v)
                    # Ensure u < v for consistency
                    if u > v:
                        u, v = v, u
                    new_active_edges.append((u, v))
                # If both in included, handled above.
                # If one in included, one not available (excluded previously), nothing to do.
                # If both not available, nothing to do.
                
            if valid:
                # Prune active edges that are no longer relevant (endpoints not in new_available)
                # We already did this partially in the loop, but let's be safe
                # Also, if an edge (u,v) exists, and u or v is not in new_available, it's redundant.
                # The list comprehension above handles removal when we force exclusion.
                # But what if an edge was added in a previous step, and now one endpoint is excluded?
                # The list comprehension `if a != v and b != v` only removes edges involving the *specific* u or v being excluded.
                # This is correct.
                
                dfs(new_available, new_included, new_active_edges, current_size + 1)

    # Start DFS
    # Initial available is all 1..128
    initial_available = (1 << n) - 1
    # Note: bit 0 is element 1, bit 127 is element 128.
    # (1 << 128) - 1 has bits 0..127 set.
    
    dfs(initial_available, 0, [], 0)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
