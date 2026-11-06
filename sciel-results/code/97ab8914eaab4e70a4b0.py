"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Build conflict hyperedges: triples (a,b,c) where gcd(a,b)=gcd(a,c)=gcd(b,c)
    # For each vertex v, compute its "conflict set" - vertices that cannot coexist with v
    # in a valid set (i.e., there exists some u such that {v,u,w} forms a bad triple for some w)
    
    # Actually, the constraint is: no three distinct elements a,b,c with gcd(a,b)=gcd(a,c)=gcd(b,c)
    # This is a 3-uniform hypergraph where hyperedges are the bad triples.
    # We want maximum independent set in this hypergraph.
    
    # Key insight: if we select a set S, then for any d, the elements in S with pairwise gcd d
    # must form a set where no two have gcd d with a third element also having gcd d with both.
    # More precisely: for fixed d, let S_d = {x in S : there exists y in S with gcd(x,y)=d}
    # Actually, the constraint is: for any three distinct a,b,c in S, NOT (gcd(a,b)=gcd(a,c)=gcd(b,c))
    
    # Let's think differently. For a candidate element x to be added to set S:
    # For each d, consider T_x_d = {y in S : gcd(x,y) = d}
    # If |T_x_d| >= 2, say y1, y2 in T_x_d, then we need gcd(y1,y2) != d
    # So x can be added iff for all d, no pair in T_x_d has gcd equal to d.
    
    # Precompute for each x and d, the mask of y where gcd(x,y)=d
    gcd_mask = [[0]*(N+1) for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            d = gcd_table[x][y]
            gcd_mask[x][d] |= (1 << y)
    
    # Precompute pairs_by_gcd[d] as bitmask pairs for fast checking
    # For each d, store list of (a,b) where a<b and gcd(a,b)=d
    pairs_by_gcd = defaultdict(list)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = gcd_table[a][b]
            pairs_by_gcd[d].append((a, b))
    
    def can_add(x, selected_mask):
        """Check if x can be added to set represented by selected_mask."""
        for d in range(1, N+1):
            mask_x_d = gcd_mask[x][d]
            candidates = selected_mask & mask_x_d
            if candidates == 0:
                continue
            # Check if any pair (a,b) in pairs_by_gcd[d] has both a and b in candidates
            for a, b in pairs_by_gcd[d]:
                if (candidates & (1 << a)) and (candidates & (1 << b)):
                    return False
        return True
    
    def mask_to_list(mask):
        """Convert bitmask to list of elements."""
        result = []
        for i in range(1, N+1):
            if mask & (1 << i):
                result.append(i)
        return result
    
    # Phase 1: Get a good initial solution via greedy + local search
    def greedy(order):
        selected_mask = 0
        for x in order:
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask
    
    # Try several orderings
    best_mask = 0
    for order in [list(range(1, N+1)), list(range(N, 0, -1))]:
        m = greedy(order)
        if bin(m).count('1') > bin(best_mask).count('1'):
            best_mask = m
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        m = greedy(order)
        if bin(m).count('1') > bin(best_mask).count('1'):
            best_mask = m
    
    # Local search: try adding, then try swaps
    def local_search(start_mask, time_limit):
        best = start_mask
        best_size = bin(best).count('1')
        
        while time.time() < time_limit:
            improved = False
            # Try adding
            for x in range(1, N+1):
                if best & (1 << x):
                    continue
                if can_add(x, best):
                    best |= (1 << x)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try 1-out 2-in swaps
            found_swap = False
            for i in range(1, N+1):
                if not (best & (1 << i)):
                    continue
                temp = best & ~(1 << i)
                added = 0
                for x in range(1, N+1):
                    if temp & (1 << x) or x == i:
                        continue
                    if can_add(x, temp):
                        temp |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    best = temp
                    best_size += added - 1
                    found_swap = True
                    break
            if not found_swap:
                break
        return best
    
    if time.time() < deadline - 25:
        best_mask = local_search(best_mask, deadline - 25)
    
    # Phase 2: Branch and bound to improve
    # We use a recursive approach: at each step, either include or exclude a vertex.
    # Upper bound: current_size + number of remaining candidates that could potentially be added
    # (optimistically, all of them)
    
    # For better pruning, we can compute for each unselected vertex how many selected vertices
    # it conflicts with in a "would form bad triple" sense.
    
    # Simpler B&B: try to add vertices in order of "least restrictive" first
    # Use the fact that if we have a partial solution, we can compute an upper bound
    
    def branch_and_bound(current_mask, candidates_mask, best_size, time_limit):
        """
        current_mask: vertices already in the solution
        candidates_mask: vertices that could still potentially be added
        best_size: current best solution size
        """
        current_size = bin(current_mask).count('1')
        
        # Upper bound: current_size + popcount(candidates_mask)
        upper_bound = current_size + bin(candidates_mask).count('1')
        if upper_bound <= best_size:
            return current_mask, best_size
        
        if time.time() > time_limit:
            return current_mask, best_size
        
        if candidates_mask == 0:
            return current_mask, best_size
        
        # Pick a vertex to branch on
        # Choose vertex with fewest "conflicts" among candidates (most flexible)
        # Or just pick the first available
        # For efficiency, pick vertex that appears in most potential conflicts
        
        # Simple heuristic: pick vertex with smallest index among candidates
        # Better: pick vertex that, when excluded, removes most candidates
        # For now, use a simple ordering: try vertices in order of their "degree" in conflict hypergraph
        
        # Compute for each candidate v, how many other candidates u would conflict with v
        # (i.e., there exists w in current_mask such that {v,u,w} is a bad triple)
        # This is expensive, so use a simpler heuristic
        
        # Just iterate through candidates in order
        v_list = []
        temp = candidates_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            v_list.append(v)
            temp &= temp - 1
        
        # Sort by some heuristic: vertices with fewer gcd partners in current set first
        # (less restrictive to add)
        def restrictiveness(v):
            count = 0
            for d in range(1, N+1):
                count += bin(current_mask & gcd_mask[v][d]).count('1')
            return count
        
        v_list.sort(key=lambda v: restrictiveness(v))
        
        for v in v_list:
            if time.time() > time_limit:
                break
            if not (candidates_mask & (1 << v)):
                continue
            
            # Try including v
            if can_add(v, current_mask):
                new_current = current_mask | (1 << v)
                # Update candidates: remove v and any that conflict with v
                new_candidates = candidates_mask & ~(1 << v)
                # For each u in new_candidates, check if u can still be added given new_current
                # This is expensive; instead, just remove v and recurse
                # The can_add check will handle conflicts when we try to add u later
                result, best_size = branch_and_bound(new_current, new_candidates, best_size, time_limit)
                current_mask = result
                candidates_mask = candidates_mask & ~(1 << v)
                
                if bin(current_mask).count('1') > best_size:
                    best_size = bin(current_mask).count('1')
            else:
                # v cannot be added, just remove it from candidates
                candidates_mask &= ~(1 << v)
        
        return current_mask, best_size
    
    if time.time() < deadline - 5:
        # Start B&B from best solution, trying to add more
        remaining = ((1 << (N+1)) - 1) & ~((1 << 1) - 1) & ~best_mask
        # Clear bit 0
        remaining &= ~1
        best_mask, _ = branch_and_bound(best_mask, remaining, bin(best_mask).count('1'), deadline - 5)
    
    result = mask_to_list(best_mask)
    return result if result else [1]
# EVOLVE-BLOCK-END
