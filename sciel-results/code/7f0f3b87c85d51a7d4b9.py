"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Precompute neighbors for all vertices
    # Since N=9, there are 2^9 = 512 vertices. We can use a list of lists.
    all_vertices = range(1 << N)
    neighbors = [[] for _ in all_vertices]
    for v in all_vertices:
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # For bitset operations, we can use Python integers as bitsets.
    # Bit i is 1 if vertex i is in the set.
    
    def greedy_extend_bitset(start_v, bit_order_hint=None):
        """
        Extend path from start_v using bitset-based forward checking.
        Returns the path as a list of vertices.
        """
        path = [start_v]
        used_mask = 1 << start_v
        
        # forbidden_mask: vertices that cannot be added next because they are at dist 1 
        # from some vertex in path (excluding the last one, which is allowed to be dist 1).
        # Actually, the condition is: new vertex v must not be at dist 1 from any path[i] 
        # for i < len(path)-1.
        # Let's maintain a mask 'blocked' which contains all vertices that are at dist 1 
        # from any vertex in path, EXCEPT we need to be careful about the last vertex.
        
        # A more efficient way:
        # Let S be the set of vertices in path.
        # A candidate v is valid if:
        # 1. v not in S
        # 2. dist(v, u) != 1 for all u in S \ {last}
        
        # We can maintain a mask 'danger' which is the union of neighbors of all vertices in S, 
        # but we need to subtract the neighbors of the last vertex? No, that's not quite right 
        # because the last vertex's neighbors are allowed (one of them is the next step).
        
        # Let's maintain:
        # used_mask: bits set for vertices in path.
        # blocked_mask: bits set for vertices that are at Hamming distance 1 from ANY vertex in path,
        #               EXCEPT we allow the neighbors of the current end.
        
        # Actually, simpler:
        # valid_candidates = All vertices 
        #                   - used_mask 
        #                   - (union of neighbors of all path vertices except the last)
        
        # Let's maintain 'forbidden' mask.
        # forbidden = union of neighbors of path[0], ..., path[-2]
        # Then valid next steps are neighbors of path[-1] that are not in used_mask and not in forbidden.
        
        forbidden = 0
        # Initialize forbidden with neighbors of path[0] if len(path) > 1? 
        # No, initially path=[v0]. Forbidden should be neighbors of v0?
        # If we pick v1, v1 must not be dist 1 from v0? No, v1 MUST be dist 1 from v0.
        # The condition is: non-consecutive vertices differ in at least two bits.
        # So v1 is consecutive to v0, so dist 1 is OK.
        # v2 must be dist 1 from v1, and dist >= 2 from v0.
        
        # So, when we are at path[-1], the next vertex v must satisfy:
        # 1. dist(v, path[-1]) == 1
        # 2. dist(v, path[i]) >= 2 for all i < len(path)-1
        
        # This means v must NOT be in the set of neighbors of {path[0], ..., path[-2]}.
        
        # Let's maintain 'blocked' = union of neighbors of path[0], ..., path[-2].
        # When we append a new vertex v to path, the new blocked set becomes:
        # blocked_new = blocked_old | neighbors(v)  (since v becomes path[-2] for the next step? 
        # No, v becomes path[-1]. The previous path[-1] becomes path[-2].
        # So we add neighbors of the OLD path[-1] to the blocked set?
        # Wait.
        # Step 0: path=[v0]. blocked = empty set.
        #   Candidates: neighbors(v0) \ used.
        #   Pick v1.
        # Step 1: path=[v0, v1].
        #   We need to ensure future v2 is not neighbor of v0.
        #   So blocked should now include neighbors(v0).
        #   Candidates: neighbors(v1) \ used \ blocked.
        #   Pick v2.
        # Step 2: path=[v0, v1, v2].
        #   We need to ensure future v3 is not neighbor of v0 or v1.
        #   So blocked should now include neighbors(v0) U neighbors(v1).
        #   Candidates: neighbors(v2) \ used \ blocked.
        
        # So the update rule is:
        # When we move from path[-1] to new_v:
        # 1. Check new_v is in neighbors(path[-1]) \ used \ blocked.
        # 2. Update blocked = blocked | neighbors(path[-1])
        # 3. Update used = used | (1 << new_v)
        # 4. Append new_v to path.
        
        # Let's implement this.
        
        # Precompute neighbor masks for fast OR
        neighbor_masks = [0] * (1 << N)
        for v in all_vertices:
            mask = 0
            for n in neighbors[v]:
                mask |= (1 << n)
            neighbor_masks[v] = mask

        while True:
            if time.time() > deadline:
                break
                
            cur = path[-1]
            candidates_mask = neighbor_masks[cur] & ~used_mask & ~blocked
            
            if candidates_mask == 0:
                break
            
            # Heuristic: choose candidate with fewest options (minimum number of future neighbors)
            # Or random if tie.
            # To find the best candidate, we iterate through set bits.
            # Since N=9, there are at most 9 candidates.
            
            best_cand = -1
            min_options = 100
            best_mask = 0
            # For "most options" heuristic, we usually pick the one that leaves the most freedom?
            # Or "fewest options" (fail-first)?
            # For snakes, "most options" (maximizing the number of valid next steps for the next step) 
            # is often better to avoid dead ends early.
            # Let's try "most options": maximize popcount(neighbor_masks[cand] & ~used & ~blocked)
            # Note: used and blocked will change, but this is a good heuristic.
            
            # We need to check candidates.
            # Extract bits from candidates_mask
            m = candidates_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                # Count valid options for v
                # Options for v are neighbors of v that are not used and not blocked.
                # Note: if we pick v, the new blocked will be blocked | neighbor_masks[cur].
                # So the "effective blocked" for v's neighbors is blocked | neighbor_masks[cur].
                eff_blocked = blocked | neighbor_masks[cur]
                options_mask = neighbor_masks[v] & ~used_mask & ~eff_blocked
                options_count = options_mask.bit_count()
                
                if options_count > min_options:
                    min_options = options_count
                    best_cand = v
                    best_mask = options_mask
                elif options_count == min_options and options_count > 0:
                    # Tie-breaking: random
                    if random.random() < 0.5:
                        best_cand = v
                        best_mask = options_mask

            if best_cand == -1:
                break
            
            # Commit to best_cand
            # Update blocked: add neighbors of current vertex (which becomes non-consecutive)
            blocked |= neighbor_masks[cur]
            used_mask |= (1 << best_cand)
            path.append(best_cand)
            
        return path

    best_path = []
    
    # Try multiple starting vertices and randomizations
    # Since the graph is symmetric, starting at 0 is as good as any, but the heuristic 
    # (tie-breaking) introduces randomness.
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        # Random start
        start_v = random.randint(0, (1 << N) - 1)
        
        # We can also try different heuristic weights or tie-breaking strategies
        # For now, just use the standard "most options" with random tie-breaking.
        
        path = greedy_extend_bitset(start_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
            # print(f"New best: {len(best_path)} at step {i}") # Debug
    
    # If we didn't find anything (shouldn't happen), return a trivial path
    if not best_path:
        best_path = [0]
        
    return best_path
# EVOLVE-BLOCK-END
