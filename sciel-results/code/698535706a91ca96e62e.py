import time
import random
import math

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for fast validity checks
# NEIGHBOR_MASK[v] is a bitmask where bit u is 1 if u is a neighbor of v
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(20260715)
    
    # State representation:
    # path: list of vertices
    # used_mask: integer bitmask of used vertices
    # neighbor_mask: integer bitmask of all neighbors of used vertices (forbidden for extension)
    # Actually, for a snake, the constraint is:
    # 1. Consecutive vertices differ by 1 bit.
    # 2. Non-consecutive vertices differ by >= 2 bits.
    # This is equivalent to: The set of vertices in the path must be an independent set in the 
    # "chord" graph, but simpler: 
    # A vertex v can be added to the end of path P if:
    # - v is not in P
    # - v is not adjacent to any vertex in P except the last one.
    # Let's maintain:
    # - path: list
    # - used_mask: bitmask of vertices in path
    # - forbidden_mask: bitmask of vertices that are adjacent to any vertex in path[:-1] 
    #   (i.e., cannot be added next because they would form a chord with a non-consecutive vertex)
    #   Note: The last vertex path[-1] is allowed to be adjacent to the new vertex.
    #   So forbidden_mask should exclude neighbors of path[-1]? 
    #   Actually, if we add v to path, v must not be in used_mask.
    #   And v must not be adjacent to any u in path[:-1].
    #   So v must not be in (used_mask | neighbor_mask_of_path_except_last).
    
    # Let's precompute a function to check if a vertex can be added
    # For speed, we will maintain:
    # - used_mask
    # - forbidden_mask: union of neighbors of all vertices in path except the last one.
    #   When we add v to path:
    #   - new_used = used_mask | (1 << v)
    #   - new_forbidden = (forbidden_mask | NEIGHBOR_MASK[path[-1]]) & ~(1 << v) 
    #     Wait, NEIGHBOR_MASK[path[-1]] contains path[-2] and v. 
    #     We need to forbid neighbors of the OLD last vertex (which becomes non-consecutive) 
    #     but allow the NEW last vertex's neighbors? No.
    #     Let's trace:
    #     Path: ... a, b. 
    #     Forbidden for next step c: c cannot be neighbor of a (or any before a).
    #     c CAN be neighbor of b.
    #     So forbidden_mask currently contains neighbors of a (and others before).
    #     When we add c:
    #     New path: ... a, b, c.
    #     Forbidden for next step d: d cannot be neighbor of a OR b.
    #     So new_forbidden = old_forbidden | NEIGHBOR_MASK[b].
    #     But we must ensure c is not in old_forbidden and c is not in used_mask.
    #     Also, c must be a neighbor of b.
    
    def can_add(v, last, used_mask, forbidden_mask):
        if (used_mask >> v) & 1: return False
        if (forbidden_mask >> v) & 1: return False
        # Check if v is neighbor of last
        if (NEIGHBOR_MASK[last] >> v) & 1:
            return True
        return False

    def add_vertex(path, used_mask, forbidden_mask, v):
        # v is the new vertex
        last = path[-1]
        new_used = used_mask | (1 << v)
        # The old last vertex 'last' becomes non-consecutive to future vertices.
        # So its neighbors become forbidden (unless they are the new last, but v is the new last).
        # Wait, 'last' is now at index -2. Its neighbors are forbidden for the NEXT vertex.
        # So we add NEIGHBOR_MASK[last] to forbidden_mask.
        new_forbidden = forbidden_mask | NEIGHBOR_MASK[last]
        # However, the new vertex v is the new last. Its neighbors are NOT yet forbidden.
        # But v itself is in used_mask.
        # Also, we must remove v from forbidden_mask if it was there? 
        # can_add ensures v is not in forbidden_mask.
        # So:
        return path + [v], new_used, new_forbidden

    def remove_last(path, used_mask, forbidden_mask):
        if len(path) <= 1:
            return [], 0, 0
        last = path[-1]
        new_path = path[:-1]
        new_used = used_mask & ~(1 << last)
        # We need to subtract NEIGHBOR_MASK[new_path[-1]] from forbidden_mask?
        # This is hard with bitmasks because of overlap.
        # For 2-opt reversal, we don't remove one by one, we reverse a segment.
        # Let's implement 2-opt differently.
        
        # 2-opt Move:
        # Pick i < j. Reverse path[i:j+1].
        # This changes the connections at i-1/i and j/j+1.
        # Internal connections are preserved (reversed).
        # Validity of internal chords is preserved because the set of vertices is the same.
        # We only need to check:
        # 1. Is path[i-1] adjacent to path[j]? (New connection)
        # 2. Is path[i] adjacent to path[j+1]? (New connection)
        # 3. Are there any new chords?
        #    Chords are pairs (u,v) in path with |idx(u)-idx(v)| > 1 and dist(u,v)=1.
        #    Reversing a segment preserves the relative order of indices within the segment.
        #    Pairs both inside the segment: relative distance is preserved? 
        #    If u was at i+k and v at i+l, k<l. After reversal, u is at j-k, v at j-l.
        #    Distance is | (j-k) - (j-l) | = |l-k|. Same. So internal chords are preserved.
        #    Pairs one inside, one outside:
        #    Let u be in segment, v be outside.
        #    The index difference changes.
        #    However, the "forbidden" constraint is global: no two non-consecutive vertices can be adjacent.
        #    So we just need to ensure that the new connections don't create chords with ANY other vertex.
        #    Actually, the condition is: for any pair of vertices in the path, if they are not consecutive, they must not be adjacent.
        #    Reversing a segment changes which pairs are consecutive.
        #    Old consecutive: (i-1, i) and (j, j+1).
        #    New consecutive: (i-1, j) and (i, j+1).
        #    Old non-consecutive pairs that become consecutive? No, consecutive pairs are removed from non-consecutive.
        #    New non-consecutive pairs that were consecutive? 
        #    The pairs (i-1, i) and (j, j+1) are no longer consecutive. They become non-consecutive (unless i=j or i-1=j etc).
        #    So we must check if (i-1, i) and (j, j+1) are adjacent. If they are, and they are now non-consecutive, it's a chord.
        #    Also, the new consecutive pairs (i-1, j) and (i, j+1) must be valid edges (dist 1).
        #    And they must not create chords with other vertices? 
        #    If (i-1, j) is an edge, it's consecutive, so it's allowed.
        #    But does (i-1, j) being an edge imply a chord with some other vertex k?
        #    No, the chord definition is about the pair itself.
        #    Wait, the constraint is: "non-consecutive vertices differ in at least two bits".
        #    So if we make (i-1, j) consecutive, it's fine.
        #    But we broke the consecutiveness of (i-1, i). Now i-1 and i are non-consecutive.
        #    If dist(i-1, i) == 1, then we have a chord! 
        #    So we must ensure dist(i-1, i) >= 2 OR i-1 and i are still consecutive? 
        #    In 2-opt, i-1 and i are NOT consecutive anymore (unless the segment is length 1, which is no-op).
        #    So we MUST have dist(i-1, i) >= 2.
        #    Similarly, dist(j, j+1) >= 2.
        #    And the new edges: dist(i-1, j) == 1 and dist(i, j+1) == 1.
        #    Also, we must ensure that the new edges don't create chords with the rest?
        #    No, the edge itself is valid. The only risk is if some OTHER pair becomes a chord.
        #    But as established, internal pairs preserve relative distance.
        #    Pairs across the boundary:
        #    Consider u in segment, v outside (v < i-1 or v > j+1).
        #    The index difference changes.
        #    Does the "chord" status change?
        #    Chord status depends on whether they are consecutive in the path.
        #    u and v are consecutive only if they are at the boundary.
        #    The only pairs that change consecutiveness status are the boundary pairs.
        #    Therefore, if the boundary conditions are met, the rest of the snake remains valid.
        
        pass

    # Initialize with a greedy snake
    def greedy_snake(priority, start=0):
        path = [start]
        used_mask = 1 << start
        forbidden_mask = 0
        cur = start
        while True:
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if can_add(v, cur, used_mask, forbidden_mask):
                    nxt = v
                    break
            if nxt is None:
                break
            path, used_mask, forbidden_mask = add_vertex(path, used_mask, forbidden_mask, nxt)
            cur = nxt
        return path

    # Generate initial pool
    pool = []
    for i in range(20):
        if time.time() > deadline - 5:
            break
        priority = list(range(N))
        rnd.shuffle(priority)
        start = rnd.randint(0, SIZE - 1)
        p = greedy_snake(priority, start)
        if len(p) > 10:
            pool.append(p)
            
    if not pool:
        pool = [greedy_snake(list(range(N)), 0)]
        
    best_path = max(pool, key=len)
    best_len = len(best_path) - 1
    
    # Simulated Annealing with 2-opt
    # State: path
    # Move: 2-opt (i, j)
    # Cost: -len(path) (we want to maximize length)
    # But 2-opt preserves length! 
    # 2-opt only reshapes. It doesn't extend.
    # We need to combine 2-opt with Extension.
    # Strategy:
    # 1. Do a 2-opt move.
    # 2. Try to extend the new end (and start) greedily.
    # 3. If extended length > old length, accept.
    # 4. If not, accept with probability exp((new_len - old_len) / T).
    
    def try_extend(path):
        # Extend from the end
        used_mask = 0
        for v in path:
            used_mask |= 1 << v
        forbidden_mask = 0
        # Rebuild forbidden mask for the whole path except last
        if len(path) > 1:
            for v in path[:-1]:
                forbidden_mask |= NEIGHBOR_MASK[v]
        
        cur = path[-1]
        new_path = list(path)
        new_used = used_mask
        new_forb = forbidden_mask
        
        # Try to extend
        extended = False
        while True:
            # Use a random priority for extension to diversify
            # But for speed, let's try all neighbors in random order
            neighbors = [cur ^ (1 << b) for b in range(N)]
            rnd.shuffle(neighbors)
            nxt = None
            for v in neighbors:
                if can_add(v, cur, new_used, new_forb):
                    nxt = v
                    break
            if nxt is None:
                break
            new_path, new_used, new_forb = add_vertex(new_path, new_used, new_forb, nxt)
            cur = nxt
            extended = True
            
        if not extended:
            return path
        
        # Also try extending from the start?
        # Reverse path, extend, reverse back?
        # Let's just extend from the end for now.
        
        return new_path

    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    current_path = best_path
    current_len = len(current_path) - 1
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Pick a 2-opt move
        L = len(current_path)
        if L < 5:
            break
            
        i = rnd.randint(0, L - 2)
        j = rnd.randint(i + 1, L - 1)
        
        # Check boundary conditions for 2-opt
        # Old consecutive: (i-1, i) and (j, j+1)
        # New consecutive: (i-1, j) and (i, j+1)
        
        # Case 1: i == 0
        # No left boundary.
        # Right boundary: (j, j+1) becomes non-consecutive. Must not be edge.
        # New edge: (0, j+1) must be edge.
        if i == 0:
            # Check if (j, j+1) is an edge. If so, and j != L-1, it becomes a chord.
            # If j == L-1, there is no j+1.
            if j < L - 1:
                if (current_path[j] ^ current_path[j+1]).bit_count() == 1:
                    continue # Invalid 2-opt, would create chord
            # Check new edge (0, j+1)
            if j < L - 1:
                if (current_path[0] ^ current_path[j+1]).bit_count() != 1:
                    continue
            else:
                # j is last. New edge is (0, L-1)? No.
                # Segment [0..L-1] reversed is [L-1..0].
                # New path starts with L-1.
                # Edge between old start (0) and new start (L-1)? No, they are not connected.
                # Wait, 2-opt on [0..L-1] just reverses the whole path.
                # Consecutive pairs are reversed. 
                # (0,1) -> (1,0). (L-2, L-1) -> (L-1, L-2).
                # No new boundary edges to check?
                # The "chord" check for (j, j+1) doesn't apply if j is last.
                # So reversing the whole path is always valid if the path is valid.
                # But it doesn't help extension.
                pass
            
        # Case 2: j == L - 1
        # No right boundary.
        # Left boundary: (i-1, i) becomes non-consecutive. Must not be edge.
        # New edge: (i-1, L-1) must be edge.
        if j == L - 1:
            if i > 0:
                if (current_path[i-1] ^ current_path[i]).bit_count() == 1:
                    continue
                if (current_path[i-1] ^ current_path[L-1]).bit_count() != 1:
                    continue
            else:
                # i=0, j=L-1. Reverse whole path.
                pass

        # Case 3: 0 < i < j < L - 1
        if 0 < i < j < L - 1:
            # Check (i-1, i) not edge
            if (current_path[i-1] ^ current_path[i]).bit_count() == 1:
                continue
            # Check (j, j+1) not edge
            if (current_path[j] ^ current_path[j+1]).bit_count() == 1:
                continue
            # Check new edges
            if (current_path[i-1] ^ current_path[j]).bit_count() != 1:
                continue
            if (current_path[i] ^ current_path[j+1]).bit_count() != 1:
                continue

        # Perform reversal
        new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
        
        # Try to extend
        extended_path = try_extend(new_path)
        new_len = len(extended_path) - 1
        
        delta = new_len - current_len
        if delta >= 0 or rnd.random() < math.exp(delta / T):
            current_path = extended_path
            current_len = new_len
            
            if current_len > best_len:
                best_len = current_len
                best_path = current_path
                
        T *= cooling
        
    # Final greedy extension on best_path
    final_path = try_extend(best_path)
    if len(final_path) - 1 > best_len:
        best_path = final_path
        best_len = len(final_path) - 1

    # Validation
    def is_valid(path):
        if not path: return False
        n = len(path)
        if n == 1: return True
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    if (path[i] ^ path[j]).bit_count() == 1:
                        return False
        return True

    if is_valid(best_path):
        return best_path
    else:
        # Fallback
        return [0]

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
