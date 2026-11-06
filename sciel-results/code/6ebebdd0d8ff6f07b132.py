import time
import random
import sys
from collections import defaultdict

# --- Problem Constants ---
N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
POINTS.sort()
P = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}
coords = POINTS

# Precomputation
# dist2[i][j] is squared distance between points i and j
dist2 = [[0]*P for _ in range(P)]
for i in range(P):
    xi, yi = coords[i]
    for j in range(P):
        xj, yj = coords[j]
        dist2[i][j] = (xi-xj)**2 + (yi-yj)**2

# iso_partners[u][v] is a bitmask of w such that {u, v, w} forms an isosceles triangle.
# Note: w != u and w != v.
iso_partners = [[0]*P for _ in range(P)]

# Optimization: Precompute for each pair (u, v), the list of w that form isosceles.
# We can compute this efficiently.
# For a fixed pair (u, v), d(u,v) is fixed.
# w forms isosceles if:
# 1. d(u,w) == d(u,v)  (w is on circle centered at u with radius d(u,v))
# 2. d(v,w) == d(u,v)  (w is on circle centered at v with radius d(u,v))
# 3. d(u,w) == d(v,w)  (w is on perpendicular bisector of uv)

# Let's just iterate all w for simplicity first, P=256 so P^3 = 16M is acceptable.
# But we can do better.
# For each u, group points by distance squared.
# dist_map[u] = dict: {dist_sq: list of indices}

dist_map = [defaultdict(list) for _ in range(P)]
for u in range(P):
    for v in range(P):
        if u != v:
            d = dist2[u][v]
            dist_map[u][d].append(v)

for u in range(P):
    for v in range(P):
        if u == v:
            continue
        d_uv = dist2[u][v]
        mask = 0
        # Case 1: d(u,w) == d(u,v)
        if d_uv in dist_map[u]:
            for w in dist_map[u][d_uv]:
                if w != u and w != v:
                    mask |= (1 << w)
        
        # Case 2: d(v,w) == d(u,v)
        if d_uv in dist_map[v]:
            for w in dist_map[v][d_uv]:
                if w != u and w != v:
                    mask |= (1 << w)
        
        # Case 3: d(u,w) == d(v,w)
        # This means w is equidistant from u and v.
        # We can check all w and see if dist2[u][w] == dist2[v][w].
        # But iterating all w for every u,v is P^3.
        # Let's see if we can precompute equidistant sets.
        # For each pair (u, v), the set of w such that d(u,w)==d(v,w).
        # This is the perpendicular bisector.
        # In a grid, this is a line.
        # Let's just do the P^3 loop for case 3 if it's not too slow, or optimize.
        # Actually, we can precompute eq_mask[u][v] for case 3? No, that's P^2 masks of P bits.
        # Let's just do the check in the loop for now, or merge into the previous loops?
        # Wait, the previous loops cover cases 1 and 2.
        # Case 3 is distinct.
        
        # Let's check if the P^3 loop for case 3 is the bottleneck.
        # 256 * 256 * 256 = 16M.
        # In Python, 16M simple operations is ~1s.
        # Let's try to avoid it.
        # We can precompute for each w, the set of pairs (u, v) such that d(u,w) == d(v,w)?
        # No, we need for each (u, v) the set of w.
        
        # Let's just do the inner loop for case 3.
        for w in range(P):
            if w == u or w == v:
                continue
            if dist2[u][w] == dist2[v][w]:
                mask |= (1 << w)
                
        iso_partners[u][v] = mask

# Precompute bit counts for masks? Or just use bin(x).count('1')?
# bin(x).count('1') is fast enough for 256-bit ints.

def is_independent(S_mask):
    """
    Check if S_mask is an independent set (no isosceles triangles).
    For all u, v in S, (S & iso_partners[u][v]) must be 0.
    """
    s = S_mask
    while s:
        lsb_u = s & -s
        u = lsb_u.bit_length() - 1
        s_rest = s ^ lsb_u
        t = s_rest
        while t:
            lsb_v = t & -t
            v = lsb_v.bit_length() - 1
            if S_mask & iso_partners[u][v]:
                return False
            t ^= lsb_v
        s = s_rest
    return True

def is_blocked(c, S_mask):
    """
    Check if point c is blocked by set S_mask.
    c is blocked if there exists u in S such that (S & iso_partners[u][c]) != 0.
    """
    s = S_mask
    while s:
        lsb = s & -s
        u = lsb.bit_length() - 1
        if S_mask & iso_partners[u][c]:
            return True
        s ^= lsb
    return False

def get_unblocked_points(S_mask):
    """
    Return list of points not in S that are NOT blocked by S.
    """
    not_in_S = ((1 << P) - 1) ^ S_mask
    unblocked = []
    s = not_in_S
    while s:
        lsb = s & -s
        c = lsb.bit_length() - 1
        if not is_blocked(c, S_mask):
            unblocked.append(c)
        s ^= lsb
    return unblocked

def is_maximal(S_mask):
    """
    Check if S_mask is a maximal independent set.
    All points not in S must be blocked.
    """
    not_in_S = ((1 << P) - 1) ^ S_mask
    s = not_in_S
    while s:
        lsb = s & -s
        c = lsb.bit_length() - 1
        if not is_blocked(c, S_mask):
            return False
        s ^= lsb
    return True

def construct():
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    best_size = P
    best_set = 0
    
    # Strategy:
    # 1. Simulated Annealing / Local Search
    # State: S_mask
    # Objective: Minimize |S| subject to S being independent and maximal.
    # Moves:
    #   - Remove a point v from S. (If S\{v} is independent and maximal, accept. If independent but not maximal, we need to repair? Or reject?)
    #   - Add a point v to S. (If S\{v} is not independent, reject. If independent, check maximality? Adding a point might make it non-maximal? No, adding a point to a maximal set makes it non-maximal? Wait.
    #     If S is maximal, adding v makes S' = S U {v}. Is S' maximal?
    #     Points not in S' are subset of points not in S.
    #     If a point u not in S' was blocked by S, it's still blocked by S'.
    #     So S' is maximal if S is maximal?
    #     Wait, if S is maximal, every u not in S is blocked by S.
    #     If we add v to S, the set of points not in S shrinks.
    #     The remaining points are still blocked by S (and thus S').
    #     So S' is maximal.
    #     But S' is NOT independent if v conflicts with S.
    #     So we can only add v if it's independent with S.
    #     If we add v, size increases. We want to minimize size.
    #     So adding is generally bad, unless it allows future removals.
    
    #   - Swap: Remove v, add u.
    #     If S\{v} U {u} is independent and maximal, accept.
    
    # Let's use a "Repair" approach for removal.
    # If removing v makes S not maximal, find unblocked points U.
    # Try to add some u in U to block them? No, adding u increases size.
    # Try to add some w NOT in U that blocks U?
    # If we remove v, and u is unblocked, we need to add w such that u is blocked by S\{v} U {w}.
    # And S\{v} U {w} must be independent.
    # This is a "swap" v -> w.
    # If we can find w such that swap reduces size? No, swap keeps size same.
    # We need to find a sequence of swaps that leads to a state where a removal is possible.
    
    # Simulated Annealing:
    # Energy = |S| + lambda * (number of unblocked points) + mu * (number of conflicts)
    # We want Energy -> 0.
    # If we maintain independence, conflicts = 0.
    # So Energy = |S| + lambda * (unblocked_count).
    # We want to minimize |S|, but we must have unblocked_count = 0.
    # So we can use a penalty for unblocked points.
    
    # Let's try a simpler local search first:
    # 1. Start with a maximal independent set (greedy).
    # 2. Loop:
    #    a. Try to remove a point v.
    #       - If S\{v} is independent and maximal, accept.
    #       - If S\{v} is independent but not maximal, find unblocked points.
    #         Try to add a point w to block them.
    #         If we can add w such that S\{v} U {w} is independent and maximal, accept (swap).
    #    b. Try to swap v for w to improve "potential" for removal?
    
    # Let's implement the "Remove or Swap" local search.
    
    def get_conflicts(p, S_mask):
        """
        Return bitmask of points in S that conflict with p.
        p conflicts with u in S if (S & iso_partners[p][u]) != 0?
        No, p conflicts with u if {p, u, w} is isosceles for some w in S.
        So for each u in S, if (S & iso_partners[p][u]) != 0, then p conflicts with u.
        Wait, iso_partners[p][u] contains w such that {p, u, w} is isosceles.
        If w is in S, then {p, u, w} is an isosceles triangle in S U {p}.
        So p conflicts with u if (S & iso_partners[p][u]) != 0.
        """
        conflicts = 0
        s = S_mask
        while s:
            lsb = s & -s
            u = lsb.bit_length() - 1
            if S_mask & iso_partners[p][u]:
                conflicts |= (1 << u)
            s ^= lsb
        return conflicts

    def can_add(p, S_mask):
        """
        Check if p can be added to S_mask while maintaining independence.
        """
        return (get_conflicts(p, S_mask) == 0)

    # Precompute a list of all points for random selection
    all_points = list(range(P))
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Greedy Construction
        # Random order
        order = list(all_points)
        random.shuffle(order)
        
        S_mask = 0
        for p in order:
            if can_add(p, S_mask):
                S_mask |= (1 << p)
        
        current_size = bin(S_mask).count('1')
        
        # 2. Local Search
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            removal_order = list(all_points)
            random.shuffle(removal_order)
            
            for v in removal_order:
                if not (S_mask & (1 << v)):
                    continue
                
                new_S_mask = S_mask & ~(1 << v)
                
                # Check independence (should be true if S was independent and we removed a point)
                # But let's be safe.
                # Actually, removing a point from an independent set always yields an independent set.
                
                # Check maximality
                if is_maximal(new_S_mask):
                    S_mask = new_S_mask
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = S_mask
                else:
                    # Try to repair by swapping
                    # Find unblocked points
                    unblocked = get_unblocked_points(new_S_mask)
                    if not unblocked:
                        # This shouldn't happen if is_maximal returned False
                        continue
                    
                    # Try to add a point w to block the unblocked points
                    # We want w such that:
                    # 1. w is not in new_S_mask
                    # 2. w is independent with new_S_mask
                    # 3. new_S_mask U {w} is maximal (i.e., w blocks all unblocked points? No, w just needs to make the set maximal)
                    #    Actually, if we add w, the set of unblocked points might change.
                    #    We need new_S_mask U {w} to be maximal.
                    
                    # To be efficient, we can filter candidates w:
                    # w must be independent with new_S_mask.
                    # w should block at least one unblocked point?
                    # If w doesn't block any unblocked point, and doesn't unblock any blocked point (impossible, adding points only blocks more),
                    # then the set of unblocked points remains the same or shrinks.
                    # So w must block at least one point in `unblocked`.
                    
                    candidates = []
                    not_in_new_S = ((1 << P) - 1) ^ new_S_mask
                    s_cand = not_in_new_S
                    while s_cand:
                        lsb = s_cand & -s_cand
                        w = lsb.bit_length() - 1
                        if can_add(w, new_S_mask):
                            # Check if w blocks any unblocked point
                            blocks_any = False
                            for u in unblocked:
                                if is_blocked(u, new_S_mask | (1 << w)):
                                    blocks_any = True
                                    break
                            if blocks_any:
                                candidates.append(w)
                        s_cand ^= lsb
                    
                    # Try candidates
                    random.shuffle(candidates)
                    for w in candidates:
                        candidate_mask = new_S_mask | (1 << w)
                        if is_maximal(candidate_mask):
                            # Accept swap
                            S_mask = candidate_mask
                            # Size remains same
                            improved = True
                            break
                        # If not maximal, we could try to continue searching, but let's keep it simple
                        # and just try the next candidate or break.
                        # Trying all candidates might be expensive.
                        # Let's just try the first few.
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_mask

    # Convert best_set to list of points
    result = []
    for i in range(P):
        if best_set & (1 << i):
            result.append(POINTS[i])
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    # Verify
    mask = 0
    for p in res:
        idx = POINT_INDEX[p]
        mask |= (1 << idx)
    if not is_independent(mask):
        print("ERROR: Not independent")
    if not is_maximal(mask):
        print("ERROR: Not maximal")
