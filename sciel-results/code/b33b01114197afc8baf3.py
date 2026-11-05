import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [s, s+l-1] subset of [1, 1008]
    such that for ALL x, y in the interval (including x=y), x*y % 1009 != 1.
    Maximize l.
    """
    p = 1009
    n = 1008
    deadline = time.time() + 55

    # 1. Precompute the inverse map: inv[r] = x if x^-1 = r mod p
    # Since x is in [1, 1008], x^-1 is also in [1, 1008].
    inv_map = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = pow(x, p - 2, p)
        inv_map[ix] = x

    # 2. Find the "most restrictive" start for any window ending at R
    # Let S = [s, s+l-1]. For every element y in S, its inverse inv_map[y]
    # must NOT be in S.
    # This means for every y in [s, e]:
    #   1. inv_map[y] < s  OR  inv_map[y] > e
    #
    # We can determine the valid starts for a fixed end `e`.
    # Consider any candidate start `s`.
    # The window is [s, e].
    # A constraint is violated if there exists y in [s, e] such that inv_map[y] is also in [s, e].
    #
    # For a fixed `e`, let's find the set of forbidden starts.
    # If y is in the window, y is in [s, e]. This means s <= y <= e.
    # If inv_map[y] is also in the window, then s <= inv_map[y] <= e.
    # This implies that the interval [min(y, inv_map[y]), max(y, inv_map[y])] is "covered" by the window.
    # Actually, simpler logic:
    # For each y <= e, we require NOT (s <= inv_map[y] AND inv_map[y] <= e).
    # Assuming s <= y (since y is in the window starting at s), we must have either:
    #   a) inv_map[y] < s
    #   b) inv_map[y] > e
    #
    # If inv_map[y] > e, the condition is satisfied regardless of s (as long as s <= y).
    # If inv_map[y] <= e, we must have inv_map[y] < s, which means s > inv_map[y].
    # So, s must be at least max{ inv_map[y] + 1 | y <= e and inv_map[y] <= e }.
    #
    # Also, s must be <= y. Since we want the window to include y (which is up to e),
    # the constraint s <= min(y | y causes restriction) is not quite right.
    # The start s just needs to be <= e (since the interval must be non-empty)
    # and s must be small enough to cover the "bad" elements if we want a long interval.
    # But we are scanning `e`. For a fixed `e`, let's calculate the minimum valid `s`.
    #
    # Let M_e = max{ inv_map[y] + 1 | 1 <= y <= e and inv_map[y] <= e }.
    # If we pick s = M_e, then for any y in [M_e, e]:
    #   If inv_map[y] <= e, we have inv_map[y] + 1 <= M_e = s, so inv_map[y] < s.
    #   If inv_map[y] > e, it's not in the window.
    # So [M_e, e] is a valid window (provided M_e <= e).
    #
    # Can we start earlier? s' < M_e?
    # By definition, there exists some y* <= e such that inv_map[y*] + 1 = M_e and inv_map[y*] <= e.
    # If we start at s' = M_e - 1, then s' <= inv_map[y*] <= e.
    # Also, is y* in [s', e]? We need s' <= y*.
    # If s' <= y*, then y* is in the window.
    # Its inverse inv_map[y*] is also in the window (since s' <= inv_map[y*] <= e).
    # This creates a forbidden pair (y*, inv_map[y*]).
    # So we cannot start earlier than M_e, provided that y* >= M_e - 1.
    # Actually, we just need to know if such a y* exists such that the conflict is real.
    # The condition for M_e being a hard lower bound is that for the y* that achieved the max, y* must be inside the window [s', e].
    # Since we are looking for the *longest* interval, we want the smallest valid s.
    # If s = M_e is valid, is it the smallest?
    # If there is a y with inv_map[y] = M_e - 1, then s must be > M_e - 1 => s >= M_e.
    # This holds if M_e - 1 is in the window, i.e., if M_e - 1 <= e. (It is, since we only consider inv_map[y] <= e).
    # Does y itself need to be in the window? Yes, the conflict is (y, inv_map[y]).
    # If s > y, then y is not in the window.
    # So the constraint "s >= inv_map[y] + 1" is only active if y >= s.
    # This makes it slightly tricky: the bound M_e depends on y being in [s, e].
    #
    # Let's simplify by iterating e and checking if [s, e] is valid.
    # We know that if [s, e] is valid, any sub-interval is valid.
    # The optimal interval will have the property that moving s leftwards or e rightwards breaks validity.
    #
    # Alternative View:
    # Consider the graph where edges connect x and inv_map[x].
    # We want a path in the linear order 1..1008 that contains no complete edges (pairs x, inv_map[x]).
    # This is equivalent to finding the longest interval [s, e] such that for all x in [s, e], inv_map[x] is not in [s, e].
    #
    # Algorithm:
    # Iterate e from 1 to 1008.
    # Maintain the minimum valid start s for the current e.
    # Let current_min_start = 1.
    # For each e, update current_min_start based on constraints introduced by e.
    # But constraints from previous y might become irrelevant or relevant?
    # Actually, the condition is:
    # s > inv_map[y] for all y in [s, e] where inv_map[y] <= e.
    #
    # Let's just iterate e, and for each e, find the largest "forbidden" start.
    # If a pair (u, v) is fully contained in [s, e], it's forbidden.
    # This requires s <= min(u, v) and e >= max(u, v).
    # So, for a fixed e, we must have s > min(u, v) for every pair (u, v) such that max(u, v) <= e.
    # Let B_e = max { min(u, v) + 1 | pair (u, v) with max(u, v) <= e }.
    # If we set s = B_e, then for any pair (u, v) with max(u, v) <= e:
    # s = B_e > min(u, v).
    # Since u, v are in the window, we must have min(u, v) >= s.
    # But s > min(u, v). Contradiction.
    # Thus, no pair (u, v) can be fully contained in [s, e].
    # Is this sufficient?
    # We also need to ensure s <= e. If B_e > e, the window is empty (or invalid).
    # So valid lengths are e - B_e + 1 (if B_e <= e).
    #
    # We need to compute B_e efficiently.
    # Let's precompute for each pair (u, v) (where v = inv_map[u]) the value min(u, v).
    # Let w = max(u, v).
    # For a given e, we only care about pairs where w <= e.
    # We want to maximize min(u, v) among these pairs.
    # Let's create an array `max_min_at_w` where `max_min_at_w[w]` is the max of min(u, v) for pairs with max(u, v) = w.
    # Then B_e = max(1, max(B_{e-1}, max_min_at_w[e] + 1)).
    
    max_min_at_w = [0] * (n + 1)
    
    for u in range(1, n + 1):
        v = inv_map[u]
        # u and v are distinct (self-inverses 1, 1008) or identical.
        # If u == v (i.e., u=1 or u=1008), the pair is (u, u).
        # min(u, u) = u, max(u, u) = u.
        # If u is in the window, u*u % 1009 == 1 is forbidden.
        # So for u=1, w=1, min=1. Forbidden if 1 is in window.
        # For u=1008, w=1008, min=1008. Forbidden if 1008 is in window.
        
        w = max(u, v)
        m = min(u, v)
        if m + 1 > max_min_at_w[w]:
            max_min_at_w[w] = m + 1

    best_len = 0
    best_start = 1
    current_min_start = 1
    
    for e in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Update the minimum valid start based on pairs ending at e
        if max_min_at_w[e] > current_min_start:
            current_min_start = max_min_at_w[e]
            
        s = current_min_start
        if s <= e:
            curr_len = e - s + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = s
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
