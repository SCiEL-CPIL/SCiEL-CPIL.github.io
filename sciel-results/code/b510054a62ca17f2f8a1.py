import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 % MOD == 1
    self_inv = set()
    for i in range(1, N + 1):
        if i * i % MOD == 1:
            self_inv.add(i)

    # For each position i, compute the minimum start index such that
    # [start, i] contains no conflict pair (x, inv[x]) where both are in [start, i].
    # A conflict exists in [s, e] if there's some x in [s, e] such that inv[x] is also in [s, e].
    # For a fixed end e, the constraint is: for all x in [s, e], inv[x] < s or inv[x] > e.
    # Since inv[x] > e is automatic when inv[x] > e, we need: for all x in [s, e], inv[x] < s.
    # Wait, that's not quite right. Let me think more carefully.
    #
    # For interval [s, e], the condition is: for every x, y in [s, e], x*y % MOD != 1.
    # This means: for every x in [s, e], inv[x] is NOT in [s, e].
    # So for every x in [s, e], either inv[x] < s or inv[x] > e.
    #
    # For a fixed e, as we decrease s, the condition becomes harder to satisfy
    # because more x's are included and the range [s, e] grows.
    #
    # Let's define for each position e, the maximum valid start s_max(e) such that
    # [s_max(e), e] is valid. Then the answer is max over e of (e - s_max(e) + 1).
    #
    # Actually, let's use a different approach. For each end position e, we want the
    # largest s such that [s, e] is valid. As s decreases, we add elements to the left.
    # When we add element s-1, we need to check if inv[s-1] is in [s, e]. If it is,
    # then s-1 cannot be in the interval (along with e), so the minimum valid start
    # for end e is at least s.
    #
    # Let's define conflict_left[i] = the minimum start index s such that if i is in
    # [s, e] for some e >= i, then inv[i] must not be in [s, e].
    # If inv[i] <= i, then for [s, e] to be valid with i in it, we need s > inv[i].
    # If inv[i] > i, then for [s, e] to be valid with i in it and e < inv[i], it's fine.
    # But if e >= inv[i], then we need s > inv[i]... wait no.
    #
    # Let me reconsider. For a fixed interval [s, e], the condition fails if there
    # exists x in [s, e] such that inv[x] is also in [s, e].
    #
    # For each x, the "conflict" with inv[x] means: if both x and inv[x] are in the
    # interval, it's invalid. So for each pair (x, inv[x]) with x != inv[x], if the
    # interval contains both, it's invalid. For self-inverse x, if the interval contains
    # x, it's invalid.
    #
    # For a fixed end e, the minimum valid start is:
    # s_min(e) = max over all x in [s, e] of (if inv[x] <= e, then inv[x]+1, else 1)
    # But this is circular since s appears on both sides.
    #
    # Better approach: For each position e, compute the constraint on s:
    # For each x such that x <= e, if inv[x] <= e, then we need s > min(x, inv[x]) 
    # ... no wait. We need that NOT both x and inv[x] are in [s, e].
    # If inv[x] <= e, then to avoid having both in [s,e], we need s > inv[x] (if inv[x] < x)
    # or s > x (if x < inv[x]). Actually, we need s > min(x, inv[x]) when both are <= e?
    # No: if both x and inv[x] are in [s, e], that's bad. So we need s > max(x, inv[x]) 
    # ... no, s > min(x, inv[x]) doesn't help if the larger one is still in range.
    #
    # Let me think again. Both x and inv[x] are in [s, e] iff s <= min(x, inv[x]) AND 
    # max(x, inv[x]) <= e. So to avoid this, for each pair (x, inv[x]) with max(x,inv[x]) <= e,
    # we need s > min(x, inv[x]).
    #
    # So for fixed e: s_min(e) = 1 + max{ min(x, inv[x]) : x in [1,e] and inv[x] <= e and x != inv[x] }
    # Also, for self-inverse x in [1, e], we need s > x, i.e., s >= x+1.
    #
    # Let's define for each e:
    #   bound(e) = max of:
    #     - x for self-inverse x <= e
    #     - min(x, inv[x]) for x <= e where inv[x] <= e and x != inv[x]
    #   Then s_min(e) = bound(e) + 1, and the length is e - s_min(e) + 1 = e - bound(e).
    #
    # We can compute bound(e) incrementally. As e increases, new constraints are added.

    # For each e, we track the maximum "conflict bound"
    # conflict_bound[e] = max value such that s must be > conflict_bound[e] for [s,e] to be valid
    
    conflict_bound = [0] * (N + 1)
    current_bound = 0

    for e in range(1, N + 1):
        if e in self_inv:
            current_bound = max(current_bound, e)
        else:
            ie = inv[e]
            if ie <= e:
                # The pair (e, ie) with ie < e (since e is not self-inverse)
                # Both are in [s, e] if s <= ie and e <= e (always true)
                # So we need s > ie
                current_bound = max(current_bound, ie)
            # If ie > e, then for end position e, the pair (e, ie) doesn't create
            # a constraint yet since ie > e means ie is not in [s, e]
        
        conflict_bound[e] = current_bound

    best_start = 2
    best_len = 1

    for e in range(1, N + 1):
        s_min = conflict_bound[e] + 1
        if s_min > e:
            continue
        length = e - s_min + 1
        if length > best_len:
            best_len = length
            best_start = s_min

    return [best_start, best_len]
