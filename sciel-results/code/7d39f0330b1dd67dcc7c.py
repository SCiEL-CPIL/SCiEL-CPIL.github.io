"""Rewritten program: find the longest interval [s, e] within [1, 1008] such that
for all x, y in [s, e], x*y % 1009 != 1.

Approach:
1. Precompute modular inverses of 1..1008 mod 1009.
2. For each i in [1, 1008], the forbidden partners are i and inv[i] (if inv[i] != i).
   An interval [s, e] is valid iff for every i in [s, e], inv[i] is NOT in [s, e].
3. Use a two-pointer sweep to find the longest valid interval in O(n).
"""

def construct():
    MOD = 1009
    N = 1008  # values are 1..1008

    # Precompute modular inverse for each i in 1..1008
    # Since MOD is prime, inv[i] = pow(i, MOD-2, MOD)
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # For a valid interval [s, e], we need: for all i in [s, e], inv[i] not in [s, e].
    # Equivalently: for all i in [s, e], either inv[i] < s or inv[i] > e.
    #
    # Two-pointer approach:
    # For a fixed left endpoint s, the maximum valid right endpoint e is constrained by:
    #   For every i in [s, e], inv[i] must be outside [s, e].
    #   If inv[i] >= s, then we need inv[i] > e, i.e., e < inv[i].
    #   So e < min(inv[i] for i in [s, e] where inv[i] >= s).
    #
    # Alternatively, think of it as: an interval [s, e] is invalid if there exists
    # i in [s, e] such that inv[i] is also in [s, e].
    #
    # Let's use a different approach: for each pair (i, inv[i]), both i and inv[i]
    # cannot be in the interval simultaneously (unless inv[i] == i, i.e., i is a
    # self-inverse).
    #
    # Self-inverse elements: i * i % MOD == 1, i.e., i^2 ≡ 1 (mod 1009).
    # 1009 is prime, so solutions are i ≡ 1 and i ≡ -1 ≡ 1008 (mod 1009).
    # So 1 and 1008 are self-inverse. They cannot appear in any valid interval.
    #
    # For non-self-inverse pairs {i, inv[i]} with i != inv[i], at most one of them
    # can be in the interval.
    #
    # Strategy: mark forbidden positions. For each i in 1..1008:
    #   If i is self-inverse (i == 1 or i == 1008), it's always forbidden.
    #   Otherwise, the pair (i, inv[i]) means that if i is in the interval, inv[i]
    #   must not be, and vice versa.
    #
    # Two-pointer: maintain [s, e]. When expanding e, check if adding e creates a
    # conflict. A conflict occurs if inv[e] is in [s, e] (and inv[e] != e).
    # If conflict, we need to move s past inv[e] (or past e if e is self-inverse).
    
    # Actually, let me think more carefully.
    # For interval [s, e] to be valid:
    #   For every i in [s, e]: inv[i] not in [s, e].
    #
    # When we extend the right endpoint from e to e+1:
    #   We need inv[e+1] not in [s, e+1].
    #   Since e+1 is in [s, e+1], we need inv[e+1] != e+1 (not self-inverse) AND
    #   inv[e+1] not in [s, e].
    #   If inv[e+1] == e+1, then e+1 is self-inverse, so [s, e+1] is invalid.
    #   If inv[e+1] < s, then it's fine (inv[e+1] is outside the interval).
    #   If s <= inv[e+1] <= e, then conflict: we need to move s to inv[e+1] + 1.
    #   If inv[e+1] > e+1, then inv[e+1] is outside [s, e+1], so it's fine.
    #
    # But we also need to check that all existing elements in [s, e] still satisfy
    # the condition. However, if [s, e] was valid, and we only add e+1, the only
    # new constraint is about inv[e+1]. The existing elements already had their
    # inverses outside [s, e]. But now the interval is [s, e+1], so we need their
    # inverses to be outside [s, e+1]. Previously their inverses were outside [s, e].
    # If an inverse was exactly e+1, that would be a problem. But inv[i] = e+1 means
    # i = inv[e+1]. So if i is in [s, e] and inv[i] = e+1, then inv[e+1] = i which
    # is in [s, e], which is the same condition as above. So checking inv[e+1] is
    # sufficient.
    
    best_s = 1
    best_e = 1  # will be adjusted
    s = 1
    e = 0
    
    for e in range(1, N + 1):
        # Check if e is self-inverse
        if inv[e] == e:
            # e cannot be in any valid interval, so we must move s past e
            if s <= e:
                s = e + 1
            continue
        
        # Check if inv[e] is in [s, e-1] (it can't be e since inv[e] != e)
        if s <= inv[e] <= e - 1:
            # Conflict: need to move s past inv[e]
            s = inv[e] + 1
        # If inv[e] < s or inv[e] > e, no conflict
    
    # Wait, this approach has an issue. When I move s, I might create new conflicts
    # for elements that were previously fine. Let me reconsider.
    
    # Better approach: for each possible s, find the maximum valid e.
    # For a fixed s, e can be at most:
    #   min over i in [s, e] of (inv[i] - 1 if inv[i] >= s, else infinity)
    # This is circular. Let me use a different method.
    
    # Alternative: use the fact that the constraint is pairwise.
    # For each i, define forbidden range. If i is in the interval, then inv[i] must
    # not be. So for a fixed s, the maximum e is the minimum over all i >= s where
    # inv[i] >= s of (inv[i] - 1), and also e must not be a self-inverse.
    # But we also need to consider that as e increases, we include more i's, each
    # potentially constraining e further.
    
    # Let me just do a clean two-pointer:
    s = 1
    best_len = 0
    best_start = 1
    
    for e in range(1, N + 1):
        # Ensure [s, e] is valid by moving s forward as needed
        while s <= e:
            if inv[s] == s:
                # s is self-inverse, must exclude it
                s += 1
            elif s < inv[s] <= e:
                # inv[s] is in (s, e], conflict. Move s past inv[s]
                s = inv[s] + 1
            else:
                # inv[s] < s (impossible since s >= 1 and inv[s] >= 1) or inv[s] > e
                # Actually inv[s] < s is possible. In that case, inv[s] is not in [s, e], so fine.
                break
        
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    return [best_start, best_len]
