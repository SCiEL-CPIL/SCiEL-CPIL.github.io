"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    p = 1009
    p_minus_1 = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Strategy: Find the longest contiguous interval [s, e] in [1, 1008]
    # such that for ALL x, y in [s, e], x * y % p != 1.
    # 
    # Key insight: An interval [s, e] is valid iff:
    # 1. No element in [s, e] is a self-inverse (i.e., x^2 ≡ 1 mod p), so 1 and 1008 cannot be in [s, e].
    # 2. For no x in [s, e] is inv[x] also in [s, e] (including x itself, but we already handled self-inverses).
    #
    # So [s, e] is valid iff:
    # - s >= 2 and e <= 1007 (excluding self-inverses 1 and 1008)
    # - For all x in [s, e], inv[x] < s or inv[x] > e
    #
    # This means: for all x in [s, e], inv[x] is NOT in [s, e].
    # Equivalently: the set {inv[x] : x in [s, e]} is disjoint from [s, e].
    
    best_len = 0
    best_start = 2
    
    # We can iterate over possible starts s from 2 to 1007.
    # For each s, find the maximum e such that [s, e] is valid.
    # [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    # The constraint that breaks validity is: exists x in [s, e] with s <= inv[x] <= e.
    # For a fixed s, as e increases, we need to check if any x in [s, e] has inv[x] in [s, e].
    
    # For each s, the maximum valid e is the smallest e' >= s such that there exists x in [s, e'] with inv[x] in [s, e'], minus 1.
    # Actually, let's think differently. For a fixed s, define:
    # max_e(s) = max e such that for all x in [s, e], inv[x] < s or inv[x] > e.
    # This is equivalent to: for all x in [s, e], inv[x] not in [s, e].
    
    # Let's precompute for each position i, what is the "blocking" effect.
    # Actually, let's just do a direct search.
    
    # For each start s in [2, 1007]:
    #   Find the maximum e such that [s, e] is valid.
    #   The interval [s, e] is invalid if there exists x in [s, e] with inv[x] in [s, e].
    #   For a given s, as we extend e from s upward, the interval becomes invalid at the first e
    #   where some x in [s, e] has inv[x] in [s, e].
    
    # More efficiently: for each s, the constraint is that no x in [s, e] maps via inv into [s, e].
    # Let's define for each s: the "forbidden" e values are those where there exists x in [s, e] with s <= inv[x] <= e.
    # The smallest such e for a given s determines the max valid e.
    
    # Alternative approach: For each pair (x, inv[x]) with x != inv[x], both x and inv[x] cannot be in the same interval.
    # So the interval must avoid containing both x and inv[x] for any pair.
    # Also must avoid 1 and 1008.
    
    # Let's collect all "conflict pairs": for x in [2, 1007], if inv[x] != x, then (min(x, inv[x]), max(x, inv[x])) is a conflict pair.
    # An interval [s, e] is valid iff it doesn't contain both elements of any conflict pair, and doesn't contain 1 or 1008.
    
    # Since 1 and 1008 are excluded, we only consider s >= 2 and e <= 1007.
    
    # For each conflict pair (a, b) with a < b, an interval [s, e] contains both iff s <= a and b <= e.
    # So [s, e] is valid iff for all conflict pairs (a, b): NOT (s <= a and b <= e).
    # Equivalently: for all conflict pairs (a, b): s > a or e < b.
    
    # For a fixed s, the maximum valid e is: min over all conflict pairs (a, b) with a >= s of (b - 1), and also e <= 1007.
    # Wait, that's not quite right. Let me reconsider.
    
    # For a fixed s, [s, e] is invalid if there exists a conflict pair (a, b) with s <= a and b <= e.
    # So the maximum valid e for start s is: min(1007, min{b - 1 : (a, b) is a conflict pair with a >= s} - 1)
    # Wait, no. If there's a conflict pair (a, b) with a >= s, then e must be < b. So e <= b - 1.
    # The maximum valid e is min(1007, min{b : (a, b) is a conflict pair with a >= s} - 1).
    # But we also need e >= s.
    
    # Let's build this properly.
    
    # Collect conflict pairs
    conflict_pairs = []
    for x in range(2, 1008):  # x in [2, 1007]
        ix = inv[x]
        if ix != x and ix >= 2 and ix <= 1007:
            a, b = min(x, ix), max(x, ix)
            conflict_pairs.append((a, b))
    
    # Sort by a
    conflict_pairs.sort()
    
    # For each possible start s, find the minimum b among conflict pairs with a >= s.
    # Then max_e(s) = min(1007, min_b - 1) where min_b is the min b for pairs with a >= s.
    # If no such pair, max_e(s) = 1007.
    
    # Precompute suffix minimum of b values
    # Create an array where for each a, we store the minimum b among pairs with that a.
    # Then compute suffix minimum.
    
    min_b_for_a = [float('inf')] * (p_minus_1 + 2)  # index up to 1008
    for a, b in conflict_pairs:
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
    
    # Suffix minimum: suf_min[i] = min(min_b_for_a[i], min_b_for_a[i+1], ..., min_b_for_a[1008])
    suf_min = [float('inf')] * (p_minus_1 + 3)
    for i in range(p_minus_1, 0, -1):
        suf_min[i] = min(min_b_for_a[i], suf_min[i + 1])
    
    # Now for each s in [2, 1007]:
    #   min_b = suf_min[s]  (minimum b among conflict pairs with a >= s)
    #   If min_b == inf, max_e = 1007
    #   Else max_e = min(1007, min_b - 1)
    #   If max_e >= s, then length = max_e - s + 1
    
    for s in range(2, 1008):
        if time.time() > deadline:
            break
        
        min_b = suf_min[s]
        if min_b == float('inf'):
            max_e = 1007
        else:
            max_e = min(1007, min_b - 1)
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
