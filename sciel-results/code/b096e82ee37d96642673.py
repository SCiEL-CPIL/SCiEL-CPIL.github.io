"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute modular inverses: inv[i] = i^(-1) mod p
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # An interval [s, e] (inclusive, within [1, p-1]) is valid iff:
    # 1. No element x in [s,e] is a self-inverse (x^2 ≡ 1 mod p), i.e., x ∈ {1, p-1}
    # 2. For every x in [s,e], inv[x] is NOT in [s,e]
    #    Equivalently: for every x in [s,e], inv[x] < s or inv[x] > e
    
    # Strategy: For each possible start s, find the maximum valid end e.
    # We can do this efficiently by tracking, for each position, the constraints.
    
    # For a fixed start s, the interval [s, e] is valid if:
    #   - s != 1 and s != p-1 (diagonal check for s)
    #   - For all x in [s, e]: x != 1, x != p-1, and inv[x] not in [s, e]
    
    # Key insight: For a fixed s, as we extend e from s upward, the interval remains
    # valid until we hit an element x where inv[x] >= s and inv[x] <= e (i.e., inv[x] is
    # in [s, e]). Since inv[x] is fixed, the constraint for a given x is that e < inv[x]
    # (if inv[x] >= s) or s > inv[x] (if inv[x] < s, which is already satisfied if we
    # start at s and inv[x] < s).
    
    # More precisely, for start s, the maximum e is limited by:
    #   - e cannot include 1 or p-1
    #   - For each x in [s, e], if inv[x] >= s, then we need e < inv[x]
    #   - For each x in [s, e], if inv[x] < s, no additional constraint (inv[x] is outside)
    
    # So for a fixed s, the max valid e is:
    #   min over all x in [s, e] of (inv[x] - 1 if inv[x] >= s else infinity), 
    #   also bounded by p-2 (to avoid p-1) and by avoiding 1 (but s >= 2 if we avoid 1)
    
    # Let's compute this more carefully. For each start s (from 2 to p-2, since 1 and p-1 
    # are self-inverses and can't be in the interval):
    #   The interval [s, e] is valid iff for all x in [s, e]:
    #     a) x != 1 and x != p-1 (already satisfied if 2 <= s <= e <= p-2)
    #     b) inv[x] < s or inv[x] > e
    #   Condition (b) fails for x if s <= inv[x] <= e.
    #   So we need: for all x in [s, e], NOT (s <= inv[x] <= e)
    #   i.e., for all x in [s, e], inv[x] < s or inv[x] > e
    #   i.e., max over x in [s,e] of min(inv[x], ...) -- let me think differently.
    
    # For a fixed s, define the "blocking" elements: x where inv[x] >= s.
    # For such x, we need e < inv[x]. So the max e is min over all x in [s, e] with inv[x] >= s of (inv[x] - 1).
    # But this is circular (e appears on both sides).
    
    # Better approach: For each s, iterate e from s upward, and keep track of whether
    # the interval [s, e] is still valid. The interval becomes invalid at the first e
    # where either e is a self-inverse, or inv[e] is in [s, e].
    
    # Wait, that's not quite right either. Adding e to the interval [s, e-1] to make [s, e]
    # is invalid if:
    #   - e is a self-inverse (e == 1 or e == p-1), OR
    #   - inv[e] is in [s, e], i.e., s <= inv[e] <= e
    
    # But also, previous elements' constraints might be violated. No wait, if [s, e-1] was valid,
    # then for all x in [s, e-1], inv[x] was not in [s, e-1]. When we add e, we need inv[e] not in [s, e].
    # But also, for x in [s, e-1], we previously had inv[x] not in [s, e-1]. Now the interval is [s, e],
    # so we need inv[x] not in [s, e]. This means inv[x] could now be equal to e, which wasn't checked before.
    
    # So the condition for [s, e] to be valid given [s, e-1] was valid:
    #   - e is not a self-inverse
    #   - inv[e] is not in [s, e] (i.e., inv[e] < s or inv[e] > e)
    #   - For all x in [s, e-1], inv[x] != e (since inv[x] was not in [s, e-1], the only new value is e)
    
    # So we need: inv[e] not in [s, e] AND no x in [s, e-1] has inv[x] == e.
    # Note: inv[x] == e means x == inv[e]. So this is equivalent to: inv[e] not in [s, e].
    # Because if inv[e] == x for some x in [s, e-1], then x = inv[e], and inv[x] = inv[inv[e]] = e.
    # So the condition "no x in [s, e-1] has inv[x] == e" is the same as "inv[e] not in [s, e-1]".
    # Combined with "inv[e] not in [s, e]", which includes inv[e] != e, we get:
    #   inv[e] < s or inv[e] > e
    
    # So the incremental check is: [s, e] is valid iff [s, e-1] was valid AND (e not self-inverse) AND (inv[e] < s or inv[e] > e).
    
    # This means for each s, we can scan e from s upward until the first invalid e.
    
    best_len = 0
    best_start = 1
    
    for s in range(2, p - 1):  # s from 2 to p-2
        # Early termination: max possible length from s is p - 1 - s + 1 = p - s
        if p - s <= best_len:
            break
        
        valid = True
        for e in range(s, p - 1):  # e from s to p-2
            # Check if e is self-inverse
            if e == 1 or e == p - 1:
                break
            # Check if inv[e] is in [s, e]
            if s <= inv[e] <= e:
                break
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
