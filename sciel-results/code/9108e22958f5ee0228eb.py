"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses for all residues 1..p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Find the longest valid interval
    # An interval [s, e] is valid if for every x in [s,e], inv[x] is not in [s,e]
    # Equivalently, no pair {x, inv[x]} is fully contained in [s,e]
    
    best_len = 0
    best_start = 1
    
    # For each start s, find the maximum valid end e
    # As we extend e from s, the interval becomes invalid when we hit
    # some x such that inv[x] is also in [s, e]
    
    for s in range(1, p):
        # If even extending to p-1 can't beat best_len, skip
        if (p - 1) - s + 1 <= best_len:
            break
        
        # Track which inverses we've "covered" - when we add x to the interval,
        # if inv[x] is already in [s, current_e], it's invalid
        # Better: for a given s, find the smallest e >= s such that the interval [s,e] is invalid
        # The interval [s,e] is invalid iff there exists x in [s,e] with inv[x] in [s,e]
        
        # For fixed s, as e increases, the constraint is:
        # For each x in [s,e], inv[x] must be < s or > e
        # The binding constraint is the minimum over x in [s,e] of:
        #   if inv[x] < s: no constraint on e from x
        #   if inv[x] >= s: we need e < inv[x] (if inv[x] > x) or e < ... 
        # Actually: if inv[x] >= s and inv[x] <= e, it's a conflict.
        # So for x in [s,e], we need inv[x] < s or inv[x] > e.
        # The maximum valid e for start s is: min over x in [s, e_max] of (inv[x]-1) where inv[x] >= s
        # But this is circular. Let me just iterate.
        
        # For start s, the interval [s,e] is valid iff for all x in [s,e]: inv[x] < s or inv[x] > e
        # Let's find the max e by checking constraints:
        # For each x >= s, if inv[x] >= s, then e must be < inv[x] (to avoid having both x and inv[x])
        # Wait, that's not quite right either. If inv[x] >= s, we need either x not in interval or inv[x] not in interval.
        # Since x is in [s,e], we need inv[x] > e.
        # So e < inv[x] for all x in [s,e] where inv[x] >= s.
        
        # The maximum valid e is: min of (inv[x]-1) for all x >= s where inv[x] >= s, but also e >= s.
        # Actually, we need e such that for ALL x in [s,e], inv[x] < s or inv[x] > e.
        # Let me define: for a candidate e, it's valid if for all x in [s,e], inv[x] < s or inv[x] > e.
        # The constraint from x is: if inv[x] >= s, then e < inv[x].
        # So max valid e = min(inv[x]-1 for x in [s, e_candidate] where inv[x] >= s)
        # But we don't know e_candidate. Let's iterate e from s upward and check.
        
        max_e = s - 1  # no valid interval yet
        for e in range(s, p):
            # Check if [s,e] is valid: for all x in [s,e], inv[x] < s or inv[x] > e
            # We can check incrementally: when we add e to the interval, check if inv[e] is in [s,e]
            # If inv[e] >= s and inv[e] <= e, then it's invalid (since inv[e] is already in [s,e-1] or e itself)
            # Also check: for any x in [s,e-1], is inv[x] == e? That would mean inv[x] is now in the interval.
            # So we need: inv[e] < s or inv[e] > e, AND for all x in [s,e-1], inv[x] != e.
            # The second condition is equivalent to: e is not the inverse of any x in [s,e-1], i.e., inv[e] is not in [s,e-1].
            # But inv[e] is a single value, so inv[e] in [s,e-1] iff s <= inv[e] <= e-1.
            # And inv[e] == e iff e is self-inverse (e=1 or e=p-1).
            
            # So [s,e] is valid iff:
            # 1. inv[e] < s or inv[e] > e (e's inverse is outside [s,e])
            # 2. For all x in [s,e-1], inv[x] > e or inv[x] < s (already checked when we added x)
            # Condition 2 was satisfied for [s,e-1]. Adding e only adds the constraint that inv[e] is not in [s,e].
            # And for existing x in [s,e-1], we need inv[x] != e, i.e., inv[x] not in [s,e] but it was not in [s,e-1], 
            # so the only new possibility is inv[x] == e.
            
            # So [s,e] is valid iff [s,e-1] was valid AND inv[e] < s AND (inv[e] > e or inv[e] not in [s,e-1])
            # Wait, inv[e] < s already means inv[e] is not in [s,e]. 
            # If inv[e] > e, it's not in [s,e]. 
            # If s <= inv[e] <= e, then inv[e] is in [s,e], invalid.
            # But we also need: for x in [s,e-1], inv[x] != e. Since inv is a bijection, inv[x]=e iff x=inv[e].
            # So we need inv[e] not in [s,e-1], i.e., inv[e] < s or inv[e] > e-1, i.e., inv[e] < s or inv[e] >= e.
            # Combined with inv[e] not in [s,e]: inv[e] < s or inv[e] > e.
            # And inv[e] not in [s,e-1]: inv[e] < s or inv[e] >= e.
            # The stricter one is inv[e] < s or inv[e] > e.
            
            # So the incremental check: [s,e] valid iff [s,e-1] valid and (inv[e] < s or inv[e] > e)
            # BUT we also need to check that no x in [s,e-1] has inv[x] == e.
            # inv[x] == e iff x == inv[e]. So we need inv[e] not in [s, e-1].
            # inv[e] not in [s, e-1] means inv[e] < s or inv[e] >= e.
            # Combined: we need inv[e] < s or inv[e] > e (for inv[e] not in [s,e]) 
            # AND inv[e] < s or inv[e] >= e (for inv[e] not in [s,e-1]).
            # The intersection: if inv[e] < s, both satisfied. If inv[e] > e, both satisfied. 
            # If e <= inv[e] <= e, impossible. If s <= inv[e] < e, first condition fails.
            # If inv[e] == e, then inv[e] is in [s,e] (if s<=e), so first condition fails unless s > e.
            
            # So the check is simply: inv[e] < s or inv[e] > e.
            # Wait, but what about the case where inv[e] is in [s, e-1]? That means s <= inv[e] <= e-1.
            # Then inv[e] is in the interval [s,e], so having both e and inv[e] is a conflict.
            # So yes, we need inv[e] < s or inv[e] > e.
            
            # But hold on - this only checks the pair involving e. What about pairs not involving e?
            # Those were already checked when we built up to e-1. So incremental is correct.
            
            if e == s:
                # First element: check if inv[s] is in [s,s], i.e., inv[s] == s
                # inv[s] == s means s is self-inverse: s=1 or s=p-1
                if inv[s] == s:
                    break
                else:
                    max_e = s
            else:
                # Check if adding e creates a conflict
                if inv[e] < s or inv[e] > e:
                    max_e = e
                else:
                    break
            
            if max_e - s + 1 > best_len:
                best_len = max_e - s + 1
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
