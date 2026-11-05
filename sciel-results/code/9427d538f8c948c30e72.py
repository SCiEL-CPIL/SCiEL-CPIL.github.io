"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    The condition x*y ≡ 1 (mod 1009) means y ≡ x^(-1) (mod 1009).
    So we need an interval where no element is the modular inverse of another element
    (including itself).
    
    Self-inverse elements: x^2 ≡ 1 (mod 1009) => x ≡ 1 or x ≡ -1 ≡ 1008 (mod 1009).
    So 1 and 1008 cannot be in any valid interval.
    
    For other elements, if x is in the interval, then inv(x) cannot be in the interval.
    
    We use a two-pointer sliding window approach:
    - Precompute modular inverses for all values in [1, 1008].
    - Use a sliding window to find the longest interval where no two elements are inverses.
    """
    MOD = 1009
    N = 1008  # max value
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        # Use Fermat's little theorem since 1009 is prime
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer sliding window
    # We want the longest contiguous interval [l, r] such that for all x, y in [l, r],
    # x * y % MOD != 1.
    # 
    # Key insight: if x is in the window, then inv[x] cannot be in the window.
    # So we can track which inverses are "blocked" by elements in the current window.
    
    # For a window [l, r], it's valid if for every x in [l, r], inv[x] is NOT in [l, r].
    
    # We'll use a two-pointer approach:
    # For each left pointer l, extend r as far as possible.
    # When adding r, check if inv[r] is already in [l, r-1]. If so, we can't extend further
    # from this l... but actually we need to be more careful.
    
    # Alternative: for each position r, what's the earliest l such that [l, r] is valid?
    # If we add r to the window, we need inv[r] to not be in the window.
    # So l must be > inv[r] if inv[r] >= l, or equivalently, if inv[r] is in [l, r-1], we need to move l.
    
    # Let's think differently. For a window [l, r], it's invalid if there exists x in [l, r] 
    # such that inv[x] is also in [l, r].
    
    # For each r, the constraint from r itself is: inv[r] must not be in [l, r].
    # Since inv[r] could be less than r or greater than r.
    # If inv[r] < r, then we need l > inv[r].
    # If inv[r] > r, then inv[r] is not in [l, r] automatically (since inv[r] > r).
    
    # But we also need to check all previous elements in the window.
    
    # Better approach: for each possible start l, find the maximum r such that [l, r] is valid.
    # [l, r] is valid iff for all x in [l, r], inv[x] is not in [l, r].
    
    # Let's precompute for each x, the "conflict" with inv[x].
    # For a window to be valid, no pair (x, inv[x]) should both be in the window.
    
    # Two-pointer: 
    # Maintain window [l, r]. When we try to extend to r+1, check if inv[r+1] is in [l, r].
    # If yes, we need to advance l until inv[r+1] is no longer in the window.
    # But advancing l might also resolve conflicts for other elements.
    
    # Actually, let's just do it straightforwardly:
    # For each l from 1 to N, find the max r >= l such that [l, r] is valid.
    # Use a sliding window where we track the minimum "blocking" constraint.
    
    # For a window [l, r], it's valid if for every x in [l, r], inv[x] < l or inv[x] > r.
    # Equivalently, min_{x in [l,r]} (something) ... 
    
    # Let me think of it as: for each x, define constraint that l > inv[x] OR r < inv[x].
    # For a fixed r, we need l > max{inv[x] : x in [l, r] and inv[x] < r} ... this is circular.
    
    # Simpler: just iterate and use the two-pointer with a set or count.
    
    # Let's use the approach: for each r, maintain the smallest l such that [l, r] is valid.
    # When we move r to r+1, the new element r+1 has inverse inv[r+1].
    # If inv[r+1] is in [l, r], we need to increase l to inv[r+1] + 1.
    # But increasing l might make some previous elements' inverses fall out of the window,
    # which is fine (it only helps validity).
    
    # Wait, that's not quite right. Let me reconsider.
    
    # A window [l, r] is valid if and only if for all x in [l, r], inv[x] is not in [l, r].
    
    # Consider extending the window from [l, r] to [l, r+1]. The new element is r+1.
    # We need inv[r+1] not in [l, r+1]. Since r+1 is in the window, we need inv[r+1] != r+1 
    # (which is true unless r+1 is self-inverse, i.e., r+1 in {1, 1008}) and inv[r+1] not in [l, r].
    
    # Also, for all existing x in [l, r], we already had inv[x] not in [l, r]. Now the window
    # is [l, r+1], so we additionally need inv[x] != r+1 for all x in [l, r].
    # inv[x] = r+1 means x = inv[r+1]. So we need inv[r+1] not in [l, r].
    # This is the same condition! So the only new constraint is that inv[r+1] is not in [l, r].
    
    # So the algorithm:
    # l = 1
    # For r = 1 to N:
    #   While inv[r] is in [l, r-1] (i.e., l <= inv[r] <= r-1):
    #     l = inv[r] + 1
    #   But wait, this only handles the constraint from r. What about constraints from previous elements?
    
    # Hmm, let me reconsider. When we have window [l, r-1] that is valid, and we add r:
    # We need to ensure inv[r] is not in [l, r]. 
    # If inv[r] < l, fine. If inv[r] > r, fine. If l <= inv[r] <= r, problem.
    # But also, for each x in [l, r-1], we need inv[x] not in [l, r]. Previously inv[x] was not in [l, r-1].
    # Now inv[x] could equal r. That means x = inv[r]. So we need inv[r] not in [l, r-1].
    # Same condition!
    
    # So indeed, when extending from [l, r-1] to [l, r], the only new constraint is:
    # inv[r] must not be in [l, r]. Which simplifies to: if inv[r] >= l and inv[r] <= r, we have a conflict.
    # Since inv[r] <= r means inv[r] < r (can't equal r unless r is self-inverse), we need:
    # if l <= inv[r] < r, then we must advance l to inv[r] + 1.
    
    # But advancing l might create issues? No, because removing elements from the left can only
    # make the window more valid (fewer pairs to check).
    
    # Wait, but if we advance l, we might still have conflicts with other elements?
    # Let's say we had a valid window [l, r-1]. We add r. If inv[r] is in [l, r-1], we set l = inv[r] + 1.
    # Now the window is [inv[r]+1, r]. Is this valid?
    # For any x in [inv[r]+1, r-1], inv[x] was not in [l_old, r-1] where l_old <= inv[r].
    # Now the window starts at inv[r]+1, so inv[x] could potentially be in [inv[r]+1, r-1] even if it wasn't in [l_old, r-1]?
    # No, because [inv[r]+1, r-1] is a subset of [l_old, r-1], so if inv[x] wasn't in [l_old, r-1], it's not in the subset.
    # And for x = r, we've ensured inv[r] is not in [inv[r]+1, r] (since inv[r] < inv[r]+1).
    # So yes, the window remains valid.
    
    # Great, so the algorithm is:
    l = 1
    best_start = 1
    best_length = 0
    
    for r in range(1, N + 1):
        # Check if r is self-inverse
        if r == 1 or r == N:  # 1 and 1008
            # These can't be in any valid window, so reset
            # Actually, if r is self-inverse, inv[r] = r, which is in [l, r], so we need l > r, which is impossible.
            # So we skip this r entirely by setting l = r + 1
            l = r + 1
            continue
        
        inv_r = inv[r]
        # If inv_r is in [l, r-1], we need to advance l
        if l <= inv_r <= r - 1:
            l = inv_r + 1
        
        # Now [l, r] should be valid
        if r - l + 1 > best_length:
            best_length = r - l + 1
            best_start = l
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
