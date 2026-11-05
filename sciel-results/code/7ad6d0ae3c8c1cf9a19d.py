import time

def construct():
    """
    Construct a nonempty subset S of [1, 256] maximizing cardinality
    such that for all a < b <= c in S, a does not divide b+c.
    
    Strategy:
    1. Start with the guaranteed valid set [171, 256] (size 86).
    2. Greedily add smaller elements using a fast bitset-based divisibility check.
    3. Use freed time for local search: try swapping elements to increase density.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Precompute: for each x in 1..256, the bitmask of sums s in [2,512] where x | s
    # We represent the set of achievable sums b+c (b<=c, b,c in S) as a bitset.
    # sum_bits: bit s is set iff there exist b<=c in S with b+c = s.
    # To check if x is compatible: no bit in sum_bits should be set at a position that's a multiple of x.
    # Precompute conflict_mask[x] = bitmask with bits set at all multiples of x in [2, 512].
    # Then x is compatible iff (sum_bits & conflict_mask[x]) == 0.
    
    # conflict_mask[x]: bits at positions 2x, 3x, 4x, ... up to 512
    conflict_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for k in range(2, 513 // x + 1):
            s = x * k
            if s <= 512:
                mask |= (1 << s)
        conflict_mask[x] = mask

    # Initialize S with [171, 256]
    S = set(range(171, n + 1))
    
    # Build sum_bits for initial S
    def build_sum_bits(S_set):
        bits = 0
        elems = sorted(S_set)
        L = len(elems)
        for i in range(L):
            b = elems[i]
            for j in range(i, L):
                c = elems[j]
                bits |= (1 << (b + c))
        return bits

    sum_bits = build_sum_bits(S)

    # Greedy: try adding elements from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline:
            break
        # Check compatibility: x must not divide any b+c for b<=c in S
        # i.e., (sum_bits & conflict_mask[x]) == 0
        if (sum_bits & conflict_mask[x]) == 0:
            S.add(x)
            # Update sum_bits: new sums involving x
            # x is smaller than all existing elements, so x can pair with any element
            # New sums: x + y for y in S (y > x, so x < y, valid as b=x, c=y or b=y, c=x? No, b<=c)
            # Since x < all y in old S, pairs are (x, y) with x <= y, sum = x+y
            # Also (x, x) but x not in S before, so only x+y for y in S
            for y in S:
                if y != x:
                    sum_bits |= (1 << (x + y))
            # Note: x+x would be 2x, but x wasn't in S before so this pair is new
            # Actually x is now in S, so x+x is a valid pair
            sum_bits |= (1 << (2 * x))

    # Local search: try to improve by swapping or adding/removing elements
    # Strategy: try to add any remaining element, or swap a pair
    # We'll do multiple passes of local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding any element not in S
        for x in range(1, n + 1):
            if x in S:
                continue
            if (sum_bits & conflict_mask[x]) == 0:
                S.add(x)
                for y in S:
                    if y != x:
                        sum_bits |= (1 << (x + y))
                sum_bits |= (1 << (2 * x))
                improved = True
                break
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove u, add v
        # For each u in S, for each v not in S, check if S - {u} + {v} is valid
        # This is expensive. Let's try a smarter approach:
        # For each u in S, compute what sums become available when u is removed.
        # Then check if v can be added to S - {u}.
        
        # Alternative: try removing one element and see if we can add two
        # Or just do pairwise swaps with a time limit
        
        # Let's try: for each pair (u in S, v not in S), check validity of swap
        # To speed up, we can precompute for each u the "conflict" it causes
        # But let's just do a limited search
        
        S_list = sorted(S)
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        
        # Try swaps: remove u, add v
        # For efficiency, only try a subset of swaps per iteration
        max_swaps_to_try = 2000
        swaps_tried = 0
        for u in S_list:
            if swaps_tried >= max_swaps_to_try:
                break
            if time.time() > deadline - 5:
                break
            # Temporarily remove u
            S.remove(u)
            # Rebuild sum_bits without u (expensive, but let's try)
            # Actually, let's compute what sums u contributed and remove them
            # u contributed: u+y for y in S (old, including u itself)
            # After removing u, sums that ONLY u contributed to should be cleared
            # This is complex. Let's just rebuild.
            old_S = set(S)
            new_sum_bits = build_sum_bits(old_S)
            
            # Now try to add v
            for v in not_in_S:
                if swaps_tried >= max_swaps_to_try:
                    break
                swaps_tried += 1
                if (new_sum_bits & conflict_mask[v]) == 0:
                    # Valid swap!
                    S.add(v)
                    # Update sum_bits
                    for y in S:
                        if y != v:
                            new_sum_bits |= (1 << (v + y))
                    new_sum_bits |= (1 << (2 * v))
                    sum_bits = new_sum_bits
                    improved = True
                    break
            if not improved:
                # Restore u
                S.add(u)
            else:
                break
            # Restore u if we didn't find a swap
            if u not in S:
                S.add(u)
        
        if not improved and time.time() < deadline - 10:
            # Try 2-for-1: remove two elements, add one
            # Or 1-for-2: remove one, add two
            # Let's try 1-for-2: remove u, add v1 and v2
            for u in sorted(S):
                if time.time() > deadline - 5:
                    break
                S.remove(u)
                old_S = set(S)
                new_sum_bits = build_sum_bits(old_S)
                
                # Try adding one element
                added_one = None
                for v in not_in_S:
                    if (new_sum_bits & conflict_mask[v]) == 0:
                        S.add(v)
                        for y in S:
                            if y != v:
                                new_sum_bits |= (1 << (v + y))
                        new_sum_bits |= (1 << (2 * v))
                        added_one = v
                        break
                
                if added_one is not None:
                    # Try adding a second element
                    for w in not_in_S:
                        if w == added_one:
                            continue
                        if (new_sum_bits & conflict_mask[w]) == 0:
                            S.add(w)
                            for y in S:
                                if y != w:
                                    new_sum_bits |= (1 << (w + y))
                            new_sum_bits |= (1 << (2 * w))
                            sum_bits = new_sum_bits
                            improved = True
                            break
                    
                    if not improved:
                        # Only got one, still a net gain? No, we removed u and added one, net zero
                        # Undo the addition
                        S.remove(added_one)
                        S.add(u)
                else:
                    S.add(u)
                
                if improved:
                    break
    
    # Final verification
    if time.time() < deadline - 2:
        S_sorted = sorted(S)
        L = len(S_sorted)
        valid_final = True
        for i in range(L):
            a = S_sorted[i]
            for j in range(i + 1, L):
                b = S_sorted[j]
                for k in range(j, L):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        valid_final = False
                        break
                if not valid_final:
                    break
            if not valid_final:
                break
        
        if not valid_final:
            return list(range(171, 257))

    return list(S)

# Test harness (not included in submission, just for local verification)
if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
