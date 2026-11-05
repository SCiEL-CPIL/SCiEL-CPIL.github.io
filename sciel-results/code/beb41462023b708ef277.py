def construct():
    import time
    import random
    from collections import defaultdict

    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit

    N = 512

    # Precompute all 4-APs: list of (a, b, c, d) where a, a+d, a+2d, a+3d are in [1, N]
    # We'll store for each element, the list of APs it participates in
    # Also, for fast conflict checking, we'll use a bitset representation

    # Let's use a bitset for the set S.
    # N=512, so we can use 8 integers of 64 bits, or just use Python's arbitrary precision integers.
    # Python integers are efficient for bit operations.

    # Precompute APs
    # For each AP (a, b, c, d), we need to know if adding a new element x creates a conflict.
    # Actually, it's easier to precompute for each pair (x, y) in S, the other two elements of the AP.
    # But that's O(N^2) memory and computation.

    # Alternative: Precompute for each x, the list of APs containing x.
    # When adding x, check if any AP containing x has the other 3 elements in S.
    # An AP is (a, a+d, a+2d, a+3d).
    # If x is the 1st term: a=x, need x+d, x+2d, x+3d in S.
    # If x is the 2nd term: a+d=x -> a=x-d, need x-d, x+d, x+2d in S.
    # If x is the 3rd term: a+2d=x -> a=x-2d, need x-2d, x-d, x+d in S.
    # If x is the 4th term: a+3d=x -> a=x-3d, need x-3d, x-2d, x-d in S.

    # This is what the original code did. Let's optimize it.
    # We can precompute for each x, the list of (other1, other2, other3) tuples.
    # Actually, let's just use the bitset and check bits.

    # Bitset S: S is an integer where bit i is 1 if (i+1) is in S.
    # So element 1 is bit 0, element 512 is bit 511.

    def can_add(S_bits, x):
        """Check if x can be added to S (represented as bits) without forming a 4-AP."""
        # x is 1-indexed. Bit index is x-1.
        idx = x - 1
        if (S_bits >> idx) & 1:
            return False  # Already in S

        # Check all d
        # Case 1: x is 4th term (a+3d=x). a = x-3d.
        # Need a, a+d, a+2d in S. i.e., x-3d, x-2d, x-d in S.
        for d in range(1, 171):
            a = x - 3*d
            if a < 1: break
            # Check bits for a, a+d, a+2d
            if ((S_bits >> (a-1)) & 1) and ((S_bits >> (a+d-1)) & 1) and ((S_bits >> (a+2*d-1)) & 1):
                return False

        # Case 2: x is 3rd term (a+2d=x). a = x-2d.
        # Need a, a+d, a+3d in S. i.e., x-2d, x-d, x+d in S.
        for d in range(1, 171):
            a = x - 2*d
            b = x + d
            if a < 1 or b > N: continue
            if ((S_bits >> (a-1)) & 1) and ((S_bits >> (x-1-1)) & 1) and ((S_bits >> (b-1)) & 1):
                return False
            # Wait, x-d is the middle one. x is a+2d.
            # Terms: a, a+d, a+2d (x), a+3d (b).
            # Need a, a+d, a+3d in S.
            # a = x-2d, a+d = x-d, a+3d = x+d.
            # So check x-2d, x-d, x+d.

        # Case 3: x is 2nd term (a+d=x). a = x-d.
        # Need a, a+2d, a+3d in S. i.e., x-d, x+d, x+2d in S.
        for d in range(1, 171):
            a = x - d
            c = x + d
            b = x + 2*d
            if a < 1 or b > N: continue
            if ((S_bits >> (a-1)) & 1) and ((S_bits >> (c-1)) & 1) and ((S_bits >> (b-1)) & 1):
                return False

        # Case 4: x is 1st term (a=x).
        # Need a+d, a+2d, a+3d in S. i.e., x+d, x+2d, x+3d in S.
        for d in range(1, 171):
            b = x + d
            c = x + 2*d
            e = x + 3*d
            if e > N: break
            if ((S_bits >> (b-1)) & 1) and ((S_bits >> (c-1)) & 1) and ((S_bits >> (e-1)) & 1):
                return False

        return True

    # Simulated Annealing
    # State: S_bits
    # Move: Remove one element, Add one element.
    # Or: Add one element (if valid).
    # Or: Remove one element, Add two elements (if valid).

    # Let's start with a greedy construction to get a good initial state.
    # Greedy: Pick elements in random order, add if valid.

    def greedy_construct(seed):
        random.seed(seed)
        S_bits = 0
        count = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S_bits, x):
                S_bits |= (1 << (x-1))
                count += 1
        return S_bits, count

    # Find best greedy
    best_S_bits = 0
    best_count = 0
    for i in range(50):
        if time.time() > deadline: break
        S_bits, count = greedy_construct(i)
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Simulated Annealing
    # Temperature T
    # Energy = -count (we want to maximize count)
    # Or Energy = N - count? No, just use count.
    # Delta E = new_count - old_count.
    # If Delta E > 0, accept.
    # If Delta E < 0, accept with prob exp(Delta E / T).

    # Moves:
    # 1. Try to add a random x. If valid, accept (Delta E = +1).
    # 2. Try to remove a random x from S, then add a random y.
    #    If valid, Delta E = 0 (if we just swap) or +1 (if we add without removing? No, remove then add is swap).
    #    Actually, remove x, add y. Size stays same. Delta E = 0.
    #    But we might enable adding more later.
    # 3. Try to remove x, add y, add z. Size +1. Delta E = +1.

    # Let's implement a simple SA.
    # To speed up, we can precompute conflicts?
    # can_add is O(N) roughly.
    # SA steps: ~100k steps?
    # 100k * O(N) = 50M ops. Might be slow in Python.

    # Let's optimize can_add.
    # Instead of checking all d, we can precompute for each x, the list of conflicting triples.
    # But the state S changes.

    # Alternative: Use a list of APs.
    # For each AP, keep a count of how many of its 4 elements are in S.
    # If count becomes 4, it's a conflict.
    # When adding x, for each AP containing x, increment count. If any count becomes 4, conflict.
    # When removing x, for each AP containing x, decrement count.

    # Let's precompute APs for each element.
    ap_list = [[] for _ in range(N+1)]
    # Each AP is (a, b, c, d)
    # We need to store the AP index or just the other elements.
    # Let's store the other 3 elements for each position in the AP.
    # For element x in AP (a, b, c, d):
    # If x==a, others are b,c,d.
    # If x==b, others are a,c,d.
    # etc.

    # Actually, storing the AP tuple is fine.
    # ap_list[x] contains list of (other1, other2, other3)

    for a in range(1, N+1):
        for d in range(1, 171):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N: break
            # AP: a, b, c, e
            ap_list[a].append((b, c, e))
            ap_list[b].append((a, c, e))
            ap_list[c].append((a, b, e))
            ap_list[e].append((a, b, c))

    # Now can_add can be faster?
    # Still need to check if the other 3 are in S.
    # With bitset, it's 3 bit checks per AP.
    # Number of APs containing x is roughly O(N).
    # So can_add is O(N) bit checks.
    # This is faster than the loop over d with multiple checks.

    def can_add_fast(S_bits, x):
        idx = x - 1
        if (S_bits >> idx) & 1:
            return False
        for (o1, o2, o3) in ap_list[x]:
            if ((S_bits >> (o1-1)) & 1) and ((S_bits >> (o2-1)) & 1) and ((S_bits >> (o3-1)) & 1):
                return False
        return True

    # Let's re-run greedy with fast can_add
    best_S_bits = 0
    best_count = 0
    for i in range(100):
        if time.time() > deadline: break
        random.seed(i)
        S_bits = 0
        count = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add_fast(S_bits, x):
                S_bits |= (1 << (x-1))
                count += 1
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Simulated Annealing
    S_bits = best_S_bits
    count = best_count

    # We need to track which elements are in S for quick removal.
    # We can just use the bitset.
    # To pick a random element to remove, we need to find set bits.
    # We can maintain a list of elements in S.

    S_list = [i+1 for i in range(N) if (best_S_bits >> i) & 1]

    T = 5.0
    cooling = 0.999
    min_T = 0.1

    # Precompute list of all elements for adding
    all_elements = list(range(1, N+1))

    # SA Loop
    steps = 0
    while T > min_T and time.time() < deadline - 2.0:
        # Pick a move
        # 70% swap, 30% add
        if random.random() < 0.7:
            # Swap: remove x, add y
            if not S_list:
                continue
            x = random.choice(S_list)
            y = random.choice(all_elements)
            if x == y:
                continue

            # Remove x
            S_bits ^= (1 << (x-1))
            S_list.remove(x)
            count -= 1

            # Try to add y
            if can_add_fast(S_bits, y):
                S_bits |= (1 << (y-1))
                S_list.append(y)
                count += 1
                delta = 0 # Net change 0
                # Accept (always, or with prob 1)
                # Actually, if we failed to add y, we should revert?
                # In SA, we evaluate the new state.
                # New state: S - {x} + {y} (if valid) or S - {x} (if invalid).
                # If invalid, we revert to S.
                # Let's implement properly.

                # Re-do logic:
                # 1. Remove x. State S'.
                # 2. Try add y to S'.
                # 3. If valid, New State S''. Count' = Count. Delta = 0.
                # 4. If invalid, New State S'. Count' = Count - 1. Delta = -1.
                # 5. Accept New State with prob exp(Delta/T) if Delta < 0, else 1.
                # 6. If rejected, revert to S.

                # Wait, if I remove x and can't add y, I have a smaller set.
                # I should probably only consider swaps where I successfully add y?
                # Or allow shrinking to escape?
                # Shrinking is bad. Let's only accept if we maintain or increase size.
                # So if can_add_fast fails, revert x.
            else:
                # Revert x
                S_bits |= (1 << (x-1))
                S_list.append(x)
                count += 1
                continue

            # If we are here, we successfully swapped.
            # Delta = 0. Always accept.
            # But we want to explore.
            # Let's allow some random walks.
            # Actually, if Delta=0, accept with prob 1.
            # This might get stuck.
            # Let's try "remove x, add y, add z" move.

        else:
            # Add move: try to add a random y
            y = random.choice(all_elements)
            if can_add_fast(S_bits, y):
                S_bits |= (1 << (y-1))
                S_list.append(y)
                count += 1
                # Accept (Delta = +1)
            else:
                # Rejected (or just try another)
                pass

        # Update T
        T *= cooling
        steps += 1

        # Update best
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Final local search: try to add any valid element
    for x in all_elements:
        if time.time() > deadline - 1.0: break
        if can_add_fast(best_S_bits, x):
            best_S_bits |= (1 << (x-1))
            best_count += 1

    # Convert to list
    result = [i+1 for i in range(N) if (best_S_bits >> i) & 1]
    return sorted(result)
