def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # Precompute conflict masks: for each a, which pairs (b,c) with b!=c, b,c != a
    # satisfy a | b*c. We'll use this for fast inclusion checks.
    # But more useful: for a given set S, check if adding x creates any triple conflict.
    # A triple (x, b, c) conflicts if x|b*c or b|x*c or c|x*b.

    # Precompute for each x, a bitmask (or set) of numbers y such that there exists
    # some z in the full range where x|y*z or y|x*z or z|x*y.
    # Actually, let's precompute: for each pair (i,j), which a's conflict.
    # conflict[a] = set of (b,c) pairs where a|b*c, b<c, b!=a, c!=a

    # Better approach for speed: precompute a boolean matrix
    # div[a][b][c] is too big. Instead:
    # For each a, precompute which b's have the property that there exists c
    # such that a|b*c. But c must also be in S.

    # Let's use a different strategy: precompute for each a, the set of (b,c)
    # where a divides b*c. Then when checking if we can add x to set S:
    # - Check if any (b,c) in S×S (b≠c) has x|b*c
    # - Check if any b in S has b|x*c for some c in S (c≠b, c≠x)
    # - Check if any c in S has c|x*b for some b in S (b≠c, b≠x)

    # For speed, precompute for each a, a bitmask of "conflicting partners"
    # Actually, let's just precompute a 2D table:
    # For each pair (b,c), compute the set of a's that divide b*c.
    # Then when checking if x can be added: x conflicts if there exist b,c in S
    # (b≠c, b,c≠x) such that x is in divisors_of[b*c].

    # Precompute: for each product value p = b*c (b,c in 2..256, b<c),
    # the set of a in 2..256 that divide p.
    # Then for checking: x can be added to S iff for all pairs (b,c) in S with b≠c,
    # x does NOT divide b*c.

    # Let's precompute a lookup: divs[p] = set of a's dividing p, for p up to 256*255
    max_prod = 256 * 255
    divs = [set() for _ in range(max_prod + 1)]
    for a in range(2, N + 1):
        for multiple in range(a, max_prod + 1, a):
            divs[multiple].add(a)

    # Greedy: add from largest to smallest
    included = set()
    for a in sorted(numbers, reverse=True):
        # Check if adding a creates any conflict
        # Need: for all b,c in included, b≠c: a ∤ b*c
        conflict = False
        inc_list = list(included)
        for i in range(len(inc_list)):
            b = inc_list[i]
            for j in range(i + 1, len(inc_list)):
                c = inc_list[j]
                if a in divs[b * c]:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            included.add(a)

    # Local search: try 1-swap moves (remove one, add one) to improve cardinality
    # or maintain cardinality while enabling further additions
    best_set = set(included)
    best_size = len(best_set)

    def is_valid_set(s):
        """Check if set s has no triple conflict."""
        sl = list(s)
        for i in range(len(sl)):
            a = sl[i]
            for j in range(i + 1, len(sl)):
                b = sl[j]
                for k in range(j + 1, len(sl)):
                    c = sl[k]
                    # Check a|b*c, b|a*c, c|a*b
                    if a in divs[b * c] or b in divs[a * c] or c in divs[a * b]:
                        return False
        return True

    # Try 1-swap: remove x from S, try adding y not in S
    # This can increase size if we remove 1 and add 2 (net +1), but that's a 2-add 1-remove
    # For 1-swap (remove 1, add 1): maintains size but may enable future additions
    # Let's try: for each x in S, for each y not in S:
    #   new_S = S - {x} + {y}
    #   if valid, accept (same size, different composition)
    #   then try to add more elements greedily

    # Better: iterative improvement
    # 1. Try to add any missing element
    # 2. If stuck, try swaps to unlock additions

    improved = True
    while improved and (time.time() - start) < DEADLINE - 5.0:
        improved = False

        # Phase 1: try to add any element not in best_set
        for y in sorted(numbers, reverse=True):
            if y in best_set:
                continue
            # Check if y can be added to best_set
            conflict = False
            bl = list(best_set)
            for i in range(len(bl)):
                b = bl[i]
                for j in range(i + 1, len(bl)):
                    c = bl[j]
                    if y in divs[b * c]:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                best_set.add(y)
                improved = True
                break  # restart phase 1 with updated set

        if (time.time() - start) > DEADLINE - 5.0:
            break

        # Phase 2: try 1-swap moves to enable future additions
        if not improved:
            # For each pair (x in best_set, y not in best_set), try swap
            # Only accept if the resulting set is valid
            # We want to find a swap that allows at least one additional element
            swapped = False
            candidates = sorted(best_set)
            for x in candidates:
                if (time.time() - start) > DEADLINE - 5.0:
                    break
                for y in sorted(numbers, reverse=True):
                    if y in best_set:
                        continue
                    # Try swap: remove x, add y
                    test_set = best_set - {x} | {y}
                    # Quick validity check
                    valid = True
                    tl = list(test_set)
                    for i in range(len(tl)):
                        a = tl[i]
                        for j in range(i + 1, len(tl)):
                            b = tl[j]
                            for k in range(j + 1, len(tl)):
                                c = tl[k]
                                if a in divs[b * c] or b in divs[a * c] or c in divs[a * b]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        # Check if this swap enables adding at least one more element
                        can_add = False
                        for z in sorted(numbers, reverse=True):
                            if z in test_set:
                                continue
                            # Check if z can be added to test_set
                            z_conflict = False
                            tl2 = list(test_set)
                            for i in range(len(tl2)):
                                b = tl2[i]
                                for j in range(i + 1, len(tl2)):
                                    c = tl2[j]
                                    if z in divs[b * c]:
                                        z_conflict = True
                                        break
                                if z_conflict:
                                    break
                            if not z_conflict:
                                can_add = True
                                break
                        if can_add:
                            best_set = test_set
                            improved = True
                            swapped = True
                            break
                if swapped:
                    break

    # Final validation
    if not is_valid_set(best_set):
        # Fallback to greedy-only result
        included_final = set()
        for a in sorted(numbers, reverse=True):
            conflict = False
            inc_list = list(included_final)
            for i in range(len(inc_list)):
                b = inc_list[i]
                for j in range(i + 1, len(inc_list)):
                    c = inc_list[j]
                    if a in divs[b * c]:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                included_final.add(a)
        best_set = included_final

    return sorted(best_set)
