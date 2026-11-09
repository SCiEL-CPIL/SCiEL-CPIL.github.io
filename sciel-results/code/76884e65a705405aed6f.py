"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 128

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # Structural insight: any three elements all >= 65 have pairwise sums > 128,
    # so they can never form a forbidden triple (a+b, a+c, b+c all in set).
    # Thus we can safely include all of {65, 66, ..., 128} (64 elements).
    base = list(range(65, N + 1))
    base_set = set(base)

    # Now we try to add small elements from {1..64} greedily, checking conflicts.
    # A forbidden triple is three DISTINCT members a,b,c such that a+b, a+c, b+c are ALL in the set.
    # Since base elements are >=65, if we add small elements, the sums of two small elements
    # could be in base (if sum >= 65) or in the small set.
    # We need to ensure no triple (a,b,c) from our final set has all three pair-sums in the set.

    # Strategy: build the set incrementally. Start with base.
    # For each candidate small element s (from 1 to 64), check if adding s creates a forbidden triple.
    # A forbidden triple involving s could be:
    #   - (s, x, y) where x,y are already in the set, and s+x, s+y, x+y are all in the set.
    #   - (s, x, y) where one of x,y is s? No, distinct members.
    #   - Actually, the triple must consist of three DISTINCT members. So s, x, y all distinct.
    #   - The pair sums s+x, s+y, x+y must all be in the set.

    # Let's maintain the current set and check conflicts when adding a new element.
    current = set(base)

    # For efficiency, we can precompute or check on the fly.
    # When adding s, we need to check all pairs (x, y) in current where x != y and x, y != s.
    # But that's O(n^2) per addition, and n can be up to ~128, so 128^2 * 64 = ~1M checks, fine.

    # Actually, let's be smarter. For a new element s, a forbidden triple (s, x, y) requires:
    #   x, y in current, x != y, x != s, y != s
    #   s+x in current, s+y in current, x+y in current
    # Since current already contains all of base, s+x is in current iff s+x >= 65 (since s+x <= 64+64=128, and if s+x >= 65 it's in base).
    # Similarly s+y in current iff s+y >= 65.
    # x+y in current: x+y could be in base (if x+y >= 65) or in the small part we've added.

    # Let's just do the direct check.
    small_added = []
    for s in range(1, 65):
        if time.time() > deadline:
            break
        # Check if adding s creates any forbidden triple
        conflict = False
        current_list = list(current)
        for i in range(len(current_list)):
            x = current_list[i]
            if x == s:
                continue
            for j in range(i + 1, len(current_list)):
                y = current_list[j]
                if y == s:
                    continue
                # Check if (s, x, y) forms a forbidden triple
                if s + x in current and s + y in current and x + y in current:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            current.add(s)
            small_added.append(s)

    result = list(current)
    # Ensure Python int types
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
