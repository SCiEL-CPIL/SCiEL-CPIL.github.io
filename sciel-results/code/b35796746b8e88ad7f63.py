"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random

N = 128

def construct():
    # We want a large subset S of {1..128} such that there do NOT exist three
    # distinct a,b,c in S with a+b, a+c, b+c all in S.
    #
    # Greedy independent set on the 3-uniform hypergraph:
    #   hyperedge = {a,b,c} iff a+b, a+c, b+c are all in [1,128] (domain).
    #   Wait: the constraint is about the PAIR SUMS being selected, not just in domain.
    #   So the "forbidden triple" depends on S itself. We handle it dynamically.
    #
    # When adding x to current S: check all pairs (a,b) in S (a<b):
    #   triple {x,a,b} is forbidden iff (x+a in S or x+a==x) and (x+b in S or x+b==x)
    #   and (a+b in S or a+b==x). Since a,b>=1, x+a>x so x+a!=x; need x+a in S.
    #   Similarly x+b in S. And a+b in S or a+b==x.
    #
    # Bit-parallel: represent S as a bitmask. For each x, precompute nothing heavy;
    # just test pairs. To speed up, iterate over a in S, b in S with a<b.
    #
    # Use randomized ordering, try many restarts, keep best.

    best = None
    best_len = 0

    rng = random.Random(12345)

    # Precompute for each x the list of (a,b) pairs with a<b, a,b in 1..128,
    # such that x+a<=128, x+b<=128, a+b<=128. These are the only pairs that
    # could form a forbidden triple with x (since pair sums must be in domain to be in S).
    # Actually pair sums must be <=128 to possibly be in S.
    # For a given x, dangerous pairs (a,b): a<b, a+b<=128, x+a<=128, x+b<=128.
    # We'll just iterate; 128 is small.

    def can_add(x, S_mask, S_set):
        # S_set is a Python set of currently selected elements (for fast membership)
        # S_mask is bitmask for potential bit-parallel use
        # Check all pairs (a,b) in S, a<b
        elems = list(S_set)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            xa = x + a
            if xa > 128:
                continue
            if xa not in S_set:
                continue
            for j in range(i+1, n):
                b = elems[j]
                xb = x + b
                if xb > 128:
                    break  # elems sorted? not guaranteed. Don't break.
                if xb not in S_set:
                    continue
                ab = a + b
                if ab == x or ab in S_set:
                    return False
        return True

    # To make it faster, keep S_set as a set and also a sorted list.
    # Number of restarts and ordering strategies.

    for trial in range(2000):
        # Random permutation of 1..128
        perm = list(range(1, 129))
        rng.shuffle(perm)

        S_set = set()
        S_mask = 0

        for x in perm:
            if can_add(x, S_mask, S_set):
                S_set.add(x)
                S_mask |= (1 << x)

        if len(S_set) > best_len:
            best_len = len(S_set)
            best = list(S_set)

        if best_len >= 90:  # early stop if we get very large
            break

    # Verify best
    def is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True

    if best is None:
        best = list(range(65, 129))

    if not is_valid(best):
        # Fallback to safe construction: top half
        best = list(range(65, 129))

    return list(best)
# EVOLVE-BLOCK-END
