import time
import random

def construct():
    start = time.time()
    deadline = start + 82.0
    N = 128
    rng = random.Random(987654321)

    def time_left():
        return deadline - time.time()

    # Precompute: for each x, the set of pairs (a,b) in 1..N such that
    # a+b, a+x, b+x are all in 1..N. If a,b,x are all in S, that's a forbidden triple.
    # Also precompute for each x as a "sum": pairs (a,b) with a+b=x.
    
    # conflict_mask[x] = bitmask of elements y such that having both x and y in S
    # could be part of a forbidden triple (i.e., there exists z in S where {x,y,z} is forbidden)
    # Actually, let's precompute: for each x, which pairs (a,b) would create a forbidden triple with x.
    
    # For a forbidden triple {a,b,c} (distinct), we need a+b, a+c, b+c all in S.
    # If x is in S, it can play the role of a, b, c, a+b, a+c, or b+c.
    
    # Let's precompute for each x: the set of "witness pairs" that would make adding x invalid.
    # If x is a member: need a,b in S (distinct from x) such that x+a, x+b, a+b all in S.
    # If x is a sum: x=a+b for distinct a,b in S, need c in S (c!=a,b) with a+c, b+c in S.
    
    # For speed, precompute for each x:
    #   member_conflicts[x] = list of (a, b) pairs where a,b < x or some ordering, such that
    #     if a,b,x are all in S, then x+a, x+b, a+b must also be in S (creating forbidden triple)
    #   Actually the forbidden condition is: a+b, a+c, b+c all in S where {a,b,c} are distinct members.
    #   So if x is a member, for each pair (a,b) of OTHER members, if x+a in S and x+b in S and a+b in S, forbidden.
    
    # Let me restructure: precompute all forbidden triples.
    # A forbidden triple is a set {a,b,c} of 3 distinct integers in 1..128 such that
    # a+b, a+c, b+c are all in 1..128. (The sums don't need to be in S, but if they are, it's forbidden.)
    
    # Actually the constraint is: S must NOT contain 3 distinct elements a,b,c AND also contain a+b, a+c, b+c.
    # So the "hyperedge" is {a,b,c,a+b,a+c,b+c} (some may coincide but a,b,c are distinct).
    # We need S to not contain all 6 (or fewer if some coincide) elements of any such hyperedge.
    
    # Precompute all minimal forbidden sets: for each triple (a,b,c) distinct in 1..128,
    # if a+b<=128, a+c<=128, b+c<=128, then the set {a,b,c,a+b,a+c,b+c} is forbidden.
    # S is valid iff S doesn't contain any such forbidden set.
    
    # For the search, we need: given current S, can we add x?
    # Adding x is bad iff there exists a forbidden set F such that F\{x} subset of S and x not in S.
    
    # Precompute: for each x, the list of "minimal forbidden sets containing x" minus x.
    # i.e., for each x, forbidden_sets_with_x[x] = list of sets T where T union {x} is a forbidden set.
    # Then x can be added iff no T in forbidden_sets_with_x[x] is a subset of current S.
    
    # But the number of such sets could be large. Let's think about the structure.
    
    # Forbidden set from triple (a,b,c): F = {a,b,c,a+b,a+c,b+c}
    # x can be any of the 6 elements (with possible coincidences).
    
    # Let me just precompute for each x, the list of "required other elements" sets.
    # For efficiency, represent each required set as a bitmask.
    
    # Actually, let's use a different approach. Precompute for each x:
    #   For x to be a "member" in a forbidden triple: need a,b in S, a!=b, a!=x, b!=x,
    #     such that (x+a) in S, (x+b) in S, (a+b) in S.
    #     Equivalently: for some a in S\{x}, b in S\{x}, a!=b:
    #       (x+a)<=N and (x+a) in S, (x+b)<=N and (x+b) in S, (a+b)<=N and (a+b) in S.
    #     Precompute pairs_a[x] = list of (a, x+a, a) such that if a and x are in S and x+a is in S...
    #     This is getting complex. Let me just use the bitmask approach for the B&B polish.
    
    # For the main search, keep the current can_add but make it faster with precomputed structures.
    
    # Precompute: for each x in 1..128, the list of (a, b) pairs (a < b, both != x) such that
    # x+a <= 128, x+b <= 128, a+b <= 128.
    # If x, a, b, x+a, x+b, a+b are all in S, forbidden.
    
    # And for x as a sum: x = a+b, a < b. Then need c != a, c != b in S such that
    # a+c in S (or a+c==x) and b+c in S (or b+c==x).
    
    # Let me precompute these lists.
    
    # member_triples[x] = list of (a, b, x+a, x+b, a+b) for a < b, a != x, b != x,
    #   x+a <= N, x+b <= N, a+b <= N
    # sum_triples[x] = list of (a, b, c) where a + b = x, a < b, c != a, c != b
    #   (then check if a+c, b+c in S or equal to x)
    
    member_triples = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                xa = x + a
                xb = x + b
                ab = a + b
                if xa <= N and xb <= N and ab <= N:
                    member_triples[x].append((a, b, xa, xb, ab))
    
    # For x as a sum: x = a + b
    sum_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s <= N:
                sum_pairs[s].append((a, b))
    
    # For sum case: x = a+b, need c in S, c != a, c != b, with a+c in S or a+c==x, b+c in S or b+c==x
    # Precompute for each x and each (a,b) with a+b=x, the list of c values to check.
    # But c ranges over 1..N, so just check all c in S.
    
    def can_add(mem, inS, x):
        # Check if adding x to S creates a forbidden triple.
        # Case 1: x is a member. Check all precomputed member_triples[x].
        for (a, b, xa, xb, ab) in member_triples[x]:
            if not inS[a] or not inS[b]:
                continue
            # Need xa in S, xb in S, ab in S (or ab == x)
            if (inS[xa] or xa == x) and (inS[xb] or xb == x) and (inS[ab] or ab == x):
                return False
        # Case 2: x is a sum. For each (a,b) with a+b=x, both in S,
        # check if there's c in S, c != a, c != b, with a+c in S or ==x, b+c in S or ==x.
        for (a, b) in sum_pairs[x]:
            if not inS[a] or not inS[b]:
                continue
            for c in mem:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                ac_ok = (ac <= N and inS[ac]) or ac == x
                bc_ok = (bc <= N and inS[bc]) or bc == x
                if ac_ok and bc_ok:
                    return False
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    def local_search(mem, inS):
        for _ in range(80):
            if time_left() < 4.0:
                break
            added = False
            for x in range(1, N + 1):
                if time_left() < 4.0:
                    break
                if inS[x]:
                    continue
                if can_add(mem, inS, x):
                    mem.append(x)
                    inS[x] = 1
                    added = True
            if added:
                continue
            # Try swaps
            sw = False
            for _ in range(80):
                if time_left() < 4.0:
                    break
                rm = rng.choice(mem)
                add = rng.randint(1, N)
                if inS[add]:
                    continue
                inS[rm] = 0
                mem.remove(rm)
                if can_add(mem, inS, add):
                    mem.append(add)
                    inS[add] = 1
                    sw = True
                    break
                else:
                    mem.append(rm)
                    inS[rm] = 1
            if not sw:
                break
        return mem

    def solve_restart(order):
        mem = [i for i in range(N // 2 + 1, N + 1)]
        inS = bytearray(N + 1)
        for i in mem:
            inS[i] = 1
        for x in order:
            if time_left() < 4.0:
                break
            if inS[x]:
                continue
            if can_add(mem, inS, x):
                mem.append(x)
                inS[x] = 1
        local_search(mem, inS)
        return mem

    base = [i for i in range(N // 2 + 1, N + 1)]
    best = base
    best_size = len(base)

    small = list(range(1, N // 2 + 1))
    orders = [list(small), list(reversed(small))]
    for _ in range(20):
        o = list(small)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time_left() < 8.0:
            break
        mem = solve_restart(o)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    # Final polish: try to improve best via targeted swaps
    if time_left() > 10.0:
        inS = bytearray(N + 1)
        for i in best:
            inS[i] = 1
        mem = list(best)
        local_search(mem, inS)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    # Branch-and-bound polish: try removing one element and see if we can add two
    if time_left() > 15.0:
        inS = bytearray(N + 1)
        for i in best:
            inS[i] = 1
        mem = list(best)
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            # Try 1-for-2 swaps: remove one, try to add two
            for i in range(len(mem)):
                if time_left() < 5.0:
                    break
                rm = mem[i]
                inS[rm] = 0
                mem.remove(rm)
                added_count = 0
                added_elems = []
                for x in range(1, N + 1):
                    if time_left() < 5.0:
                        break
                    if inS[x]:
                        continue
                    if can_add(mem, inS, x):
                        mem.append(x)
                        inS[x] = 1
                        added_count += 1
                        added_elems.append(x)
                if added_count >= 2:
                    improved = True
                    # Continue local search
                    local_search(mem, inS)
                    if len(mem) > best_size and full_valid(mem):
                        best = list(mem)
                        best_size = len(mem)
                else:
                    # Revert
                    for x in added_elems:
                        inS[x] = 0
                        mem.remove(x)
                    mem.append(rm)
                    inS[rm] = 1
            # Also try direct additions
            for x in range(1, N + 1):
                if time_left() < 5.0:
                    break
                if inS[x]:
                    continue
                if can_add(mem, inS, x):
                    mem.append(x)
                    inS[x] = 1
                    improved = True
            if improved:
                local_search(mem, inS)
                if len(mem) > best_size and full_valid(mem):
                    best = list(mem)
                    best_size = len(mem)

    if not full_valid(best):
        best = base

    return [int(x) for x in best]
