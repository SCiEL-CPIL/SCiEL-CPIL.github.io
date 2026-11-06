"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0  # stay safely under 30 s

    def sums_set(S):
        T = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            T.add(2 * a)
            for b in L[i + 1:]:
                T.add(a + b)
        return T

    def is_sidon(S):
        # distinct pair sums including diagonals
        seen = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    def addable_elements(S, T):
        # return list of x in [1,N]\S such that S U {x} is Sidon
        res = []
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if 2 * x in T:
                continue
            ok = True
            for s in S:
                if x + s in T:
                    ok = False
                    break
            if ok:
                res.append(x)
        return res

    def new_addable_after_add(S, T, x, old_addable):
        # given old_addable (list of addable missing elements before adding x),
        # return the subset that remains addable after adding x.
        # new sums introduced by x: 2x and x+s for s in S
        new_sums = {2 * x}
        for s in S:
            new_sums.add(x + s)
        rem = []
        Sset = set(S) | {x}
        T2 = T | new_sums
        for y in old_addable:
            if y == x:
                continue
            if 2 * y in T2:
                continue
            ok = True
            for s in Sset:
                if y + s in T2:
                    ok = False
                    break
            if ok:
                rem.append(y)
        return rem

    # Greedy maximality-driven construction:
    # repeatedly add the missing element that removes the most currently-addable elements.
    S = []
    T = set()
    addable = addable_elements(S, T)  # all of 1..N initially
    while addable and time.time() < deadline:
        best_x = None
        best_rem = None
        best_len = len(addable) + 1
        for x in addable:
            rem = new_addable_after_add(S, T, x, addable)
            if len(rem) < best_len:
                best_len = len(rem)
                best_x = x
                best_rem = rem
                if best_len == 0:
                    break
        if best_x is None:
            break
        S.append(best_x)
        T = T | {2 * best_x} | {best_x + s for s in S[:-1]}
        addable = best_rem if best_rem is not None else addable_elements(S, T)

    # Verify Sidon; if somehow broken (shouldn't be), fall back to a known-good greedy.
    if not is_sidon(S):
        S = []
        T = set()
        for x in range(1, N + 1):
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not T.intersection(added):
                S.append(x)
                T.update(added)
    return S
# EVOLVE-BLOCK-END
