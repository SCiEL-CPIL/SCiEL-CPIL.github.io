"""Covering 1..L by prime residue classes with reciprocal budget 1 (CSP backtracking)."""
import time
import random
from fractions import Fraction

MAXL = 100000
DEADLINE = 132.0  # safety margin before the 150s kill


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    t0 = time.time()
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # ---- Fast incremental coverage via per-position count array ----
    # cnt[x] = number of classes covering x; covered[x] = cnt[x] > 0.
    # add/remove/swap are O(p) each.

    def add_class(cnt, covered, p, r):
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            if cnt[x] == 0:
                covered[x] = 1
            cnt[x] += 1

    def remove_class(cnt, covered, p, r):
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            cnt[x] -= 1
            if cnt[x] == 0:
                covered[x] = 0

    def swap_residue(cnt, covered, p, old_r, new_r):
        remove_class(cnt, covered, p, old_r)
        add_class(cnt, covered, p, new_r)

    def compute_L(covered):
        L = 0
        for i in range(1, MAXL + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def first_gap(covered, start=1):
        for i in range(start, MAXL + 1):
            if not covered[i]:
                return i
        return None

    # ---- CSP-style backtracking with forward checking ----
    # Goal: extend L as far as possible by assigning residues to primes.
    # Variable ordering: pick the prime that best covers the current first gap.
    # Value ordering: prefer residues that cover the gap, then extend further.

    def csp_search(state, budget, cnt, covered, L, depth=0, max_depth=40):
        """Backtracking search to extend coverage. Returns (state, budget, L)."""
        if depth >= max_depth or time.time() > DEADLINE - 5:
            return state, budget, L

        gap = first_gap(covered, L + 1)
        if gap is None:
            return state, budget, L

        # Forward checking: find primes that CAN cover the gap
        # A prime p can cover gap if there exists r such that gap ≡ r (mod p),
        # i.e., r = gap % p. The constraint is just budget.
        candidates = []
        for p in primes:
            if p in state:
                # Can we change residue of p to cover gap?
                if state[p] != gap % p:
                    candidates.append((p, gap % p, 'change'))
            else:
                # Can we add p?
                if budget >= recip[p]:
                    candidates.append((p, gap % p, 'add'))

        if not candidates:
            return state, budget, L

        # Sort by prime size (smaller primes cover more densely)
        candidates.sort(key=lambda x: x[0])

        best_state = state
        best_budget = budget
        best_L = L

        # Try a limited number of candidates (forward checking prunes the rest)
        for p, r, action in candidates[:8]:
            if time.time() > DEADLINE - 5:
                break

            old_L = L
            if action == 'add':
                add_class(cnt, covered, p, r)
                state[p] = r
                budget -= recip[p]
            else:  # change
                old_r = state[p]
                swap_residue(cnt, covered, p, old_r, r)
                state[p] = r

            # Compute new L
            newL = 0
            for i in range(1, MAXL + 1):
                if covered[i]:
                    newL = i
                else:
                    break

            if newL > best_L:
                best_state = dict(state)
                best_budget = budget
                best_L = newL
                # Recurse deeper
                s2, b2, L2 = csp_search(state, budget, cnt, covered, newL, depth + 1, max_depth)
                if L2 > best_L:
                    best_state = s2
                    best_budget = b2
                    best_L = L2
                    # Restore state from best
                    state.clear()
                    state.update(best_state)
                    budget = best_budget
                    # Rebuild cnt and covered from best_state
                    cnt[:] = [0] * (MAXL + 1)
                    covered[:] = bytearray(MAXL + 1)
                    for pp, rr in best_state.items():
                        add_class(cnt, covered, pp, rr)
                    L = best_L
                    break

            # Revert
            if action == 'add':
                remove_class(cnt, covered, p, r)
                del state[p]
                budget += recip[p]
            else:
                swap_residue(cnt, covered, p, r, old_r)
                state[p] = old_r

            L = old_L

        return best_state, best_budget, best_L

    # ---- Greedy initialization to get a good starting point ----
    def greedy_init(seed):
        rng = random.Random(seed)
        state = {}
        budget = Fraction(1, 1)
        cnt = [0] * (MAXL + 1)
        covered = bytearray(MAXL + 1)
        L = 0
        remaining = primes[:]
        rng.shuffle(remaining)

        for _ in range(len(primes)):
            if time.time() > DEADLINE - 30:
                break
            best_gain = 0
            best_p = best_r = None
            best_newL = 0

            for p in remaining:
                if p in state:
                    continue
                if budget < recip[p]:
                    continue
                # Try residue covering L+1
                r = (L + 1) % p
                add_class(cnt, covered, p, r)
                newL = 0
                for i in range(L + 1, MAXL + 1):
                    if covered[i]:
                        newL = i
                    else:
                        break
                gain = newL - L
                remove_class(cnt, covered, p, r)

                if gain > best_gain:
                    best_gain = gain
                    best_p, best_r = p, r
                    best_newL = newL

            if best_p is None or best_gain == 0:
                break

            add_class(cnt, covered, best_p, best_r)
            state[best_p] = best_r
            budget -= recip[best_p]
            L = best_newL

        return state, budget, L

    # ---- Run greedy with multiple seeds, keep best ----
    best_state = None
    best_budget = Fraction(1, 1)
    best_L = 0

    for seed in range(6):
        if time.time() > DEADLINE - 35:
            break
        state, budget, L = greedy_init(seed)
        if L > best_L:
            best_L = L
            best_state = state
            best_budget = budget

    if best_state is None:
        # Fallback
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    # ---- CSP backtracking refinement from the best greedy solution ----
    state = dict(best_state)
    budget = best_budget
    cnt = [0] * (MAXL + 1)
    covered = bytearray(MAXL + 1)
    for p, r in state.items():
        add_class(cnt, covered, p, r)
    L = compute_L(covered)

    # Run CSP search with a few different randomizations
    for trial in range(3):
        if time.time() > DEADLINE - 10:
            break

        # Add a small random perturbation to escape local optima
        if trial > 0:
            # Randomly change a few residues
            rng = random.Random(42 + trial)
            keys = list(state.keys())
            for _ in range(min(3, len(keys))):
                p = rng.choice(keys)
                old_r = state[p]
                new_r = rng.randrange(p)
                if new_r != old_r:
                    swap_residue(cnt, covered, p, old_r, new_r)
                    state[p] = new_r

        L = compute_L(covered)
        s, b, L_new = csp_search(state, budget, cnt, covered, L, depth=0, max_depth=30)

        if L_new > best_L:
            best_state = s
            best_budget = b
            best_L = L_new

    # ---- Final verification and return ----
    result = [[p, r] for p, r in best_state.items()]

    # Verify budget constraint
    total = Fraction(0, 1)
    for p, _ in result:
        total += Fraction(1, p)
    if total > Fraction(1, 1):
        # Shouldn't happen, but fallback
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    return result
