"""Two Sidon sets with disjoint difference sets in [1,256].
Strategy: partition difference space, greedy build with incremental checks,
then local search (add/remove/swap) to maximize |A|+|B|."""

import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 85  # seconds budget (5s margin before 90s kill)


def _check_sidon(s):
    """Return True if s is a Sidon set (pair sums including doubles are unique)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True


def _diffs(s):
    """Return set of nonzero (positive) differences for sorted s."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d


def _valid_pair(A, B):
    """Check both are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return not (dA & dB)


def _score(A, B):
    return len(A) + len(B)


class _SidonState:
    """Incremental Sidon set with sum and diff tracking for fast add/remove."""

    def __init__(self):
        self.elements = []
        self.sums = set()
        self.diffs = set()

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def _add_elem(self, x):
        """Try to add x. Returns True if added successfully."""
        # Check pair sums: x+x and x+a for all a in elements
        if (x + x) in self.sums:
            return False
        for a in self.elements:
            if (x + a) in self.sums:
                return False
        # Check differences
        for a in self.elements:
            d = abs(x - a)
            if d in self.diffs:
                return False
        # All good, add
        self.elements.append(x)
        self.elements.sort()
        self.sums.add(x + x)
        for a in self.elements:
            if a != x:
                self.sums.add(x + a)
                self.diffs.add(abs(x - a))
        return True

    def _remove_elem(self, x):
        """Remove x from the set. Returns True if it was present."""
        if x not in self.elements:
            return False
        # Remove sums involving x
        for a in self.elements:
            self.sums.discard(x + a)
        self.elements.remove(x)
        # Remove diffs involving x
        for a in self.elements:
            self.diffs.discard(abs(x - a))
        return True

    def get_diffs(self):
        return set(self.diffs)


def _greedy_build(max_size, forbidden_diffs, pool, rng):
    """Greedily build a Sidon set avoiding forbidden_diffs."""
    state = _SidonState()
    pool = list(pool)
    rng.shuffle(pool)
    for x in pool:
        if len(state) >= max_size:
            break
        # Check if x's diffs conflict with forbidden_diffs or state.diffs
        ok = True
        for a in state.elements:
            d = abs(x - a)
            if d in state.diffs or d in forbidden_diffs:
                ok = False
                break
        if ok:
            # Check sums
            if (x + x) in state.sums:
                ok = False
            else:
                for a in state.elements:
                    if (x + a) in state.sums:
                        ok = False
                        break
        if ok:
            state._add_elem(x)
    return state


def _local_search(A_state, B_state, forbidden_A, forbidden_B, rng, time_limit):
    """Local search: try to add elements, remove+re-add, swap."""
    improved = True
    while improved and time.time() < time_limit:
        improved = False

        # Try adding to A
        if len(A_state) < MAX_EACH:
            candidates = [x for x in range(1, N + 1) if x not in A_state.elements]
            rng.shuffle(candidates)
            for x in candidates[:50]:
                # Check diffs
                ok = True
                for a in A_state.elements:
                    d = abs(x - a)
                    if d in A_state.diffs or d in forbidden_A:
                        ok = False
                        break
                if ok and (x + x) not in A_state.sums:
                    for a in A_state.elements:
                        if (x + a) in A_state.sums:
                            ok = False
                            break
                if ok:
                    A_state._add_elem(x)
                    forbidden_B = forbidden_B | A_state.diffs
                    improved = True
                    break

        # Try adding to B
        if len(B_state) < MAX_EACH:
            candidates = [x for x in range(1, N + 1) if x not in B_state.elements]
            rng.shuffle(candidates)
            for x in candidates[:50]:
                ok = True
                for b in B_state.elements:
                    d = abs(x - b)
                    if d in B_state.diffs or d in forbidden_B:
                        ok = False
                        break
                if ok and (x + x) not in B_state.sums:
                    for b in B_state.elements:
                        if (x + b) in B_state.sums:
                            ok = False
                            break
                if ok:
                    B_state._add_elem(x)
                    forbidden_A = forbidden_A | B_state.diffs
                    improved = True
                    break

        # Try removing a random element from A and re-adding a different one
        if len(A_state) > 1 and time.time() < time_limit:
            old_x = A_state.elements[rng.randrange(len(A_state.elements))]
            A_state._remove_elem(old_x)
            old_forbidden_A = set(forbidden_A)
            forbidden_A = forbidden_A - A_state.diffs
            candidates = [x for x in range(1, N + 1) if x not in A_state.elements]
            rng.shuffle(candidates)
            added = False
            for x in candidates[:80]:
                ok = True
                for a in A_state.elements:
                    d = abs(x - a)
                    if d in A_state.diffs or d in forbidden_A:
                        ok = False
                        break
                if ok and (x + x) not in A_state.sums:
                    for a in A_state.elements:
                        if (x + a) in A_state.sums:
                            ok = False
                            break
                if ok:
                    A_state._add_elem(x)
                    forbidden_B = forbidden_B | A_state.diffs
                    added = True
                    break
            if not added:
                # Revert
                A_state._add_elem(old_x)
                forbidden_A = old_forbidden_A
            else:
                improved = True

        # Try removing a random element from B and re-adding a different one
        if len(B_state) > 1 and time.time() < time_limit:
            old_x = B_state.elements[rng.randrange(len(B_state.elements))]
            B_state._remove_elem(old_x)
            old_forbidden_B = set(forbidden_B)
            forbidden_B = forbidden_B - B_state.diffs
            candidates = [x for x in range(1, N + 1) if x not in B_state.elements]
            rng.shuffle(candidates)
            added = False
            for x in candidates[:80]:
                ok = True
                for b in B_state.elements:
                    d = abs(x - b)
                    if d in B_state.diffs or d in forbidden_B:
                        ok = False
                        break
                if ok and (x + x) not in B_state.sums:
                    for b in B_state.elements:
                        if (x + b) in B_state.sums:
                            ok = False
                            break
                if ok:
                    B_state._add_elem(x)
                    forbidden_A = forbidden_A | B_state.diffs
                    added = True
                    break
            if not added:
                B_state._add_elem(old_x)
                forbidden_B = old_forbidden_B
            else:
                improved = True

    return A_state, B_state


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_A, best_B = [], []
    best_score = 0

    rng = random.Random(42)

    # Strategy: partition difference space [1,255] into two sets
    # and greedily build each Sidon set within its partition.
    # Then use local search to improve.

    # Try multiple random partitions of the difference space
    all_diffs = list(range(1, N))  # [1, 255]

    for trial in range(30):
        if time.time() > deadline - 3:
            break

        # Random partition of difference space
        rng.shuffle(all_diffs)
        split = rng.randint(80, 175)
        diff_pool_A = set(all_diffs[:split])
        diff_pool_B = set(all_diffs[split:])

        # Greedy build A using only diff_pool_A
        A_state = _SidonState()
        pool_A = list(range(1, N + 1))
        rng.shuffle(pool_A)
        for x in pool_A:
            if len(A_state) >= MAX_EACH:
                break
            ok = True
            for a in A_state.elements:
                d = abs(x - a)
                if d in A_state.diffs or d not in diff_pool_A:
                    ok = False
                    break
            if ok and (x + x) not in A_state.sums:
                for a in A_state.elements:
                    if (x + a) in A_state.sums:
                        ok = False
                        break
            if ok:
                A_state._add_elem(x)

        # Greedy build B using only diff_pool_B
        B_state = _SidonState()
        pool_B = list(range(1, N + 1))
        rng.shuffle(pool_B)
        for x in pool_B:
            if len(B_state) >= MAX_EACH:
                break
            ok = True
            for b in B_state.elements:
                d = abs(x - b)
                if d in B_state.diffs or d not in diff_pool_B:
                    ok = False
                    break
            if ok and (x + x) not in B_state.sums:
                for b in B_state.elements:
                    if (x + b) in B_state.sums:
                        ok = False
                        break
            if ok:
                B_state._add_elem(x)

        A = list(A_state.elements)
        B = list(B_state.elements)
        score = len(A) + len(B)

        if score > best_score:
            if _valid_pair(A, B):
                best_A, best_B = list(A), list(B)
                best_score = score

        # Local search to improve
        if score >= best_score - 6 and time.time() < deadline - 5:
            # Rebuild states
            A_state2 = _SidonState()
            for x in A:
                A_state2._add_elem(x)
            B_state2 = _SidonState()
            for x in B:
                B_state2._add_elem(x)

            forbidden_A = set(B_state2.diffs)
            forbidden_B = set(A_state2.diffs)

            A_state2, B_state2 = _local_search(
                A_state2, B_state2, forbidden_A, forbidden_B, rng, deadline - 3
            )

            A2 = list(A_state2.elements)
            B2 = list(B_state2.elements)
            score2 = len(A2) + len(B2)

            if score2 > best_score and _valid_pair(A2, B2):
                best_A, best_B = list(A2), list(B2)
                best_score = score2

    # Also try the classic base-4 construction as a baseline
    # Powers of 4: 1, 4, 16, 64, 256 -> but 256 is the max, so use 1,4,16,64
    # Actually, a Sidon set can be {4^i : i=0..k} but that's tiny.
    # Better: use the known construction {4i + j} style.
    # For now, the randomized approach above should be strong.

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [3]

    # Final verification
    if not _valid_pair(best_A, best_B):
        # Fallback: simple valid pair
        best_A = [1, 4, 16, 64]
        best_B = [2, 8, 32, 128]
        if not _valid_pair(best_A, best_B):
            best_A = [1]
            best_B = [3]

    return [best_A, best_B]
