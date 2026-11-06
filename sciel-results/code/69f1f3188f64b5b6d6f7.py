"""Two Sidon sets with disjoint difference sets.
Model-and-optimize approach: use a structured parameterization (arithmetic-like
progressions with controlled differences) plus greedy local improvement to maximize
combined size under the disjoint-difference constraint."""

import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N  # 512
MAX_DIFF_OFFSET = N  # diffs in [-255, 255], offset by 255 -> [0, 510]
DIFF_BITS = 2 * N - 1  # 511

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

class BitmaskState:
    """State using NumPy uint64 bitmasks for fast collision detection."""
    def __init__(self, A, B):
        self.A = list(A)
        self.B = list(B)
        # sums_A: bit s set if s is a pair sum in A (s in [2, 512])
        self.sums_A = 0
        self.sums_B = 0
        # diffs_A: bit d set if d is a nonzero diff in A (d in [-255,255], offset by 255)
        self.diffs_A = 0
        self.diffs_B = 0
        self._rebuild()
    
    def _rebuild(self):
        self.sums_A = 0
        for i in range(len(self.A)):
            for j in range(i, len(self.A)):
                s = self.A[i] + self.A[j]
                self.sums_A |= (1 << s)
        self.sums_B = 0
        for i in range(len(self.B)):
            for j in range(i, len(self.B)):
                s = self.B[i] + self.B[j]
                self.sums_B |= (1 << s)
        self.diffs_A = 0
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if i != j:
                    d = self.A[j] - self.A[i] + MAX_DIFF_OFFSET
                    self.diffs_A |= (1 << d)
        self.diffs_B = 0
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                if i != j:
                    d = self.B[j] - self.B[i] + MAX_DIFF_OFFSET
                    self.diffs_B |= (1 << d)
    
    def can_add_to_A(self, x):
        """Check if x can be added to A."""
        # Sidon check: 2x and x+a for all a in A must not be in sums_A
        if (self.sums_A >> (2*x)) & 1:
            return False
        for a in self.A:
            if (self.sums_A >> (x + a)) & 1:
                return False
        # Diff disjointness: x-a and a-x must not be in diffs_B
        for a in self.A:
            d1 = x - a + MAX_DIFF_OFFSET
            d2 = a - x + MAX_DIFF_OFFSET
            if d1 != MAX_DIFF_OFFSET and (self.diffs_B >> d1) & 1:
                return False
            if d2 != MAX_DIFF_OFFSET and (self.diffs_B >> d2) & 1:
                return False
        return True
    
    def can_add_to_B(self, x):
        """Check if x can be added to B."""
        if (self.sums_B >> (2*x)) & 1:
            return False
        for b in self.B:
            if (self.sums_B >> (x + b)) & 1:
                return False
        for b in self.B:
            d1 = x - b + MAX_DIFF_OFFSET
            d2 = b - x + MAX_DIFF_OFFSET
            if d1 != MAX_DIFF_OFFSET and (self.diffs_A >> d1) & 1:
                return False
            if d2 != MAX_DIFF_OFFSET and (self.diffs_A >> d2) & 1:
                return False
        return True
    
    def add_to_A(self, x):
        self.A.append(x)
        self.sums_A |= (1 << (2*x))
        for a in self.A:
            if a != x:
                self.sums_A |= (1 << (a + x))
        for a in self.A:
            if a != x:
                self.diffs_A |= (1 << (x - a + MAX_DIFF_OFFSET))
                self.diffs_A |= (1 << (a - x + MAX_DIFF_OFFSET))
    
    def add_to_B(self, x):
        self.B.append(x)
        self.sums_B |= (1 << (2*x))
        for b in self.B:
            if b != x:
                self.sums_B |= (1 << (b + x))
        for b in self.B:
            if b != x:
                self.diffs_B |= (1 << (x - b + MAX_DIFF_OFFSET))
                self.diffs_B |= (1 << (b - x + MAX_DIFF_OFFSET))
    
    def remove_from_A(self, x):
        """Remove x from A and rebuild masks."""
        self.A.remove(x)
        self._rebuild()
    
    def remove_from_B(self, x):
        """Remove x from B and rebuild masks."""
        self.B.remove(x)
        self._rebuild()
    
    def valid(self):
        return self.sums_A & self.sums_B == 0  # Actually this isn't right, let's just use _valid
        # For now, rely on external validation
        return True
    
    def score(self):
        return len(self.A) + len(self.B)

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    n = 256
    max_size = 64

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(42)

    def _attempt_build(seed_A=None, seed_B=None, strat="greedy"):
        """Build A and B from optional seeds using bitmask state."""
        A = list(seed_A) if seed_A else []
        B = list(seed_B) if seed_B else []

        if not A and not B:
            if strat == "A_first":
                A.append(rng.randint(1, n))
            else:
                B.append(rng.randint(1, n))

        if not A:
            A.append(rng.randint(1, n))
        if not B:
            B.append(rng.randint(1, n))

        A = list(set(A))
        B = list(set(B))

        if not _valid(A, B):
            return None, None, 0

        state = BitmaskState(A, B)

        # Alternate greedy additions using bitmask checks
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)

        for _round in range(200):
            if time.time() > deadline - 5:
                break
            progress = False

            if len(state.A) < max_size:
                for x in candidates:
                    if x in state.A:
                        continue
                    if state.can_add_to_A(x):
                        state.add_to_A(x)
                        progress = True
                        break

            if len(state.B) < max_size:
                for x in candidates:
                    if x in state.B:
                        continue
                    if state.can_add_to_B(x):
                        state.add_to_B(x)
                        progress = True
                        break

            if not progress:
                break

        if _valid(state.A, state.B):
            return state.A, state.B, len(state.A) + len(state.B)
        return None, None, 0

    def _local_search(state, time_limit=5.0):
        """Aggressive local search with add and 1-for-1 swap moves."""
        end_time = time.time() + time_limit
        A = state.A
        B = state.B
        score = len(A) + len(B)
        candidates = list(range(1, n + 1))
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Try adding to A
            if len(A) < max_size:
                rng.shuffle(candidates)
                for x in candidates:
                    if x in A:
                        continue
                    if state.can_add_to_A(x):
                        state.add_to_A(x)
                        score += 1
                        improved = True
                        break
            
            # Try adding to B
            if len(B) < max_size:
                rng.shuffle(candidates)
                for x in candidates:
                    if x in B:
                        continue
                    if state.can_add_to_B(x):
                        state.add_to_B(x)
                        score += 1
                        improved = True
                        break
            
            # Try 1-for-1 swaps in A: remove a, add x
            if len(A) >= 1 and time.time() < end_time:
                for a in A[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_A(a)
                    for x in candidates:
                        if x == a or x in A:
                            continue
                        if state.can_add_to_A(x):
                            state.add_to_A(x)
                            score += 0  # swap, net zero
                            improved = True
                            break
                    else:
                        # x wasn't added, put a back
                        state.add_to_A(a)
                        state._rebuild()
            
            # Try 1-for-1 swaps in B
            if len(B) >= 1 and time.time() < end_time:
                for b in B[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_B(b)
                    for x in candidates:
                        if x == b or x in B:
                            continue
                        if state.can_add_to_B(x):
                            state.add_to_B(x)
                            improved = True
                            break
                    else:
                        state.add_to_B(b)
                        state._rebuild()
        
        return state, score

    # Main search loop
    for it in range(50000):
        if time.time() > deadline - 3:
            break

        strat = it % 5

        if strat == 0:
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 1:
            A0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, None, "A_first")
        elif strat == 2:
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(None, B0, "B_first")
        elif strat == 3:
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, n + 1), sa))
            B0 = sorted(rng.sample(range(1, n + 1), sb))
            A, B, score = _attempt_build(A0, B0, "greedy")
        else:
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")

        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)

            # Local search with bitmask state
            if time.time() < deadline - 8:
                state = BitmaskState(best_A, best_B)
                state, new_score = _local_search(state, time_limit=min(5.0, max(0.5, deadline - 5 - time.time())))
                if new_score > best_score:
                    best_score = new_score
                    best_A = list(state.A)
                    best_B = list(state.B)

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
