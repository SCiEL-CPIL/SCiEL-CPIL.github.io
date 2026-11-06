import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s limit

    n = 256

    def get_diffs(S):
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    class SetState:
        __slots__ = ['elements', 'sums', 'diffs']

        def __init__(self):
            self.elements = []
            self.sums = set()
            self.diffs = set()

        def try_add(self, x, other_diffs):
            if x in self.elements:
                return False
            
            # Check sums: x + s for all s in elements, and 2*x
            # x + s must not be in self.sums
            # 2*x must not be in self.sums
            # Also, x + s1 != x + s2 implies s1 != s2 which is true since elements are unique.
            # But we must ensure x+s is not already a sum of two existing elements.
            # And 2*x is not already a sum.
            
            # Optimization: 
            # 1. Check 2*x
            if 2 * x in self.sums:
                return False
            
            # 2. Check x + s for all s in elements
            for s in self.elements:
                if x + s in self.sums:
                    return False
            
            # Check diffs: |x - s| for all s in elements
            # |x - s| must not be in self.diffs
            # |x - s| must not be in other_diffs
            for s in self.elements:
                d = abs(x - s)
                if d in self.diffs or d in other_diffs:
                    return False
            
            return True

        def commit(self, x):
            self.elements.append(x)
            # Add new sums: x + s for all s in elements (excluding x itself as we appended)
            # Wait, we need to add sums formed by x and all PREVIOUS elements.
            # The loop in try_add checked against self.sums (which contained sums of previous elements).
            # Now we update self.sums.
            for s in self.elements[:-1]:
                self.sums.add(x + s)
            self.sums.add(2 * x)
            
            # Add new diffs
            for s in self.elements[:-1]:
                self.diffs.add(abs(x - s))

        def rebuild(self, elements):
            self.elements = list(elements)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

        def remove(self, x):
            if x not in self.elements:
                return
            self.elements.remove(x)
            self.rebuild(self.elements)

    def greedy_balanced(ordering, max_per_set=64, prefer_A_first=True):
        A = SetState()
        B = SetState()
        for x in ordering:
            if len(A.elements) < len(B.elements):
                order = [(A, B), (B, A)]
            elif len(B.elements) < len(A.elements):
                order = [(B, A), (A, B)]
            else:
                if prefer_A_first:
                    order = [(A, B), (B, A)]
                else:
                    order = [(B, A), (A, B)]

            for target, other in order:
                if len(target.elements) >= max_per_set:
                    continue
                if target.try_add(x, other.diffs):
                    target.commit(x)
                    break
        return A, B

    def greedy_simple(ordering, prefer_A_first=True, max_per_set=64):
        A = SetState()
        B = SetState()
        for x in ordering:
            if prefer_A_first:
                order = [(A, B), (B, A)]
            else:
                order = [(B, A), (A, B)]
            for target, other in order:
                if len(target.elements) >= max_per_set:
                    continue
                if target.try_add(x, other.diffs):
                    target.commit(x)
                    break
        return A, B

    def strong_polish(A_init, B_init, max_time):
        A = SetState()
        B = SetState()
        A.rebuild(A_init)
        B.rebuild(B_init)

        best_total = len(A.elements) + len(B.elements)
        best_A = list(A.elements)
        best_B = list(B.elements)

        all_elements = set(range(1, n + 1))
        rng = random.Random(42)

        improved = True
        while improved and time.time() < max_time:
            improved = False
            used = set(A.elements) | set(B.elements)
            unused = list(all_elements - used)
            rng.shuffle(unused)

            # Phase 1: add unused elements
            for x in unused:
                if time.time() > max_time:
                    break
                for target, other in [(A, B), (B, A)]:
                    if len(target.elements) >= 64:
                        continue
                    if target.try_add(x, other.diffs):
                        target.commit(x)
                        total = len(A.elements) + len(B.elements)
                        if total > best_total:
                            best_total = total
                            best_A = list(A.elements)
                            best_B = list(B.elements)
                        improved = True
                        break

            # Phase 2: try swaps
            if time.time() > max_time - 2:
                break
            attempts = 0
            while time.time() < max_time - 1 and attempts < 5000:
                attempts += 1
                if not A.elements or not B.elements:
                    break
                if rng.random() < 0.5:
                    src, dst = A, B
                else:
                    src, dst = B, A
                idx = rng.randrange(len(src.elements))
                x = src.elements[idx]
                src.remove(x)
                if dst.try_add(x, src.diffs):
                    dst.commit(x)
                    total = len(A.elements) + len(B.elements)
                    if total > best_total:
                        best_total = total
                        best_A = list(A.elements)
                        best_B = list(B.elements)
                        improved = True
                else:
                    src.rebuild(src.elements + [x])

            # Phase 3: try 1-removal + 1-addition
            if time.time() > max_time - 2:
                break
            if improved:
                continue
            
            # Try removing from A and adding to A
            for i in range(len(A.elements)):
                if time.time() > max_time - 1:
                    break
                x_remove = A.elements[i]
                A.remove(x_remove)
                used = set(A.elements) | set(B.elements)
                unused = list(all_elements - used)
                rng.shuffle(unused)
                added = False
                for x_add in unused:
                    if A.try_add(x_add, B.diffs):
                        A.commit(x_add)
                        added = True
                        total = len(A.elements) + len(B.elements)
                        if total > best_total:
                            best_total = total
                            best_A = list(A.elements)
                            best_B = list(B.elements)
                        improved = True
                        break
                if not added:
                    A.rebuild(A.elements + [x_remove])

            # Try removing from B and adding to B
            if not improved:
                for i in range(len(B.elements)):
                    if time.time() > max_time - 1:
                        break
                    x_remove = B.elements[i]
                    B.remove(x_remove)
                    used = set(A.elements) | set(B.elements)
                    unused = list(all_elements - used)
                    rng.shuffle(unused)
                    added = False
                    for x_add in unused:
                        if B.try_add(x_add, A.diffs):
                            B.commit(x_add)
                            added = True
                            total = len(A.elements) + len(B.elements)
                            if total > best_total:
                                best_total = total
                                best_A = list(A.elements)
                                best_B = list(B.elements)
                            improved = True
                            break
                    if not added:
                        B.rebuild(B.elements + [x_remove])

        return best_A, best_B

    strategies = []

    # Ascending
    strategies.append(list(range(1, 257)))
    # Descending
    strategies.append(list(range(256, 0, -1)))
    # Odd then even
    strategies.append(list(range(1, 257, 2)) + list(range(2, 257, 2)))
    # Even then odd
    strategies.append(list(range(2, 257, 2)) + list(range(1, 257, 2)))
    # Interleaved low-high
    interleaved = []
    for i in range(128):
        interleaved.append(i + 1)
        interleaved.append(256 - i)
    strategies.append(interleaved)
    # Multiples of 4, then 2 mod 4, then odd
    strategies.append([x for x in range(4, 257, 4)] + [x for x in range(2, 257, 4)] + [x for x in range(1, 257, 2)])
    # Reverse multiples
    strategies.append([x for x in range(256, 0, -4)] + [x for x in range(255, 0, -4)] + [x for x in range(255, 0, -2)])

    # Primes-based
    primes = [p for p in range(2, 257) if all(p % q != 0 for q in range(2, int(math.isqrt(p)) + 1))]
    strategies.append(primes + [x for x in range(1, 257) if x not in primes])

    # Random shuffles - more seeds
    for seed in range(20):
        s = list(range(1, 257))
        random.Random(seed * 1000 + 7).shuffle(s)
        strategies.append(s)

    best_total = 0
    best_A = None
    best_B = None

    for strategy in strategies:
        if time.time() > deadline - 20:
            break

        for prefer_A in [True, False]:
            if time.time() > deadline - 20:
                break
            A, B = greedy_balanced(strategy, 64, prefer_A)
            total = len(A.elements) + len(B.elements)
            if total > best_total:
                best_total = total
                best_A = list(A.elements)
                best_B = list(B.elements)

        for prefer_A in [True, False]:
            if time.time() > deadline - 20:
                break
            A, B = greedy_simple(strategy, prefer_A, 64)
            total = len(A.elements) + len(B.elements)
            if total > best_total:
                best_total = total
                best_A = list(A.elements)
                best_B = list(B.elements)

    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [2]
        best_total = 2

    if time.time() < deadline - 5:
        polish_time = min(deadline - 5 - time.time(), 25.0)
        best_A, best_B = strong_polish(best_A, best_B, polish_time)
        best_total = len(best_A) + len(best_B)

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
