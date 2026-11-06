import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    n = 256

    def get_diffs(S):
        """Get all nonzero absolute differences."""
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
            """Try to add x. Returns (success, new_sums, new_diffs)."""
            new_sums = set()
            for s in self.elements:
                ns = x + s
                if ns in self.sums or ns in new_sums:
                    return False, None, None
                new_sums.add(ns)
            two_x = 2 * x
            if two_x in self.sums or two_x in new_sums:
                return False, None, None
            new_sums.add(two_x)

            new_diffs = set()
            for s in self.elements:
                d = abs(x - s)
                if d in other_diffs or d in self.diffs or d in new_diffs:
                    return False, None, None
                new_diffs.add(d)

            return True, new_sums, new_diffs

        def commit(self, x, new_sums, new_diffs):
            self.elements.append(x)
            self.sums.update(new_sums)
            self.diffs.update(new_diffs)

        def remove(self, x):
            """Remove x and recompute sums/diffs."""
            self.elements.remove(x)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

        def rebuild(self, elements):
            self.elements = list(elements)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

    def build_with_order(ordering, prefer_A_first=True):
        A = SetState()
        B = SetState()

        for x in ordering:
            if prefer_A_first:
                order = [(A, B), (B, A)]
            else:
                order = [(B, A), (A, B)]

            for target, other in order:
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    break

        return A, B

    def local_search(A_init, B_init, max_time=10.0):
        """Improve via weighted local search (Simulated Annealing style)."""
        A = SetState()
        B = SetState()
        A.rebuild(A_init)
        B.rebuild(B_init)

        best_total = len(A.elements) + len(B.elements)
        best_A = list(A.elements)
        best_B = list(B.elements)

        all_elements = set(range(1, n + 1))
        
        def current_score():
            return len(A.elements) + len(B.elements)

        def try_move(src, dst, x):
            """Move x from src to dst. Returns True if successful."""
            # Remove x from src
            src.remove(x)
            # Try to add x to dst
            success, new_sums, new_diffs = dst.try_add(x, src.diffs)
            if success:
                dst.commit(x, new_sums, new_diffs)
                return True
            else:
                # Revert
                success2, ns2, nd2 = src.try_add(x, dst.diffs)
                if success2:
                    src.commit(x, ns2, nd2)
                else:
                    # This should not happen if state was valid, but safe fallback
                    src.rebuild(src.elements + [x])
                return False

        def try_add_unused(dst, other, x):
            """Add unused x to dst."""
            success, new_sums, new_diffs = dst.try_add(x, other.diffs)
            if success:
                dst.commit(x, new_sums, new_diffs)
                return True
            return False

        # Main loop
        # We use a simple hill-climbing with random restarts/annealing
        # Objective: Maximize size. 
        # Moves:
        # 1. Add random unused element to A or B.
        # 2. Remove random element from A or B (to unblock others).
        # 3. Swap element between A and B.
        
        # To make it efficient, we track unused elements dynamically or just pick random x in 1..n
        # and check if it's in A or B.
        
        T = 1.0 # Temperature
        T_min = 0.01
        cooling = 0.999
        
        rng = random.Random(123)
        
        while time.time() < max_time:
            if T < T_min:
                break
                
            curr_score = current_score()
            
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.5:
                # Add unused
                x = rng.randint(1, n)
                if x in A.elements or x in B.elements:
                    continue
                target = A if rng.random() < 0.5 else B
                other = B if target is A else A
                if try_add_unused(target, other, x):
                    new_score = current_score()
                    if new_score >= curr_score:
                        if new_score > best_total:
                            best_total = new_score
                            best_A = list(A.elements)
                            best_B = list(B.elements)
                    else:
                        # Should not happen for add, but if it did (impossible), revert
                        pass
                else:
                    # Move failed, no state change
                    pass
                    
            elif move_type < 0.8:
                # Remove element
                if not A.elements and not B.elements:
                    continue
                if A.elements and (not B.elements or rng.random() < 0.5):
                    x = rng.choice(A.elements)
                    A.remove(x)
                else:
                    x = rng.choice(B.elements)
                    B.remove(x)
                new_score = current_score()
                delta = new_score - curr_score
                # Accept if better, or with probability exp(delta/T) if worse
                if delta >= 0 or rng.random() < math.exp(delta / T):
                    if new_score > best_total:
                        best_total = new_score
                        best_A = list(A.elements)
                        best_B = list(B.elements)
                else:
                    # Revert
                    # We need to add it back. Since we removed it, we know it was valid.
                    # But try_add might fail if state changed? No, state only changed by removal.
                    # So adding back should always succeed.
                    if x in A.elements:
                         # This is wrong, x was removed from A.
                         pass
                    # We need to know which set it was removed from.
                    # Let's restructure the remove logic to be reversible safely.
                    pass # Handled below in a cleaner way

            else:
                # Swap
                if not A.elements or not B.elements:
                    continue
                if rng.random() < 0.5:
                    x = rng.choice(A.elements)
                    if try_move(A, B, x):
                        new_score = current_score()
                        # Score doesn't change on swap (size is same)
                        # But it might unblock future moves.
                        # For pure size maximization, swap is neutral.
                        # We accept it to explore space.
                else:
                    x = rng.choice(B.elements)
                    if try_move(B, A, x):
                        pass

            T *= cooling
            
        return best_A, best_B

    # Generate orderings
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]

    # Interleaved low-high
    inter = []
    for i in range(1, n // 2 + 1):
        inter.append(i)
        inter.append(n + 1 - i)
    orderings.append(inter[:n])

    rng = random.Random(42)
    for _ in range(12):
        s = list(range(1, n + 1))
        rng.shuffle(s)
        orderings.append(s)

    best_total = 0
    best_A = None
    best_B = None

    for ordering in orderings:
        if time.time() > deadline - 30:
            break
        for prefer_A in [True, False]:
            if time.time() > deadline - 30:
                break
            A_state, B_state = build_with_order(ordering, prefer_A)
            total = len(A_state.elements) + len(B_state.elements)
            if total > best_total:
                best_total = total
                best_A = list(A_state.elements)
                best_B = list(B_state.elements)

    # Local search polish
    if best_A is not None and best_B is not None:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            polished_A, polished_B = local_search(best_A, best_B, max_time=min(remaining, 40.0))
            if len(polished_A) + len(polished_B) > best_total:
                best_A = polished_A
                best_B = polished_B
                best_total = len(best_A) + len(best_B)

    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
