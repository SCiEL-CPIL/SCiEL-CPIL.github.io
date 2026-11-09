import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 82.0


def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    def remaining():
        return DEADLINE - elapsed()

    def diff_set(S):
        return {b - a for a, b in combinations(S, 2)}

    def is_sidon(S):
        sums = [a + b for a, b in combinations_with_replacement(S, 2)]
        return len(sums) == len(set(sums))

    def can_add(x, S, D_S, D_O):
        # Check Sidon property after adding x
        # New sums involving x: x+x, x+s for s in S
        # These must not collide with existing sums or each other
        new_sums = [x + x] + [x + s for s in S]
        # Check for duplicates within new_sums
        if len(new_sums) != len(set(new_sums)):
            return False
        # Check against existing sums
        # We need to maintain sum sets, but for simplicity in this design,
        # let's just recompute or track them. For efficiency, let's track them.
        # Actually, the previous code didn't track sums, it recomputed.
        # For a greedy approach, we can track sum sets.
        # But for now, let's just check the new sums against the property.
        # A more efficient way: keep a set of sums.
        # Let's stick to a slightly slower but simpler check for now,
        # or better, track the sums in the state.
        
        # For the greedy loop, tracking sums is better.
        # But can_add is a pure function here. Let's assume we pass sum sets too?
        # Or just do the O(n) check. With n<=64, it's fine.
        
        # Check if any new sum is already in the set of all sums?
        # We don't have the set of all sums passed in.
        # Let's just compute the full set of sums for the new candidate.
        # Since MAX_EACH is 64, max sums is ~2000. It's fast enough.
        candidate = S + [x]
        if not is_sidon(candidate):
            return False
        
        # Check disjoint differences
        new_diffs = {x - s for s in S} | {s - x for s in S}
        if new_diffs & D_O:
            return False
        if new_diffs & D_S:
            # This can't happen if S is Sidon and we check unique diffs?
            # Wait, differences within S are already in D_S.
            # x-s and s-x are new. They might collide with existing D_S?
            # No, D_S is the set of diffs *within* S.
            # The new diffs involve x. They are new.
            # But they must not collide with *each other*?
            # If x-s1 == x-s2, then s1==s2. So they are unique.
            # If x-s1 == s2-x, then 2x = s1+s2.
            # This is a sum collision! x+x = s1+s2? No.
            # The Sidon property ensures that x+s_i are unique and don't collide with s_j+s_k.
            # But differences?
            # If x - s1 = s2 - x, then 2x = s1 + s2.
            # This implies x+x = s1+s2.
            # Since S is Sidon, s1+s2 is a unique sum.
            # If x is not in S, x+x is a new sum.
            # So x+x != s1+s2.
            # Thus x-s1 != s2-x.
            # So new_diffs are all distinct from each other.
            # And they are not in D_S because D_S only contains diffs between elements of S.
            # So we only need to check against D_O.
            pass
            
        return True

    # Greedy construction with randomization
    best_A = []
    best_B = []
    best_score = 0

    for _ in range(50):
        if remaining() < 5.0:
            break
            
        random.seed(int(time.time() * 1000) % 100000)
        A = []
        B = []
        D_A = set()
        D_B = set()
        
        # Randomly order the universe [1, 256]
        universe = list(range(1, N + 1))
        random.shuffle(universe)
        
        # Alternate adding to A and B, or just try to add to both
        for x in universe:
            if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
                break
            if remaining() < 1.0:
                break
                
            # Try adding to A
            if len(A) < MAX_EACH and can_add(x, A, D_A, D_B):
                A.append(x)
                # Update D_A
                for a in A[:-1]:
                    D_A.add(x - a)
                    D_A.add(a - x)
                # Continue to B? No, one element per x.
                continue
                
            # Try adding to B
            if len(B) < MAX_EACH and can_add(x, B, D_B, D_A):
                B.append(x)
                for b in B[:-1]:
                    D_B.add(x - b)
                    D_B.add(b - x)
                continue

        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]

    if best_score == 0:
        # Fallback to simple deterministic
        A = [1, 2, 4, 8, 16]
        B = [3, 5, 6, 7, 9]
        best_A = A
        best_B = B

    return [best_A, best_B]
