import random

N = 256

def construct():
    # Helper to check if x can be added to A with sums S
    def can_add(x, A, sums):
        new_sums = [x + a for a in A] + [2 * x]
        # Check for collisions within new sums
        if len(set(new_sums)) != len(new_sums):
            return False
        # Check for collisions with existing sums
        for s in new_sums:
            if s in sums:
                return False
        return True

    def is_sidon(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A, N):
        if not is_sidon(A):
            return False
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A:
                continue
            # Try adding x
            new_sums = [x + a for a in A] + [2 * x]
            # Check if adding x destroys Sidon property
            # Collision if any new sum is in sums, or new sums collide with each other
            valid = True
            new_sum_set = set(new_sums)
            if len(new_sum_set) != len(new_sums):
                valid = False # Internal collision
            else:
                for s in new_sums:
                    if s in sums:
                        valid = False
                        break
            if valid:
                # If valid, then A is NOT maximal (we could add x)
                return False
        return True

    best_set = None
    best_size = float('inf')

    # 1. Try Powers of 2
    p2 = [2**i for i in range(8) if 2**i <= N]
    if is_sidon(p2) and is_maximal(p2, N):
        best_set = p2
        best_size = len(p2)

    # 2. Try Max-Degree Greedy
    # We simulate adding elements one by one, picking the one that blocks the most others
    A = []
    sums = set()
    remaining = list(range(1, N + 1))
    
    # Pre-compute conflicts? Too complex. Just do it online.
    # To speed up, we can just pick the first valid one? No, we want max degree.
    # Max degree approximation: 
    # When considering x, how many y in remaining are blocked by x?
    # y is blocked by x if (A union {x}) union {y} is not Sidon.
    # This is expensive to compute for all pairs.
    # Let's just use Random Greedy with a bias towards larger numbers?
    # Or just standard Random Greedy.

    for _ in range(500):
        A = []
        sums = set()
        # Use a shuffled list of candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, A, sums):
                A.append(x)
                # Update sums
                for a in A[:-1]: # A[:-1] is A before x was added
                    sums.add(x + a)
                sums.add(2 * x)
        
        if is_maximal(A, N):
            if len(A) < best_size:
                best_size = len(A)
                best_set = A[:]

    # Fallback to original greedy if nothing found (should not happen)
    if best_set is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    return best_set
