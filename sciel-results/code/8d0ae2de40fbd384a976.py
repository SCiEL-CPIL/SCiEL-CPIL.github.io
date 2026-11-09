"""Two Sidon sets with disjoint difference sets — incremental bitmask approach."""
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin under 90s kill

    n = 256
    max_size = 64
    # Bitmask for differences: index (d - (-(n-1))) = d + (n-1), range 0..2(n-1)
    diff_offset = n - 1
    diff_bits = 2 * diff_offset + 1  # 511 bits, fits in Python int

    # Precompute: for each x, the diff bits it would add when added to a set S
    # Actually diffs depend on S, so we compute incrementally.

    def _can_add_to_A(x, sumA, diffA, diffB):
        """Check if x can be added to A. Returns (ok, new_sumA, new_diffA) or (False, None, None)."""
        # Check Sidon: new sums x+s for s in sumA's source... 
        # We need the actual set elements. Let's pass sets too.
        pass

    # Better: maintain sets and use incremental checks
    def _sidon_ok_with(x, S, sum_set):
        """Check if adding x to Sidon set S (with current sum_set) keeps it Sidon.
        sum_set contains all pair sums a+b for a<=b in S.
        New sums to check: x+s for s in S (including x+x).
        Return (ok, new_sum_set) or (False, None)."""
        for s in S:
            ns = x + s
            if ns in sum_set:
                return False, None
        # All new sums are distinct from each other? x+s1 = x+s2 implies s1=s2, so yes.
        # But we need to also check x+x vs x+s: x+x = x+s => x=s, which is in S, 
        # and x+x is checked as s=x case. So fine.
        new_sum_set = sum_set | set(x + s for s in S)
        return True, new_sum_set

    def _diffs_with(x, S, diff_mask):
        """Compute new diff mask after adding x to S.
        diff_mask has bits for all nonzero diffs in S.
        New diffs: x-s and s-x for s in S.
        Return new_diff_mask."""
        new_mask = diff_mask
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 != 0:
                new_mask |= (1 << (d1 + diff_offset))
            if d2 != 0:
                new_mask |= (1 << (d2 + diff_offset))
        return new_mask

    def _greedy_fill(order, prefer_A_first):
        """Greedy fill with random order. Returns (A, B)."""
        A, B = [], []
        sumA, sumB = set(), set()
        diffA, diffB = 0, 0

        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break
            placed = False
            # Try A first (or B first based on preference)
            if prefer_A_first:
                # Try A
                if len(A) < max_size:
                    ok, new_sumA = _sidon_ok_with(x, A, sumA)
                    if ok:
                        new_diffA = _diffs_with(x, A, diffA)
                        if (new_diffA & diffB) == 0:
                            A.append(x)
                            sumA = new_sumA
                            diffA = new_diffA
                            placed = True
                if not placed and len(B) < max_size:
                    ok, new_sumB = _sidon_ok_with(x, B, sumB)
                    if ok:
                        new_diffB = _diffs_with(x, B, diffB)
                        if (diffA & new_diffB) == 0:
                            B.append(x)
                            sumB = new_sumB
                            diffB = new_diffB
                            placed = True
            else:
                # Try B first
                if len(B) < max_size:
                    ok, new_sumB = _sidon_ok_with(x, B, sumB)
                    if ok:
                        new_diffB = _diffs_with(x, B, diffB)
                        if (diffA & new_diffB) == 0:
                            B.append(x)
                            sumB = new_sumB
                            diffB = new_diffB
                            placed = True
                if not placed and len(A) < max_size:
                    ok, new_sumA = _sidon_ok_with(x, A, sumA)
                    if ok:
                        new_diffA = _diffs_with(x, A, diffA)
                        if (new_diffA & diffB) == 0:
                            A.append(x)
                            sumA = new_sumA
                            diffA = new_diffA
                            placed = True
        return A, B

    def _local_search(A, B, sumA, sumB, diffA, diffB):
        """Try to improve by swapping elements between A and B."""
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(len(A)):
                if time.time() > deadline - 2.0:
                    break
                ai = A[i]
                for j in range(len(B)):
                    if time.time() > deadline - 2.0:
                        break
                    bj = B[j]
                    if ai == bj:
                        continue
                    # Try swap: remove ai from A, add bj to A; remove bj from B, add ai to B
                    A_new = A[:i] + A[i+1:]
                    B_new = B[:j] + B[j+1:]
                    
                    # Check if bj can be added to A_new
                    okA, new_sumA = _sidon_ok_with(bj, A_new, sumA)
                    if not okA:
                        continue
                    new_diffA = _diffs_with(bj, A_new, diffA)
                    # But we need to remove ai's contribution from diffA first... 
                    # This is getting complex. Let's just recompute for swaps.
                    # Actually, let's skip complex swap and just try adding new elements.
                    break
        return A, B, sumA, sumB, diffA, diffB

    def _try_add_new(A, B, sumA, sumB, diffA, diffB):
        """Try to add any remaining element to A or B."""
        used = set(A) | set(B)
        improved = True
        while improved:
            if time.time() > deadline - 1.0:
                break
            improved = False
            for x in range(1, n+1):
                if x in used:
                    continue
                # Try A
                if len(A) < max_size:
                    ok, new_sumA = _sidon_ok_with(x, A, sumA)
                    if ok:
                        new_diffA = _diffs_with(x, A, diffA)
                        if (new_diffA & diffB) == 0:
                            A.append(x)
                            sumA = new_sumA
                            diffA = new_diffA
                            used.add(x)
                            improved = True
                            break
                # Try B
                if not improved and len(B) < max_size:
                    ok, new_sumB = _sidon_ok_with(x, B, sumB)
                    if ok:
                        new_diffB = _diffs_with(x, B, diffB)
                        if (diffA & new_diffB) == 0:
                            B.append(x)
                            sumB = new_sumB
                            diffB = new_diffB
                            used.add(x)
                            improved = True
                            break
        return A, B, sumA, sumB, diffA, diffB

    def _validate(A, B):
        """Validate the result."""
        if not A or not B:
            return False
        # Check Sidon for A
        sumsA = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sumsA:
                    return False
                sumsA.add(s)
        # Check Sidon for B
        sumsB = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sumsB:
                    return False
                sumsB.add(s)
        # Check disjoint diffs
        dA = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    dA.add(A[j] - A[i])
        dB = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    dB.add(B[j] - B[i])
        if dA & dB:
            return False
        # Check range
        if any(x < 1 or x > n for x in A + B):
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        return True

    best_A, best_B = [], []
    best_score = 0

    # Multiple random restarts with different orderings
    trial = 0
    while time.time() < deadline - 5.0:
        trial += 1
        order = list(range(1, n+1))
        random.shuffle(order)
        
        # Try both preferences
        for pref in (True, False):
            if time.time() > deadline - 5.0:
                break
            A, B = _greedy_fill(order, pref)
            
            # Try to add more elements
            sumA = set()
            for i in range(len(A)):
                for j in range(i, len(A)):
                    sumA.add(A[i] + A[j])
            sumB = set()
            for i in range(len(B)):
                for j in range(i, len(B)):
                    sumB.add(B[i] + B[j])
            diffA = 0
            for i in range(len(A)):
                for j in range(len(A)):
                    if i != j:
                        d = A[j] - A[i]
                        diffA |= (1 << (d + diff_offset))
            diffB = 0
            for i in range(len(B)):
                for j in range(len(B)):
                    if i != j:
                        d = B[j] - B[i]
                        diffB |= (1 << (d + diff_offset))
            
            A, B, sumA, sumB, diffA, diffB = _try_add_new(A, B, sumA, sumB, diffA, diffB)
            
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = A[:], B[:]
        
        # Occasionally try a different random order
        if trial % 3 == 0:
            order2 = list(range(1, n+1))
            random.shuffle(order2)
            for pref in (True, False):
                if time.time() > deadline - 5.0:
                    break
                A, B = _greedy_fill(order2, pref)
                sumA = set()
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        sumA.add(A[i] + A[j])
                sumB = set()
                for i in range(len(B)):
                    for j in range(i, len(B)):
                        sumB.add(B[i] + B[j])
                diffA = 0
                for i in range(len(A)):
                    for j in range(len(A)):
                        if i != j:
                            d = A[j] - A[i]
                            diffA |= (1 << (d + diff_offset))
                diffB = 0
                for i in range(len(B)):
                    for j in range(len(B)):
                        if i != j:
                            d = B[j] - B[i]
                            diffB |= (1 << (d + diff_offset))
                A, B, sumA, sumB, diffA, diffB = _try_add_new(A, B, sumA, sumB, diffA, diffB)
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_A, best_B = A[:], B[:]

    # Final validation
    if not _validate(best_A, best_B):
        # Fallback to simple valid solution
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
