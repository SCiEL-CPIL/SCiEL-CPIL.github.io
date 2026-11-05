# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    elements = list(range(1, n + 1))

    def _check_sidon(s):
        """Check if set s is a valid Sidon set (pair sums unique including a+a)."""
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def _check_diffs_disjoint(A, B):
        """Check if nonzero difference sets of A and B are disjoint."""
        diffs_a = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_a.add(A[j] - A[i])
        diffs_b = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_b.add(B[j] - B[i])
        return not (diffs_a & diffs_b)

    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Uses in-place tracking for O(n^2) per pass.
        Returns (setA, setB) as lists.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()

        for x in order:
            if time.time() > deadline:
                break
            for side in (0, 1):
                if side == 0:
                    s = setA
                    sums = sumsA
                    diffs_own = diffsA
                    diffs_other = diffsB
                else:
                    s = setB
                    sums = sumsB
                    diffs_own = diffsB
                    diffs_other = diffsA

                # Check Sidon: all pair sums with x must be unique and not in existing sums
                ok = True
                new_sums = []
                for a in s:
                    sm = a + x
                    if sm in sums:
                        ok = False
                        break
                    new_sums.append(sm)
                # Also x + x
                sm = x + x
                if sm in sums:
                    ok = False
                else:
                    new_sums.append(sm)

                if not ok:
                    continue

                # Check differences: new diffs from x with existing elements must not conflict
                new_diffs = set()
                for a in s:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)

                # Check against own diffs (should be empty since we track all)
                if new_diffs & diffs_own:
                    continue

                # Check against other set's diffs
                if new_diffs & diffs_other:
                    continue

                # Add x to this set
                s.append(x)
                sums.update(new_sums)
                diffs_own.update(new_diffs)

        return setA, setB

    def _local_search(setA, setB, deadline):
        """
        Improved local search with systematic candidate scanning and smart kicks.
        """
        setA = list(setA)
        setB = list(setB)
        
        # State tracking
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        
        def _update_state(s, sums, diffs):
            sums.clear()
            diffs.clear()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    sums.add(s[i] + s[j])
            for i in range(len(s)):
                for j in range(len(s)):
                    if i != j:
                        diffs.add(s[j] - s[i])
        
        _update_state(setA, sumsA, diffsA)
        _update_state(setB, sumsB, diffsB)

        best_total = len(setA) + len(setB)
        best_A = list(setA)
        best_B = list(setB)
        
        max_kicks = 50
        kick_count = 0
        no_improve_count = 0
        max_no_improve = 5

        def _count_blocked(s, sums, diffs_own, diffs_other, setA, setB):
            """Count how many candidate elements are blocked by each element in s."""
            blocked_count = {}
            all_diffs = diffs_own | diffs_other
            for i, elem in enumerate(s):
                count = 0
                for x in range(1, 257):
                    if x in s:
                        continue
                    # Check if x would be blocked by this elem
                    # Sum check
                    sm = elem + x
                    if sm in sums:
                        count += 1
                        continue
                    # Diff check
                    d1 = x - elem
                    d2 = elem - x
                    if d1 in all_diffs or d2 in all_diffs:
                        count += 1
                        continue
                    # Also check if x+x creates conflict
                    sm2 = x + x
                    if sm2 in sums:
                        count += 1
                        continue
                blocked_count[i] = count
            return blocked_count

        while time.time() < deadline and kick_count < max_kicks:
            improved_in_pass = False
            
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return best_A, best_B
                
                # Try adding x to A
                if x not in setA:
                    ok = True
                    new_sums = set()
                    for a in setA:
                        sm = a + x
                        if sm in sumsA or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sumsA or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    
                    if ok:
                        new_diffs = set()
                        for a in setA:
                            new_diffs.add(x - a)
                            new_diffs.add(a - x)
                        if not (new_diffs & diffsA) and not (new_diffs & diffsB):
                            setA.append(x)
                            sumsA.update(new_sums)
                            diffsA.update(new_diffs)
                            improved_in_pass = True
                            break
                
                # Try adding x to B
                if x not in setB:
                    ok = True
                    new_sums = set()
                    for b in setB:
                        sm = b + x
                        if sm in sumsB or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sumsB or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    
                    if ok:
                        new_diffs = set()
                        for b in setB:
                            new_diffs.add(x - b)
                            new_diffs.add(b - x)
                        if not (new_diffs & diffsB) and not (new_diffs & diffsA):
                            setB.append(x)
                            sumsB.update(new_sums)
                            diffsB.update(new_diffs)
                            improved_in_pass = True
                            break

            if improved_in_pass:
                total = len(setA) + len(setB)
                if total > best_total:
                    best_total = total
                    best_A = list(setA)
                    best_B = list(setB)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            if not improved_in_pass:
                kick_count += 1
                # Smart kick: remove element with highest blocking count
                best_idx = -1
                best_side = -1
                best_block = -1
                
                if setA:
                    blockedA = _count_blocked(setA, sumsA, diffsA, diffsB, setA, setB)
                    for i, cnt in blockedA.items():
                        if cnt > best_block:
                            best_block = cnt
                            best_idx = i
                            best_side = 0
                
                if setB:
                    blockedB = _count_blocked(setB, sumsB, diffsB, diffsA, setA, setB)
                    for i, cnt in blockedB.items():
                        if cnt > best_block:
                            best_block = cnt
                            best_idx = i
                            best_side = 1
                
                if best_side == 0:
                    setA.pop(best_idx)
                elif best_side == 1:
                    setB.pop(best_idx)
                else:
                    # Fallback to random
                    if random.random() < 0.5 and setA:
                        setA.pop(random.randint(0, len(setA) - 1))
                    elif setB:
                        setB.pop(random.randint(0, len(setB) - 1))
                    else:
                        break
                
                _update_state(setA, sumsA, diffsA)
                _update_state(setB, sumsB, diffsB)

        return best_A, best_B

    best_total = -1
    best_result = ([1], [2])

    # Try many random orderings
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = elements[:]
        random.shuffle(order)

        setA, setB = _greedy_one_ordering(order, deadline)
        if time.time() > deadline:
            break

        # Local search to try to improve
        setA, setB = _local_search(setA, setB, deadline)

        total = len(setA) + len(setB)
        if total > best_total:
            best_total = total
            best_result = (sorted(setA), sorted(setB))

        # Early stop if we hit the max possible
        if best_total >= 128:
            break

    return list(best_result)

# EVOLVE-BLOCK-END
