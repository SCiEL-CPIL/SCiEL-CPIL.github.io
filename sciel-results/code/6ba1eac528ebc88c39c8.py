# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 52.0
    n = 256
    
    def _valid_sidon(s):
        if len(s) == 0:
            return True
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True
    
    def _diffs(s):
        d = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d.add(s[j] - s[i])
        return d
    
    def _check_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        if not _valid_sidon(A):
            return False
        if not _valid_sidon(B):
            return False
        da = _diffs(A)
        db = _diffs(B)
        if da & db:
            return False
        return True
    
    def _greedy(order, deadline):
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
                    s, sums, do, d1 = setA, sumsA, diffsA, diffsB
                else:
                    s, sums, do, d1 = setB, sumsB, diffsB, diffsA
                
                if x in s:
                    continue
                
                ok = True
                new_sums = []
                for a in s:
                    sm = a + x
                    if sm in sums:
                        ok = False
                        break
                    new_sums.append(sm)
                sm = x + x
                if sm in sums:
                    ok = False
                else:
                    new_sums.append(sm)
                
                if not ok:
                    continue
                
                new_diffs = set()
                for a in s:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                
                if new_diffs & do or new_diffs & d1:
                    continue
                
                s.append(x)
                sums.update(new_sums)
                do.update(new_diffs)
                break
        
        return setA, setB
    
    def _local_search(setA, setB, deadline):
        setA = list(setA)
        setB = list(setB)
        
        def _rebuild():
            sumsA = set()
            sumsB = set()
            diffsA = set()
            diffsB = set()
            for i in range(len(setA)):
                for j in range(i, len(setA)):
                    sumsA.add(setA[i] + setA[j])
            for i in range(len(setB)):
                for j in range(i, len(setB)):
                    sumsB.add(setB[i] + setB[j])
            for i in range(len(setA)):
                for j in range(len(setA)):
                    if i != j:
                        diffsA.add(setA[j] - setA[i])
            for i in range(len(setB)):
                for j in range(len(setB)):
                    if i != j:
                        diffsB.add(setB[j] - setB[i])
            return sumsA, sumsB, diffsA, diffsB
        
        sumsA, sumsB, diffsA, diffsB = _rebuild()
        best_total = len(setA) + len(setB)
        best_A = list(setA)
        best_B = list(setB)
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline:
            improved = False
            
            # Try adding random elements
            for _ in range(20):
                if time.time() > deadline:
                    break
                x = random.randint(1, n)
                for side in (0, 1):
                    if side == 0:
                        s, sums, do, d1 = setA, sumsA, diffsA, diffsB
                    else:
                        s, sums, do, d1 = setB, sumsB, diffsB, diffsA
                    
                    if x in s:
                        continue
                    
                    ok = True
                    new_sums = set()
                    for a in s:
                        sm = a + x
                        if sm in sums or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sums or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    
                    if not ok:
                        continue
                    
                    new_diffs = set()
                    for a in s:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    
                    if new_diffs & do or new_diffs & d1:
                        continue
                    
                    s.append(x)
                    sums.update(new_sums)
                    do.update(new_diffs)
                    improved = True
                    break
                if improved:
                    break
            
            if not improved:
                # Try swap moves
                if len(setA) > 1 and len(setB) > 1:
                    # Try removing from A and adding to B, or vice versa
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        
                        if random.random() < 0.5:
                            # Remove from A, try to add to B
                            if not setA:
                                continue
                            idx = random.randint(0, len(setA) - 1)
                            removed = setA.pop(idx)
                            sumsA, sumsB, diffsA, diffsB = _rebuild()
                            
                            if removed not in setB:
                                ok = True
                                new_sums = set()
                                for b in setB:
                                    sm = b + removed
                                    if sm in sumsB or sm in new_sums:
                                        ok = False
                                        break
                                    new_sums.add(sm)
                                sm = removed + removed
                                if sm in sumsB or sm in new_sums:
                                    ok = False
                                else:
                                    new_sums.add(sm)
                                if ok:
                                    new_diffs = set()
                                    for b in setB:
                                        new_diffs.add(removed - b)
                                        new_diffs.add(b - removed)
                                    if not (new_diffs & diffsB) and not (new_diffs & diffsA):
                                        setB.append(removed)
                                        sumsB.update(new_sums)
                                        diffsB.update(new_diffs)
                                        improved = True
                                if not improved:
                                    setA.append(removed)
                                    sumsA, sumsB, diffsA, diffsB = _rebuild()
                        else:
                            # Remove from B, try to add to A
                            if not setB:
                                continue
                            idx = random.randint(0, len(setB) - 1)
                            removed = setB.pop(idx)
                            sumsA, sumsB, diffsA, diffsB = _rebuild()
                            
                            if removed not in setA:
                                ok = True
                                new_sums = set()
                                for a in setA:
                                    sm = a + removed
                                    if sm in sumsA or sm in new_sums:
                                        ok = False
                                        break
                                    new_sums.add(sm)
                                sm = removed + removed
                                if sm in sumsA or sm in new_sums:
                                    ok = False
                                else:
                                    new_sums.add(sm)
                                if ok:
                                    new_diffs = set()
                                    for a in setA:
                                        new_diffs.add(removed - a)
                                        new_diffs.add(a - removed)
                                    if not (new_diffs & diffsA) and not (new_diffs & diffsB):
                                        setA.append(removed)
                                        sumsA.update(new_sums)
                                        diffsA.update(new_diffs)
                                        improved = True
                                if not improved:
                                    setB.append(removed)
                                    sumsA, sumsB, diffsA, diffsB = _rebuild()
                
                if not improved:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
            else:
                no_improve = 0
            
            total = len(setA) + len(setB)
            if total > best_total:
                best_total = total
                best_A = list(setA)
                best_B = list(setB)
        
        return best_A, best_B
    
    best_total = -1
    best_result = ([1], [2])
    
    # Try multiple random orderings
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        setA, setB = _greedy(order, deadline)
        if time.time() > deadline:
            break
        
        setA, setB = _local_search(setA, setB, deadline)
        
        total = len(setA) + len(setB)
        if total > best_total:
            best_total = total
            best_result = (sorted(setA), sorted(setB))
        
        if best_total >= 40:
            break
    
    # Final validation with fallback
    A, B = best_result
    if not _check_solution(A, B):
        # Try a simple known-good construction
        # Powers of 2 for A
        A = [1, 2, 4, 8, 16, 32, 64, 128]
        # For B, use numbers that don't create conflicting differences
        # Differences of A: all values from -127 to 127 that are differences of powers of 2
        # Let's just try small odd numbers for B
        B = [3, 5, 7, 9, 11, 13, 15, 17]
        
        if not _check_solution(A, B):
            # Even simpler fallback
            A = [1, 2, 4, 8]
            B = [3, 5, 7, 9]
        
        if not _check_solution(A, B):
            A = [1]
            B = [2]
    
    return [list(A), list(B)]

# EVOLVE-BLOCK-END
