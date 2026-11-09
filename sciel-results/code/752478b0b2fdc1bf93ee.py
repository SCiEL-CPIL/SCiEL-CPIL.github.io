"""
Combined approach: Fast Greedy + Enhanced Local Search (Remove-Add)
"""
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # Safety margin


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
            
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
            
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(abs(A[i] - A[j]))
                
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(abs(B[i] - B[j]))
                
    if diffsA & diffsB:
        return False
        
    # Check range
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
            
    return True


def _greedy_build(order, t0, rng):
    """
    Greedy build using incremental sums and bitmask diffs.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = 0
    diffsB = 0
    
    # Precompute diff masks for speed? No, dynamic is fine for n=256.
    
    for x in order:
        if time.time() - t0 > DEADLINE - 2:
            break
            
        added = False
        # Randomly choose order of trying A or B to diversify
        if rng.random() < 0.5:
            sides = [0, 1]
        else:
            sides = [1, 0]
            
        for side in sides:
            if side == 0:
                if len(A) >= MAX_EACH:
                    continue
                # Check Sidon A
                if (x + x) in sumsA:
                    continue
                valid = True
                for a in A:
                    if (x + a) in sumsA:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check Diffs
                new_diffs = 0
                for a in A:
                    d = abs(a - x)
                    if d > 0:
                        new_diffs |= (1 << d)
                        
                if new_diffs & diffsB:
                    continue
                    
                # Add
                A.append(x)
                sumsA.add(x + x)
                for a in A[:-1]:
                    sumsA.add(x + a)
                diffsA |= new_diffs
                added = True
                break
            else:
                if len(B) >= MAX_EACH:
                    continue
                # Check Sidon B
                if (x + x) in sumsB:
                    continue
                valid = True
                for b in B:
                    if (x + b) in sumsB:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check Diffs
                new_diffs = 0
                for b in B:
                    d = abs(b - x)
                    if d > 0:
                        new_diffs |= (1 << d)
                        
                if new_diffs & diffsA:
                    continue
                    
                # Add
                B.append(x)
                sumsB.add(x + x)
                for b in B[:-1]:
                    sumsB.add(x + b)
                diffsB |= new_diffs
                added = True
                break
                
    return A, B


def _local_search(A, B, t0, rng):
    """
    Enhanced local search:
    1. Try to add any unused element.
    2. If stuck, try "Remove 1, Add 2" moves.
    """
    A = list(A)
    B = list(B)
    
    # Recompute state
    sumsA = set()
    sumsB = set()
    diffsA = 0
    diffsB = 0
    
    for x in A:
        for y in A:
            if x <= y:
                sumsA.add(x + y)
            if x != y:
                diffsA |= (1 << abs(x - y))
                
    for x in B:
        for y in B:
            if x <= y:
                sumsB.add(x + y)
            if x != y:
                diffsB |= (1 << abs(x - y))
                
    used = set(A) | set(B)
    unused = [x for x in range(1, N + 1) if x not in used]
    
    improved = True
    while improved and time.time() - t0 < DEADLINE - 2:
        improved = False
        
        # Phase 1: Simple Add
        # Shuffle unused to try different orders
        rng.shuffle(unused)
        for x in unused:
            if time.time() - t0 > DEADLINE - 2:
                break
            
            # Try A
            if len(A) < MAX_EACH:
                if (x + x) not in sumsA:
                    ok = True
                    for a in A:
                        if (x + a) in sumsA:
                            ok = False
                            break
                    if ok:
                        new_diffs = 0
                        for a in A:
                            d = abs(a - x)
                            if d > 0:
                                new_diffs |= (1 << d)
                        if not (new_diffs & diffsB):
                            A.append(x)
                            sumsA.add(x + x)
                            for a in A[:-1]:
                                sumsA.add(x + a)
                            diffsA |= new_diffs
                            used.add(x)
                            unused.remove(x)
                            improved = True
                            break # Restart loop with new state? Or continue? 
                            # Restarting is safer for state consistency in simple loops, 
                            # but we can just mark and continue if we update state correctly.
                            # Let's break and restart the while loop to be safe and simple.
                            break
                            
            # Try B
            if len(B) < MAX_EACH:
                if (x + x) not in sumsB:
                    ok = True
                    for b in B:
                        if (x + b) in sumsB:
                            ok = False
                            break
                    if ok:
                        new_diffs = 0
                        for b in B:
                            d = abs(b - x)
                            if d > 0:
                                new_diffs |= (1 << d)
                        if not (new_diffs & diffsA):
                            B.append(x)
                            sumsB.add(x + x)
                            for b in B[:-1]:
                                sumsB.add(x + b)
                            diffsB |= new_diffs
                            used.add(x)
                            unused.remove(x)
                            improved = True
                            break
                            
        if improved:
            continue
            
        # Phase 2: Remove-Add (Exchange)
        # Try removing one element from A or B and see if we can add 2 new ones.
        # This is expensive, so limit attempts.
        
        # Sample some elements to remove
        candidates_to_remove = []
        if A:
            candidates_to_remove.extend([(0, a) for a in rng.sample(A, min(10, len(A)))])
        if B:
            candidates_to_remove.extend([(1, b) for b in rng.sample(B, min(10, len(B)))])
            
        rng.shuffle(candidates_to_remove)
        
        for side, elem in candidates_to_remove:
            if time.time() - t0 > DEADLINE - 2:
                break
                
            # Save state
            old_A = A[:]
            old_B = B[:]
            old_sumsA = sumsA.copy()
            old_sumsB = sumsB.copy()
            old_diffsA = diffsA
            old_diffsB = diffsB
            old_used = used.copy()
            old_unused = unused[:]
            
            # Remove elem
            if side == 0:
                A.remove(elem)
                # Recompute sums and diffs for A (simpler than incremental removal)
                sumsA = set()
                diffsA = 0
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        sumsA.add(A[i] + A[j])
                    if i != j:
                        diffsA |= (1 << abs(A[i] - A[j]))
                # Note: diffsA calculation above is slightly off for i!=j inside loop, 
                # let's just recompute fully for correctness in this rare path.
                diffsA = 0
                for i in range(len(A)):
                    for j in range(len(A)):
                        if i != j:
                            diffsA |= (1 << abs(A[i] - A[j]))
            else:
                B.remove(elem)
                sumsB = set()
                diffsB = 0
                for i in range(len(B)):
                    for j in range(i, len(B)):
                        sumsB.add(B[i] + B[j])
                diffsB = 0
                for i in range(len(B)):
                    for j in range(len(B)):
                        if i != j:
                            diffsB |= (1 << abs(B[i] - B[j]))
                            
            used.remove(elem)
            unused.append(elem)
            
            # Now try to add 2 elements from unused
            added_count = 0
            # We need to be careful not to add `elem` back immediately if it doesn't help, 
            # but adding it back is a valid move (swap). 
            # However, we want NET gain. So we need to add 2 NEW elements.
            # So we should exclude `elem` from the "new" additions? 
            # If we add `elem` back and one other, net is 0. 
            # If we add two others, net is +1.
            
            # Let's try to greedily add 2 elements from unused (excluding elem for now to ensure net gain)
            new_unused = [u for u in unused if u != elem]
            rng.shuffle(new_unused)
            
            added_elements = []
            
            for x in new_unused:
                if added_count >= 2:
                    break
                if time.time() - t0 > DEADLINE - 1:
                    break
                    
                # Try A
                if len(A) < MAX_EACH:
                    if (x + x) not in sumsA:
                        ok = True
                        for a in A:
                            if (x + a) in sumsA:
                                ok = False
                                break
                        if ok:
                            new_diffs = 0
                            for a in A:
                                d = abs(a - x)
                                if d > 0:
                                    new_diffs |= (1 << d)
                            if not (new_diffs & diffsB):
                                A.append(x)
                                sumsA.add(x + x)
                                for a in A[:-1]:
                                    sumsA.add(x + a)
                                diffsA |= new_diffs
                                used.add(x)
                                added_elements.append(x)
                                added_count += 1
                                continue
                                
                # Try B
                if len(B) < MAX_EACH:
                    if (x + x) not in sumsB:
                        ok = True
                        for b in B:
                            if (x + b) in sumsB:
                                ok = False
                                break
                        if ok:
                            new_diffs = 0
                            for b in B:
                                d = abs(b - x)
                                if d > 0:
                                    new_diffs |= (1 << d)
                            if not (new_diffs & diffsA):
                                B.append(x)
                                sumsB.add(x + x)
                                for b in B[:-1]:
                                    sumsB.add(x + b)
                                diffsB |= new_diffs
                                used.add(x)
                                added_elements.append(x)
                                added_count += 1
                                continue
                                
            if added_count >= 2:
                # Success! Keep the new state.
                # Update unused list: remove added elements
                for x in added_elements:
                    unused.remove(x)
                # elem is still in unused? No, we added it back to unused before.
                # We want elem to remain unused (removed from A/B).
                # So state is already correct.
                improved = True
                break
            else:
                # Revert
                A = old_A
                B = old_B
                sumsA = old_sumsA
                sumsB = old_sumsB
                diffsA = old_diffsA
                diffsB = old_diffsB
                used = old_used
                unused = old_unused
                # elem is back in A/B, so not in unused.
                
    return A, B


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_A, best_B = [], []
    best_size = 0
    
    # Multiple restarts with different strategies
    i = 0
    while time.time() - t0 < DEADLINE - 2:
        # Vary the order
        order = list(range(1, N + 1))
        r = random.Random(42 + i)
        
        # Mix of strategies
        strat = i % 3
        if strat == 0:
            r.shuffle(order)
        elif strat == 1:
            order.sort()
        else:
            order.sort(reverse=True)
            
        A, B = _greedy_build(order, t0, r)
        
        if not A or not B:
            i += 1
            continue
            
        # Local search
        A, B = _local_search(A, B, t0, r)
        
        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)
            
            if best_size == 128:
                break
                
        i += 1
        
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    if not _validate(best_A, best_B):
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
