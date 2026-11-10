import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    N = 128
    rng = random.Random(42)

    def time_left():
        return deadline - time.time()

    def can_add(inS, x):
        # inS is a list of bools or bytearray
        # x is the element to add
        # Check if adding x creates a forbidden triple {a,b,c} with sums in S
        # The triple must include x.
        
        # Case 1: x is the largest element c in {a,b,c}
        # a < b < x
        # Need a+b, a+x, b+x in S
        for a in range(1, x):
            if not inS[a]: continue
            sum_ax = a + x
            if sum_ax > N or not inS[sum_ax]: continue
            for b in range(a + 1, x):
                if not inS[b]: continue
                sum_bx = b + x
                sum_ab = a + b
                if sum_bx <= N and inS[sum_bx] and sum_ab <= N and inS[sum_ab]:
                    return False
        
        # Case 2: x is the middle element b in {a,b,c}
        # a < x < c
        # Need a+x, a+c, x+c in S
        for a in range(1, x):
            if not inS[a]: continue
            sum_ax = a + x
            if sum_ax > N or not inS[sum_ax]: continue
            for c in range(x + 1, N + 1):
                if not inS[c]: continue
                sum_xc = x + c
                sum_ac = a + c
                if sum_xc <= N and inS[sum_xc] and sum_ac <= N and inS[sum_ac]:
                    return False
                    
        # Case 3: x is the smallest element a in {a,b,c}
        # x < b < c
        # Need x+b, x+c, b+c in S
        for b in range(x + 1, N + 1):
            if not inS[b]: continue
            sum_xb = x + b
            if sum_xb > N or not inS[sum_xb]: continue
            for c in range(b + 1, N + 1):
                if not inS[c]: continue
                sum_xc = x + c
                sum_bc = b + c
                if sum_xc <= N and inS[sum_xc] and sum_bc <= N and inS[sum_bc]:
                    return False
                    
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    # Initial set: Upper half
    # This is always valid because sums exceed N
    inS = bytearray(N + 1)
    mem = []
    for i in range(N // 2 + 1, N + 1):
        inS[i] = 1
        mem.append(i)
        
    best_mem = list(mem)
    best_size = len(mem)

    # Greedy addition phase
    def greedy_add():
        nonlocal inS, mem
        added_any = True
        while added_any and time_left() > 5.0:
            added_any = False
            # Try to add elements from 1 to N
            # Random order might help escape local optima faster? 
            # Let's try sorted order first for determinism, then randomize later
            for x in range(1, N + 1):
                if inS[x]:
                    continue
                if can_add(inS, x):
                    inS[x] = 1
                    mem.append(x)
                    added_any = True
                    if len(mem) > best_size:
                        best_size = len(mem)
                        best_mem = list(mem)
            # If we added something, we loop again to see if more can be added
            # because adding x might enable adding y? No, adding x only adds constraints.
            # Wait, adding x to S does NOT make other elements *easier* to add.
            # It makes them *harder* (more sums in S).
            # So if x couldn't be added, it won't become addable later in the same phase.
            # So we only need one pass?
            # Actually, if we add x, it might block y. But if y was blocked before, it stays blocked.
            # If y was addable before, and we add x, y might become unaddable.
            # So we should try to add all currently addable elements.
            # But the set of addable elements shrinks as we add.
            # So a single pass is sufficient if we check all x.
            # However, if we check x=1, add it. Then x=2 might be blocked by x=1.
            # If we checked x=2 before x=1, we might have added it.
            # So order matters.
            # To maximize, we should try all permutations? Too slow.
            # Let's stick to one pass per "state".
            pass

    # Local Search / Kick Strategy
    # If stuck, remove an element and try again.
    
    # To speed up, we can precompute "conflicts" but let's rely on can_add speed.
    # N=128, can_add is ~16k ops. 90s / 16k ops is huge number of checks.
    # But Python overhead is high. ~1000 checks/sec? 90k checks.
    # Each kick involves 1 removal + 128 can_add checks (worst case).
    # So ~700 kicks.
    # 700 kicks is small. We need more.
    
    # Optimization: 
    # Instead of removing one element, remove a few?
    # Or, use a smarter removal: remove element that is part of the most "blocked" additions?
    
    # Let's try a different approach: Simulated Annealing on the set itself?
    # State: bitmask.
    # Move: Flip a bit.
    # Cost: Number of violations.
    # We want 0 violations and max size.
    # This is complex to implement correctly and fast.
    
    # Let's stick to Greedy + Random Kick but make it faster.
    # We can parallelize? No.
    
    # Let's try to improve the greedy phase by trying different orders.
    # Restart with different initial orders of small numbers.
    
    def solve_with_order(order):
        nonlocal best_size, best_mem
        # Reset to base
        inS_local = bytearray(N + 1)
        mem_local = []
        for i in range(N // 2 + 1, N + 1):
            inS_local[i] = 1
            mem_local.append(i)
            
        # Greedy add in given order
        for x in order:
            if time_left() < 5.0:
                break
            if inS_local[x]:
                continue
            if can_add(inS_local, x):
                inS_local[x] = 1
                mem_local.append(x)
                
        # Local search kicks
        # Try to improve by removing one and adding more
        for _ in range(20):
            if time_left() < 5.0:
                break
            # Find an element to remove
            # Prefer removing small elements? Or random?
            # Let's try removing a random element from the lower half (added ones)
            candidates = [x for x in mem_local if x <= N // 2]
            if not candidates:
                break
            rm = rng.choice(candidates)
            inS_local[rm] = 0
            mem_local.remove(rm)
            
            # Try to add elements again
            improved = False
            for x in range(1, N + 1):
                if inS_local[x]:
                    continue
                if can_add(inS_local, x):
                    inS_local[x] = 1
                    mem_local.append(x)
                    improved = True
            if len(mem_local) > best_size:
                best_size = len(mem_local)
                best_mem = list(mem_local)
            if not improved:
                # Restore? Or keep trying?
                # If we removed rm and couldn't add anything back, we lost an element.
                # We should probably restore rm to not degrade, or try another removal.
                # But maybe removing rm allows adding *two* elements later?
                # We only did one pass. Let's do another pass.
                for x in range(1, N + 1):
                    if inS_local[x]:
                        continue
                    if can_add(inS_local, x):
                        inS_local[x] = 1
                        mem_local.append(x)
                        improved = True
                if not improved:
                    # Still stuck or worse
                    # Restore rm
                    if can_add(inS_local, rm):
                        inS_local[rm] = 1
                        mem_local.append(rm)
                    # Else, we are in a smaller set. Try next kick.
                    # Actually, if we can't restore rm, we are in a smaller set.
                    # This is bad. We should only accept kicks that don't reduce size, 
                    # or track best.
                    pass

    # Generate several orders
    small = list(range(1, N // 2 + 1))
    
    # Order 1: Ascending
    orders = [list(small)]
    # Order 2: Descending
    orders.append(list(reversed(small)))
    # Order 3-20: Random
    for _ in range(19):
        o = list(small)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time_left() < 10.0:
            break
        solve_with_order(o)

    # Final check
    if not full_valid(best_mem):
        # Fallback to base
        best_mem = [i for i in range(N // 2 + 1, N + 1)]
        
    return [int(x) for x in best_mem]
