import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    N = 256
    max_size_limit = 64
    
    def can_add_to_A(x, A, sums_A, diffs_B):
        if (sums_A >> (2 * x)) & 1:
            return False
        for a in A:
            if (sums_A >> (x + a)) & 1:
                return False
        for a in A:
            d = abs(x - a)
            if d > 0 and ((diffs_B >> d) & 1):
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if (sums_B >> (2 * x)) & 1:
            return False
        for a in B:
            if (sums_B >> (x + a)) & 1:
                return False
        for a in B:
            d = abs(x - a)
            if d > 0 and ((diffs_A >> d) & 1):
                return False
        return True

    def greedy_init():
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add_to_A(x, A, sums_A, diffs_B):
                A.append(x)
                sums_A |= (1 << (2 * x))
                for a in A[:-1]:
                    sums_A |= (1 << (x + a))
                    d = abs(x - a)
                    if d > 0:
                        diffs_A |= (1 << d)
            elif can_add_to_B(x, B, sums_B, diffs_A):
                B.append(x)
                sums_B |= (1 << (2 * x))
                for a in B[:-1]:
                    sums_B |= (1 << (x + a))
                    d = abs(x - a)
                    if d > 0:
                        diffs_B |= (1 << d)
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    best_size = 0
    best_A = []
    best_B = []
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        A, B, sums_A, sums_B, diffs_A, diffs_B = greedy_init()
        current_size = len(A) + len(B)
        
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
        
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.01
        sa_deadline = time.time() + 3.0 
        
        used = set(A) | set(B)
        
        while T > min_T and time.time() < sa_deadline:
            move_type = random.random()
            
            if move_type < 0.5:
                # Insert
                unused = [x for x in range(1, N + 1) if x not in used]
                if not unused:
                    break
                x = random.choice(unused)
                
                if len(A) < max_size_limit and can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    used.add(x)
                    sums_A |= (1 << (2 * x))
                    for a in A[:-1]:
                        sums_A |= (1 << (x + a))
                        d = abs(x - a)
                        if d > 0:
                            diffs_A |= (1 << d)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_A = list(A)
                        best_B = list(B)
                    continue
                
                if len(B) < max_size_limit and can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    used.add(x)
                    sums_B |= (1 << (2 * x))
                    for a in B[:-1]:
                        sums_B |= (1 << (x + a))
                        d = abs(x - a)
                        if d > 0:
                            diffs_B |= (1 << d)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_A = list(A)
                        best_B = list(B)
                    continue

            elif move_type < 0.8:
                # Move (A<->B)
                # Try moving x from A to B
                if A:
                    x = random.choice(A)
                    # Check if x can be added to B, given A\{x}
                    # We need to check sums in B, and diffs between A\{x} and B.
                    # Also, we must ensure that removing x from A doesn't break A's Sidon property (it doesn't).
                    # And that the new diffs of A\{x} and B+x are disjoint.
                    
                    # 1. Check B + x is Sidon
                    if (sums_B >> (2 * x)) & 1:
                        pass # Reject
                    else:
                        valid_sum = True
                        for a in B:
                            if (sums_B >> (x + a)) & 1:
                                valid_sum = False
                                break
                        if valid_sum:
                            # 2. Check diffs
                            # Diffs of A\{x} and B
                            # Diffs of A\{x} and x (which is |x-a| for a in A\{x})
                            # Diffs of B and x (which is |x-b| for b in B)
                            
                            # We need to check if any diff in A\{x} is in B, or vice versa.
                            # And if any diff involving x in the new sets conflicts.
                            
                            # This is complex. Let's just try to add x to B and remove from A, 
                            # and verify the whole thing? No, too slow.
                            # Let's approximate: just check if x can be added to B ignoring A's change, 
                            # and if removing x from A is "safe" (it always is).
                            # The risk is that a diff in A\{x} conflicts with a diff in B.
                            # But diffs_A and diffs_B were disjoint. 
                            # Removing x from A removes some diffs from A. So diffs_A' is subset of diffs_A.
                            # So diffs_A' and diffs_B are still disjoint.
                            # Now we add x to B. New diffs_B' = diffs_B U {|x-b| for b in B} U {|x-x|=0}.
                            # We need diffs_A' and diffs_B' to be disjoint.
                            # This means:
                            # 1. diffs_A' and diffs_B are disjoint (True).
                            # 2. diffs_A' and {|x-b| for b in B} are disjoint.
                            # 3. diffs_B and {|x-b| for b in B} are disjoint (This is the Sidon check for B, which we did above? 
                            #    No, Sidon check for B ensures sums are unique. It doesn't ensure diffs are unique within B? 
                            #    Wait, Sidon set definition: pair sums are unique. It does NOT require diffs to be unique within the set.
                            #    So B can have repeated diffs.
                            #    So we don't need to check 3.
                            #    We only need to check 2: No diff in A\{x} equals any diff in B+x.
                            
                            valid_diffs = True
                            # Check diffs of A\{x} against new diffs of B
                            # New diffs of B are |x-b| for b in B.
                            for b in B:
                                d_new = abs(x - b)
                                if d_new == 0: continue
                                # Check if d_new is in diffs_A (which is superset of diffs_A')
                                # If d_new is in diffs_A, it might be from x. 
                                # We need to check if d_new is in diffs_A\{diffs_involving_x}.
                                # This is getting complicated.
                                
                                # Simplification: Just check if d_new is in diffs_A.
                                # If it is, it might be a conflict.
                                if (diffs_A >> d_new) & 1:
                                    # It could be a diff involving x. 
                                    # If d_new = |x-a| for some a in A, then it's not a conflict with A\{x}.
                                    # But checking this is O(|A|).
                                    # Let's just reject if d_new is in diffs_A. This is conservative but safe.
                                    valid_diffs = False
                                    break
                            
                            if valid_diffs:
                                # Perform Move
                                A.remove(x)
                                used.discard(x)
                                # Update masks for A (remove x)
                                sums_A &= ~(1 << (2 * x))
                                for a in A:
                                    sums_A &= ~(1 << (x + a))
                                    d = abs(x - a)
                                    if d > 0:
                                        diffs_A &= ~(1 << d)
                                
                                # Update masks for B (add x)
                                B.append(x)
                                used.add(x)
                                sums_B |= (1 << (2 * x))
                                for a in B[:-1]:
                                    sums_B |= (1 << (x + a))
                                    d = abs(x - a)
                                    if d > 0:
                                        diffs_B |= (1 << d)
                                # Size unchanged
                                continue
                elif B:
                    # Move B to A
                    x = random.choice(B)
                    if (sums_A >> (2 * x)) & 1:
                        pass
                    else:
                        valid_sum = True
                        for a in A:
                            if (sums_A >> (x + a)) & 1:
                                valid_sum = False
                                break
                        if valid_sum:
                            valid_diffs = True
                            for a in A:
                                d_new = abs(x - a)
                                if d_new == 0: continue
                                if (diffs_B >> d_new) & 1:
                                    valid_diffs = False
                                    break
                            
                            if valid_diffs:
                                B.remove(x)
                                used.discard(x)
                                sums_B &= ~(1 << (2 * x))
                                for b in B:
                                    sums_B &= ~(1 << (x + b))
                                    d = abs(x - b)
                                    if d > 0:
                                        diffs_B &= ~(1 << d)
                                
                                A.append(x)
                                used.add(x)
                                sums_A |= (1 << (2 * x))
                                for a in A[:-1]:
                                    sums_A |= (1 << (x + a))
                                    d = abs(x - a)
                                    if d > 0:
                                        diffs_A |= (1 << d)
                                continue
            else:
                # Delete
                if A and random.random() < math.exp(-1.0 / T):
                    x = random.choice(A)
                    A.remove(x)
                    used.discard(x)
                    sums_A &= ~(1 << (2 * x))
                    for a in A:
                        sums_A &= ~(1 << (x + a))
                        d = abs(x - a)
                        if d > 0:
                            diffs_A &= ~(1 << d)
                    current_size -= 1
                elif B and random.random() < math.exp(-1.0 / T):
                    x = random.choice(B)
                    B.remove(x)
                    used.discard(x)
                    sums_B &= ~(1 << (2 * x))
                    for b in B:
                        sums_B &= ~(1 << (x + b))
                        d = abs(x - b)
                        if d > 0:
                            diffs_B &= ~(1 << d)
                    current_size -= 1
            
            T *= cooling_rate

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
