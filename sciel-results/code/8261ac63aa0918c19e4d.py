import time
import random

def construct():
    n = 256
    max_size = 64
    
    def check_sidon(A):
        # Check if A is a Sidon set
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def check_diffs_disjoint(A, B):
        # Check if nonzero diffs of A and B are disjoint
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[i] - A[j]
                    diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[i] - B[j]
                    diffs_B.add(d)
                    
        return diffs_A.isdisjoint(diffs_B)

    def validate(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        if not check_sidon(A):
            return False
        if not check_sidon(B):
            return False
        if not check_diffs_disjoint(A, B):
            return False
        return True

    def greedy_once(seed):
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Check if x can be added to A
            can_add_A = True
            if len(A) < max_size:
                if 2 * x in sums_A:
                    can_add_A = False
                else:
                    for a in A:
                        if (a + x) in sums_A:
                            can_add_A = False
                            break
                        d = x - a
                        if d != 0 and d in diffs_B:
                            can_add_A = False
                            break
            else:
                can_add_A = False
            
            # Check if x can be added to B
            can_add_B = True
            if len(B) < max_size:
                if 2 * x in sums_B:
                    can_add_B = False
                else:
                    for b in B:
                        if (b + x) in sums_B:
                            can_add_B = False
                            break
                        d = x - b
                        if d != 0 and d in diffs_A:
                            can_add_B = False
                            break
            else:
                can_add_B = False
            
            # Decide where to add
            # Heuristic: Prefer adding to the set that is smaller to balance sizes?
            # Or prefer adding to Both if possible?
            
            if can_add_A and can_add_B:
                # Check if adding to both is valid (diffs of new A vs new B)
                # If we add to A, new diffs are {x-a, a-x}.
                # If we add to B, new diffs are {x-b, b-x}.
                # We need new_diffs_A disjoint from old_diffs_B AND new_diffs_B disjoint from old_diffs_A.
                # Also new_diffs_A disjoint from new_diffs_B? No, only A vs B.
                # But if we add to both, we must ensure:
                # 1. New diffs from A (involving x) do not conflict with existing B diffs. (Checked in can_add_A)
                # 2. New diffs from B (involving x) do not conflict with existing A diffs. (Checked in can_add_B)
                # 3. Do new diffs from A (involving x) conflict with new diffs from B (involving x)?
                #    The task says "Their nonzero difference sets must be disjoint".
                #    This implies Diff(A) ∩ Diff(B) = ∅.
                #    If we add x to both, we must check if any d in {x-a} conflicts with any d' in {x-b}.
                #    Actually, if x is in both, the difference x-x=0 is ignored.
                #    But differences like (x-a) and (b-x) might be equal?
                #    If x-a = b-x => 2x = a+b.
                #    If this happens, then the difference is in both sets.
                #    So we must check this specific condition.
                
                valid_both = True
                for a in A:
                    d1 = x - a
                    if d1 == 0: continue
                    for b in B:
                        d2 = b - x
                        if d2 == 0: continue
                        if d1 == d2:
                            valid_both = False
                            break
                    if not valid_both:
                        break
                        # Also check d1 == x-b? No, d2 is b-x. 
                        # Diff sets contain both d and -d usually? 
                        # "Nonzero difference sets". Usually defined as {a-b | a,b in S, a!=b}.
                        # If d1 = x-a and d2 = x-b, they are in the sets if a!=x, b!=x.
                        # If d1 = d2, conflict.
                        # If d1 = -d2, i.e., x-a = x-b => a=b. If a=b, and x is added to both, 
                        # then diff x-a is in A, and a-x is in B. They are distinct unless 0.
                        # Wait, if a is in A and B, and we add x to A and B.
                        # Diff in A: x-a. Diff in B: x-a. Conflict!
                        # So if a in A ∩ B, we cannot add x to both.
                        # The check above `if d1 == d2` covers x-a = x-b => a=b.
                        # But we also need to check if x-a (in A) conflicts with b-x (in B)?
                        # d1 = x-a. d2' = b-x.
                        # If x-a = b-x => 2x = a+b.
                        # My loop above only checked d1 vs d2 (where d2 was b-x).
                        # Let's re-verify the conflict condition for "Both".
                        # Diff_A_new contains {x-a | a in A} and {a-x | a in A}.
                        # Diff_B_new contains {x-b | b in B} and {b-x | b in B}.
                        # We need Diff_A_total ∩ Diff_B_total = ∅.
                        # Existing diffs are already disjoint.
                        # We need:
                        # 1. {x-a} ∩ Diff_B_old = ∅ (Checked in can_add_A)
                        # 2. {a-x} ∩ Diff_B_old = ∅ (Symmetric to 1, since Diff_B_old is symmetric? 
                        #    If d in Diff_B, -d in Diff_B. Yes. So checking d in Diff_B is enough.)
                        # 3. {x-b} ∩ Diff_A_old = ∅ (Checked in can_add_B)
                        # 4. {b-x} ∩ Diff_A_old = ∅ (Symmetric)
                        # 5. {x-a} ∩ {x-b} = ∅ ? 
                        #    x-a = x-b => a=b. If a=b, then x-a is in Diff_A and x-a is in Diff_B. Conflict.
                        # 6. {x-a} ∩ {b-x} = ∅ ?
                        #    x-a = b-x => 2x = a+b. Conflict.
                        # 7. {a-x} ∩ {x-b} = ∅ ?
                        #    a-x = x-b => a+b = 2x. Same as 6.
                        # 8. {a-x} ∩ {b-x} = ∅ ?
                        #    a-x = b-x => a=b. Same as 5.
                        
                        # So we need to check:
                        # - If any a in A equals any b in B, we cannot add x to both (Case 5/8).
                        # - If any a in A and b in B satisfy a+b = 2x, we cannot add x to both (Case 6/7).
                        
                        if A and B:
                            # Check intersection
                            if set(A).intersection(B):
                                valid_both = False
                            else:
                                # Check a+b=2x
                                # We can just iterate
                                for a in A:
                                    for b in B:
                                        if a + b == 2 * x:
                                            valid_both = False
                                            break
                                    if not valid_both:
                                        break
                            
                if valid_both:
                    # Add to both
                    # Update A
                    for a in A:
                        sums_A.add(a + x)
                        d = x - a
                        if d != 0:
                            diffs_A.add(d)
                            diffs_A.add(-d)
                    sums_A.add(2 * x)
                    A.append(x)
                    
                    # Update B
                    for b in B:
                        sums_B.add(b + x)
                        d = x - b
                        if d != 0:
                            diffs_B.add(d)
                            diffs_B.add(-d)
                    sums_B.add(2 * x)
                    B.append(x)
                else:
                    # Cannot add to both. Choose one.
                    # Heuristic: Add to the one with fewer elements?
                    if len(A) <= len(B):
                        # Add to A
                        for a in A:
                            sums_A.add(a + x)
                            d = x - a
                            if d != 0:
                                diffs_A.add(d)
                                diffs_A.add(-d)
                        sums_A.add(2 * x)
                        A.append(x)
                    else:
                        # Add to B
                        for b in B:
                            sums_B.add(b + x)
                            d = x - b
                            if d != 0:
                                diffs_B.add(d)
                                diffs_B.add(-d)
                        sums_B.add(2 * x)
                        B.append(x)
                        
            elif can_add_A:
                for a in A:
                    sums_A.add(a + x)
                    d = x - a
                    if d != 0:
                        diffs_A.add(d)
                        diffs_A.add(-d)
                sums_A.add(2 * x)
                A.append(x)
                
            elif can_add_B:
                for b in B:
                    sums_B.add(b + x)
                    d = x - b
                    if d != 0:
                        diffs_B.add(d)
                        diffs_B.add(-d)
                sums_B.add(2 * x)
                B.append(x)
                
        return A, B

    start_time = time.time()
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    seed = 42
    while time.time() - start_time < 55:
        A, B = greedy_once(seed)
        if validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
        seed += 1
        
    return [best_A, best_B]
