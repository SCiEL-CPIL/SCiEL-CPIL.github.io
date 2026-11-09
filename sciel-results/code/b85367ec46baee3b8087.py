"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    best_total = 0
    best_A = None
    best_B = None
    
    # Increase restarts due to more efficient selection logic
    for seed in range(1000):
        rng = random.Random(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        while candidates:
            best_candidates = []
            min_options = 3
            
            for x in candidates:
                can_A = True
                can_B = True
                
                # Check A
                if x + x in sums_A:
                    can_A = False
                else:
                    for s in A:
                        if x + s in sums_A:
                            can_A = False
                            break
                if can_A:
                    for s in A:
                        d1 = x - s
                        d2 = s - x
                        if d1 in diffs_B or d2 in diffs_B:
                            can_A = False
                            break
                            
                # Check B
                if x + x in sums_B:
                    can_B = False
                else:
                    for s in B:
                        if x + s in sums_B:
                            can_B = False
                            break
                if can_B:
                    for s in B:
                        d1 = x - s
                        d2 = s - x
                        if d1 in diffs_A or d2 in diffs_A:
                            can_B = False
                            break
                            
                options = (1 if can_A else 0) + (1 if can_B else 0)
                if options == 0:
                    continue
                    
                if options < min_options:
                    min_options = options
                    best_candidates = [(x, can_A, can_B)]
                elif options == min_options:
                    best_candidates.append((x, can_A, can_B))
                    
            if not best_candidates:
                break
                
            # Pick randomly among the most constrained
            best_x, best_can_A, best_can_B = rng.choice(best_candidates)
            
            # Remove best_x from candidates
            # Optimization: Since we are in a loop, we can just filter it out next time
            # But to keep the list small, we remove it now
            # This is O(N) but N is small
            candidates.remove(best_x)
            
            # Choose set
            if best_can_A and best_can_B:
                # Balance sizes to prevent one set from getting too large and blocking the other
                if len(A) < len(B):
                    target = A
                elif len(B) < len(A):
                    target = B
                else:
                    target = A if rng.random() < 0.5 else B
            elif best_can_A:
                target = A
            else:
                target = B
                
            # Add to target
            if target is A:
                A.append(best_x)
                sums_A.add(best_x + best_x)
                for s in A[:-1]:
                    sums_A.add(best_x + s)
                    diffs_A.add(best_x - s)
                    diffs_A.add(s - best_x)
            else:
                B.append(best_x)
                sums_B.add(best_x + best_x)
                for s in B[:-1]:
                    sums_B.add(best_x + s)
                    diffs_B.add(best_x - s)
                    diffs_B.add(s - best_x)
                    
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
# EVOLVE-BLOCK-END
