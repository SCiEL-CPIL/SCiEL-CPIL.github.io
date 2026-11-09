    import random

    def construct():
        n = 256
        best_total = 0
        best_A = None
        best_B = None
        
        for seed in range(300):
            rng = random.Random(seed)
            A = []
            B = []
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            
            elements = list(range(1, n+1))
            rng.shuffle(elements)
            
            def can_add_A(x):
                if x + x in sums_A: return False
                for s in A:
                    if x + s in sums_A: return False
                for s in A:
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_B or d2 in diffs_B: return False
                return True

            def can_add_B(x):
                if x + x in sums_B: return False
                for s in B:
                    if x + s in sums_B: return False
                for s in B:
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_A or d2 in diffs_A: return False
                return True

            def add_to_A(x):
                A.append(x)
                sums_A.add(x + x)
                diffs_A.add(0) # Not needed, but for consistency? No, diffs are non-zero.
                for s in A[:-1]:
                    sums_A.add(x + s)
                    diffs_A.add(x - s)
                    diffs_A.add(s - x)

            def add_to_B(x):
                B.append(x)
                sums_B.add(x + x)
                for s in B[:-1]:
                    sums_B.add(x + s)
                    diffs_B.add(x - s)
                    diffs_B.add(s - x)
            
            # Greedy Phase
            for x in elements:
                cA = can_add_A(x)
                cB = can_add_B(x)
                if cA and cB:
                    if rng.random() < 0.5:
                        add_to_A(x)
                    else:
                        add_to_B(x)
                elif cA:
                    add_to_A(x)
                elif cB:
                    add_to_B(x)
            
            # Local Search Phase
            # Try to add more
            improved = True
            while improved:
                improved = False
                for x in elements:
                    if x in A or x in B: continue
                    if can_add_A(x):
                        add_to_A(x)
                        improved = True
                        break
                    if can_add_B(x):
                        add_to_B(x)
                        improved = True
                        break
            
            # Try Swaps/Moves to unlock additions?
            # Simple: Try to move an element from A to B or vice versa if it doesn't reduce size,
            # and see if it allows an addition.
            # This is complex. Let's just do multiple restarts with different shuffles.
            
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A[:]
                best_B = B[:]
        
        return [best_A, best_B]
    