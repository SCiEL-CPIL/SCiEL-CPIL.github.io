    def local_search(S, max_iterations=50):
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
            
            # 1. Greedy add
            added = False
            for x in range(1, 257):
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    added = True
                    break
            
            if added:
                continue

            # 2. Random swap to escape local minimum
            if S:
                a = random.choice(list(S))
                S.remove(a)
                # Try to add anything
                found = False
                for x in range(1, 257):
                    if x in S:
                        continue
                    if can_add(S, x):
                        S.add(x)
                        found = True
                        break
                if not found:
                    # Revert
                    S.add(a)
        
        return S
