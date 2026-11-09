<<<<<<< SEARCH
    def local_search(S, max_time):
        """Try to improve S by removing one element and adding one or more."""
        S = set(S)
        best = S
        end = start_time + max_time
        while time.time() < end:
            improved = False
            L = sorted(S)
            for i in range(len(L)):
                if time.time() > end:
                    break
                a = L[i]
                S2 = S - {a}
                # Try adding elements not in S2, in sorted order for determinism
                added_any = False
                for candidate in range(1, 257):
                    if candidate in S2:
                        continue
                    if can_add(S2, candidate):
                        S2.add(candidate)
                        added_any = True
                if len(S2) > len(best):
                    best = S2
                    S = set(S2)
                    improved = True
                    break
            if not improved:
                break
        return best
=======
    def local_search(S, max_time):
        """Try to improve S by removing one or two elements and adding one or more."""
        S = set(S)
        best = S
        best_size = len(S)
        end = start_time + max_time
        while time.time() < end:
            improved = False
            L = sorted(S)
            # Try single removals
            for i in range(len(L)):
                if time.time() > end:
                    break
                a = L[i]
                S2 = S - {a}
                # Try adding elements not in S2, prioritized by conflict count
                candidates = [c for c in range(1, 257) if c not in S2]
                # Sort by conflict count (ascending) - fewer conflicts first
                candidates.sort(key=lambda c: conflict_count(S2, c))
                for candidate in candidates:
                    if can_add(S2, candidate):
                        S2.add(candidate)
                if len(S2) > best_size:
                    best = set(S2)
                    best_size = len(S2)
                    S = set(S2)
                    improved = True
                    break
            if improved:
                continue
            # Try double removals if single didn't help
            if len(L) >= 2:
                for i in range(len(L)):
                    if time.time() > end:
                        break
                    for j in range(i+1, len(L)):
                        if time.time() > end:
                            break
                        a, b = L[i], L[j]
                        S2 = S - {a, b}
                        # Try adding elements not in S2
                        candidates = [c for c in range(1, 257) if c not in S2]
                        candidates.sort(key=lambda c: conflict_count(S2, c))
                        for candidate in candidates:
                            if can_add(S2, candidate):
                                S2.add(candidate)
                        if len(S2) > best_size:
                            best = set(S2)
                            best_size = len(S2)
                            S = set(S2)
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return best
>>>>>>> REPLACE

