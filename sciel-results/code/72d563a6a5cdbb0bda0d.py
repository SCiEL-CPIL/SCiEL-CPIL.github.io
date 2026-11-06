def construct():
        # Check the specific candidate
        cand = [5, 6, 7, 8, 9, 12, 230, 57960]
        # Verify sum
        s = sum(1/x for x in cand)
        if abs(s - 1.0) < 1e-10:
            return cand
        # Fallback: return a valid solution found by DP (slower but safe)
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
        raise RuntimeError("no baseline representation")
