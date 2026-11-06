"""New program using sign-based partitioning and greedy refinement."""

def construct():
    import random

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def max_sum_free(vals):
        A_set = set(vals)
        
        # Identify structural sum-free subsets
        max_val = max(vals)
        min_val = min(vals)
        
        # S_pos: x > max_val / 2
        # If x, y > max_val/2, then x+y > max_val >= max(A), so x+y not in A.
        S_pos = [x for x in vals if x > 0 and x > max_val / 2.0]
        
        # S_neg: x < min_val / 2
        # If x, y < min_val/2, then x+y < min_val <= min(A), so x+y not in A.
        S_neg = [x for x in vals if x < 0 and x < min_val / 2.0]
        
        def refine(base_set, candidates):
            S = set(base_set)
            # Sort candidates to try "safer" ones first?
            # Elements with larger absolute value are "safer" (less likely to sum to something in S)
            # Actually, elements close to the boundary of S are more likely to conflict.
            # Let's just use the given order (sorted ascending).
            for x in candidates:
                if x in S:
                    continue
                # Check validity
                if 2 * x in S:
                    continue
                valid = True
                for y in S:
                    if x + y in S:
                        valid = False
                        break
                if valid:
                    S.add(x)
            return S

        # Generate candidates for refinement
        # We want to add elements that are "compatible" with the base set.
        
        # Refine S_pos
        # Candidates: all elements not in S_pos
        cands_pos = [x for x in vals if x not in S_pos]
        S_pos_refined = refine(S_pos, cands_pos)
        
        # Refine S_neg
        cands_neg = [x for x in vals if x not in S_neg]
        S_neg_refined = refine(S_neg, cands_neg)
        
        # Also consider a "Zero-avoiding" greedy?
        # The structural sets avoid the "middle" where sums land.
        
        best = max([S_pos_refined, S_neg_refined], key=len)
        
        return list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        subset = max_sum_free(vals)
        results.append(subset)
        
    return results
