def is_valid(S):
    if not S: return True
    L = LCM
    weights = {i: L//i for i in S}
    for i in S:
        others = [j for j in S if j != i]
        # Check if weights[i] is a subset sum of weights[others]
        # Use meet-in-the-middle or simple set expansion
        # Since |others| <= 23, 2^23 is too big for simple expansion.
        # But we can use the property that we just need to check ONE target.
        # We can use a set of sums, but prune if sum > target?
        # Weights are positive.
        # Sort others by weight descending?
        # Actually, simple set expansion works if we stop early.
        sums = {0}
        for j in others:
            w = weights[j]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns == weights[i]:
                    return False
                if ns < weights[i]: # Only keep sums smaller than target?
                    # No, we need to know if we CAN reach target.
                    # If all weights are positive, we can prune sums > target.
                    new_sums.add(ns)
            # Also keep old sums?
            # We are building subset sums.
            # Standard subset sum DP.
            # Since we only care if target is reachable, we can use a set.
            # Optimization: if target is small, prune large sums.
            # Here target is L/i. Others are L/j.
            # L/i is large.
            # Let's just use a set and prune sums > target.
            sums.update(new_sums)
            # Prune
            sums = {s for s in sums if s <= weights[i]}
            if weights[i] in sums:
                return False
    return True
