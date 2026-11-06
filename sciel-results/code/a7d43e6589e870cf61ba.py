"""Rewritten program using a greedy heuristic for sum-free subsets."""

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
        # Sort in descending order to prioritize large numbers
        sorted_vals = sorted(vals, reverse=True)
        S = set()
        sums = set()
        diffs = set()
        
        for x in sorted_vals:
            # Check if x can be added to S
            # 1. x must not be a sum of two elements in S
            if x in sums:
                continue
            # 2. x must not be a difference of two elements in S
            if x in diffs:
                continue
            # 3. 2*x must not be in S (to avoid x+x=2x)
            if 2 * x in S:
                continue
            
            # Add x to S
            S.add(x)
            sums.add(2 * x)
            
            # Update sums and diffs with the new element x
            # Iterate over existing elements in S (excluding x itself for diffs, 
            # but x is already added to S, so we skip x in the loop)
            for y in S:
                if y == x:
                    continue
                sums.add(x + y)
                diffs.add(x - y)
                diffs.add(y - x)
                
        return list(S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        subset = max_sum_free(vals)
        results.append(subset)
        
    return results
