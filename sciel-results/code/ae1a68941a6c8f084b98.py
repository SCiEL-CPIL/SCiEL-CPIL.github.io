"""Rewritten program with stronger sum-free subset search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(subset_set, x):
        """Check if x can be added to subset maintaining sum-free."""
        for a in subset_set:
            if a + x in subset_set:
                return False
        if 2 * x in subset_set:
            return False
        return True

    def greedy_sum_free(elements):
        """Greedy: add elements in given order, keep if sum-free."""
        result = []
        result_set = set()
        for x in elements:
            if can_add(result_set, x):
                result.append(x)
                result_set.add(x)
        return result

    def local_search_improve(subset, elements, max_rounds=200):
        """Improve subset by repeatedly adding any element that fits."""
        s = list(subset)
        s_set = set(s)
        
        for _ in range(max_rounds):
            added_any = False
            for x in elements:
                if x in s_set:
                    continue
                if can_add(s_set, x):
                    s.append(x)
                    s_set.add(x)
                    added_any = True
            if not added_any:
                break
        
        return s

    def solve_one(elements):
        """Solve one instance."""
        n = len(elements)
        candidates = []
        
        # 1. Odd numbers
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Greedy by absolute value descending
        by_abs = sorted(elements, key=lambda x: -abs(x))
        candidates.append(greedy_sum_free(by_abs))
        
        # 3. Greedy by value ascending
        candidates.append(greedy_sum_free(sorted(elements)))
        
        # 4. Greedy by value descending
        candidates.append(greedy_sum_free(sorted(elements, reverse=True)))
        
        # 5. Random orderings
        rng = random.Random(999)
        for i in range(15):
            shuffled = list(elements)
            rng.shuffle(shuffled)
            candidates.append(greedy_sum_free(shuffled))
        
        # 6. Positive only
        candidates.append([x for x in elements if x > 0])
        
        # 7. Negative only
        candidates.append([x for x in elements if x < 0])
        
        # 8. Large absolute values
        if elements:
            max_abs = max(abs(x) for x in elements)
            threshold = max_abs * 0.5
            candidates.append([x for x in elements if abs(x) > threshold])
        
        # Improve each candidate via local search
        improved = []
        for c in candidates:
            if not c:
                continue
            ic = local_search_improve(c, elements, max_rounds=100)
            improved.append(ic)
        
        if not improved:
            return []
        
        best = max(improved, key=len)
        return best

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                results.append([])
            return results
        
        elements = instance(size, seed)
        result = solve_one(elements)
        results.append(result)
    
    return results

# EVOLVE-BLOCK-END
