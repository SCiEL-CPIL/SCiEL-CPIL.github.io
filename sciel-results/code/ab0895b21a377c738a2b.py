"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_list, S_set, x):
        """Check if adding x to S keeps it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_list:
            if a + x in S_set:
                return False
        return True

    def greedy_augment(S, all_elements, order=None):
        """Try to add elements from all_elements not in S, in given order."""
        S_set = set(S)
        S_list = list(S)
        if order is None:
            order = all_elements
        for x in order:
            if x not in S_set:
                if can_add(S_list, S_set, x):
                    S_set.add(x)
                    S_list.append(x)
        return S_list

    def solve_one(vals):
        """Find a large sum-free subset of vals."""
        n = len(vals)
        candidates = []

        # Candidate 1: all odd numbers (always sum-free)
        odd = [x for x in vals if x % 2 != 0]
        candidates.append(odd)

        # Candidate 2: largest positives
        positives = sorted([x for x in vals if x > 0], reverse=True)
        if positives:
            candidates.append(positives[:max(1, len(positives)//2)])

        # Candidate 3: smallest negatives
        negatives = sorted([x for x in vals if x < 0])
        if negatives:
            candidates.append(negatives[:max(1, len(negatives)//2)])

        # Candidate 4: top third by absolute value
        abs_sorted = sorted(vals, key=abs, reverse=True)
        candidates.append(abs_sorted[:max(1, n//3)])

        # Candidate 5: large positive by abs
        pos_abs = sorted([x for x in vals if x > 0], key=abs, reverse=True)
        if pos_abs:
            candidates.append(pos_abs[:max(1, len(pos_abs)//2)])

        # Candidate 6: large negative by abs
        neg_abs = sorted([x for x in vals if x < 0], key=abs, reverse=True)
        if neg_abs:
            candidates.append(neg_abs[:max(1, len(neg_abs)//2)])

        # Candidate 7: all positive
        pos_only = [x for x in vals if x > 0]
        candidates.append(pos_only)

        # Candidate 8: all negative
        neg_only = [x for x in vals if x < 0]
        candidates.append(neg_only)

        # Filter to sum-free candidates
        valid_candidates = []
        for c in candidates:
            if is_sum_free(c):
                valid_candidates.append(c)

        if not valid_candidates:
            valid_candidates = [odd]

        # Try multiple augmentation orders
        orders = [
            vals,  # sorted ascending
            sorted(vals, reverse=True),  # sorted descending
            sorted(vals, key=abs, reverse=True),  # by abs value
            sorted(vals, key=abs),  # by abs value ascending
        ]

        best = []
        for c in valid_candidates:
            for order in orders:
                aug = greedy_augment(c, vals, order)
                if len(aug) > len(best):
                    best = aug

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        vals = instance(size, seed)
        result = solve_one(vals)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
