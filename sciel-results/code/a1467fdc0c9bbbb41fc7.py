import random
import time
from itertools import combinations

def construct():
    # Deadline guard - evaluator kills at 90s, use 85s budget
    start_time = time.time()
    deadline = start_time + 85.0

    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_numbers = list(range(2, 41))

    def is_valid_set(S):
        """Check if all subset products are distinct."""
        S_sorted = sorted(S)
        prods = {1}
        for x in S_sorted:
            new_prods = []
            for p in prods:
                np = p * x
                if np in prods:
                    return False
                new_prods.append(np)
            prods.update(new_prods)
        return True

    # Hardcoded good start: 17 elements
    candidate_17 = [4, 5, 7, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37]
    best_result = candidate_17[:]
    best_size = 17

    # Verify the hardcoded set
    if not is_valid_set(best_result):
        # Fallback to a smaller valid set
        best_result = primes[:]
        best_size = len(best_result)

    # Try to find 18 by adding one more number to candidate_17
    candidates_to_add = [x for x in range(2, 41) if x not in candidate_17]
    for c in candidates_to_add:
        if time.time() > deadline:
            break
        new_set = candidate_17 + [c]
        if is_valid_set(new_set):
            best_result = new_set
            best_size = 18
            break

    if best_size >= 18:
        return list(best_result)

    # Try to find 18 by swapping: remove one element, add one or two others
    # For each element in candidate_17, try removing it and adding two new elements
    for i, rem in enumerate(candidate_17):
        if time.time() > deadline:
            break
        base = candidate_17[:i] + candidate_17[i+1:]
        # Try adding pairs from remaining numbers
        remaining = [x for x in range(2, 41) if x not in base]
        for a in remaining:
            if time.time() > deadline:
                break
            for b in remaining:
                if b <= a:
                    continue
                new_set = base + [a, b]
                if is_valid_set(new_set):
                    best_result = new_set
                    best_size = 18
                    break
            if best_size == 18:
                break
        if best_size == 18:
            break

    if best_size >= 18:
        return list(best_result)

    # Randomized local search with more restarts and better strategy
    random.seed(42)

    def try_add(S, P, x):
        """Try to add x to S (list) with product set P (set). Returns (new_S, new_P) or None."""
        if x in P:
            return None
        for p in P:
            if x * p in P:
                return None
        new_P = set(P)
        for p in P:
            new_P.add(x * p)
        return S + [x], new_P

    def build_products(S):
        """Build the set of all subset products of S."""
        P = {1}
        for x in S:
            new_P = set(P)
            for p in P:
                new_P.add(x * p)
            P = new_P
        return P

    # More extensive randomized search
    num_restarts = 2000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Strategy: start from candidate_17, remove 1-3 random elements, 
        # then greedily add back best candidates
        S = candidate_17[:]
        P = build_products(S)
        
        # Remove 1-3 random elements
        num_remove = random.randint(1, 3)
        remove_indices = random.sample(range(len(S)), min(num_remove, len(S)))
        remove_indices.sort(reverse=True)
        for idx in remove_indices:
            S.pop(idx)
        P = build_products(S)

        # Try to add elements greedily
        remaining = [x for x in range(2, 41) if x not in S]
        random.shuffle(remaining)
        
        for c in remaining:
            if time.time() > deadline:
                break
            res = try_add(S, P, c)
            if res:
                S, P = res

        if len(S) > best_size:
            best_size = len(S)
            best_result = S[:]
            if best_size == 18:
                break

    return list(best_result)
