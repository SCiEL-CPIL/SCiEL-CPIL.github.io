def construct():
    from math import lcm
    from itertools import combinations
    import random
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        lab = lcm(a, b)
        lac = lcm(a, c)
        lbc = lcm(b, c)
        if lab == lac == lbc:
            bad_triples.append((a, b, c))
    
    # For each pair, store the set of third elements that form a bad triple
    pair_to_thirds = {}
    for a, b, c in bad_triples:
        for pair in [(a, b), (a, c), (b, c)]:
            key = (min(pair), max(pair))
            if key not in pair_to_thirds:
                pair_to_thirds[key] = set()
            other = a + b + c - pair[0] - pair[1]
            pair_to_thirds[key].add(other)
    
    def is_valid_triple(a, b, c):
        """Check if adding c to {a,b} creates a bad triple"""
        key = (min(a, b), max(a, b))
        if key in pair_to_thirds:
            return c in pair_to_thirds[key]
        return False
    
    def can_add(x, selected):
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                if is_valid_triple(a, b, x):
                    return False
        return True
    
    # Greedy: add numbers in order, skip if conflict
    def greedy_construct(order):
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    # Try multiple orderings
    best = []
    
    # Natural order
    s1 = greedy_construct(numbers)
    if len(s1) > len(best):
        best = s1
    
    # Reverse order
    s2 = greedy_construct(numbers[::-1])
    if len(s2) > len(best):
        best = s2
    
    # Random orders
    random.seed(42)
    for _ in range(50):
        order = numbers[:]
        random.shuffle(order)
        s = greedy_construct(order)
        if len(s) > len(best):
            best = s
    
    # Local search: try to swap elements to improve
    def local_search(selected, iterations=1000):
        selected_set = set(selected)
        improved = True
        it = 0
        while improved and it < iterations:
            improved = False
            it += 1
            # Try removing one element and adding another
            for remove in list(selected_set):
                candidate_set = selected_set - {remove}
                for add in numbers:
                    if add in candidate_set:
                        continue
                    # Check if add can be added to candidate_set
                    ok = True
                    lst = list(candidate_set)
                    for i in range(len(lst)):
                        for j in range(i + 1, len(lst)):
                            if is_valid_triple(lst[i], lst[j], add):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok and len(candidate_set) + 1 > len(selected_set):
                        selected_set = candidate_set | {add}
                        improved = True
                        break
                if improved:
                    break
        return list(selected_set)
    
    best = local_search(best)
    
    return best
