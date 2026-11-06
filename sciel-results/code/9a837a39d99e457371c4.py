def construct():
    import math
    import random
    
    n = 256
    squares = set()
    i = 1
    while i * i <= 512:
        squares.add(i * i)
        i += 1
    
    # conflicts[x] = set of y such that x+y is a perfect square
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in squares:
            b = s - a
            if a <= b <= n:
                conflicts[a].add(b)
                if b != a:
                    conflicts[b].add(a)
    
    def is_valid(s):
        for a in s:
            for b in s:
                if a <= b and a + b in squares:
                    return False
        return True
    
    # Greedy with multiple orderings
    best_set = set()
    
    for trial in range(100):
        random.seed(trial)
        available = set(range(1, n + 1))
        selected = set()
        
        while available:
            # Pick element with fewest conflicts in available, with random tie-breaking
            min_deg = float('inf')
            candidates = []
            for x in available:
                deg = len(conflicts[x] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [x]
                elif deg == min_deg:
                    candidates.append(x)
            best = random.choice(candidates)
            selected.add(best)
            available.discard(best)
            for c in conflicts[best]:
                available.discard(c)
        
        # Local search: try to add elements
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x not in selected and not (conflicts[x] & selected):
                    selected.add(x)
                    improved = True
        
        # Local search: try swap moves
        improved = True
        while improved:
            improved = False
            for remove in list(selected):
                candidate = selected - {remove}
                # Try to add as many as possible
                addable = []
                for x in range(1, n + 1):
                    if x not in candidate and not (conflicts[x] & candidate):
                        addable.append(x)
                # Greedily add
                new_set = set(candidate)
                for x in sorted(addable, key=lambda y: len(conflicts[y] & new_set)):
                    if not (conflicts[x] & new_set):
                        new_set.add(x)
                if len(new_set) > len(selected):
                    selected = new_set
                    improved = True
                    break
        
        if len(selected) > len(best_set):
            best_set = selected
    
    return sorted(best_set)
