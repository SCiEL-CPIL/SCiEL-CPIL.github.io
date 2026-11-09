import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    n = 180
    # Precompute all forbidden triples (a, b, c) with a < b, c = ab/(a+b), c < a, all distinct
    # 1/a = 1/b + 1/c => 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0 => (a-b)(a-c) = a^2
    # Or simply: c = ab / (a+b)
    # We need a, b, c distinct integers in [1, n]
    # Let's generate triples (a, b, c) such that 1/a = 1/b + 1/c
    # This implies c = ab/(a+b).
    # For c to be integer, a+b must divide ab.
    # Let g = gcd(a,b). a=ga, b=gb, gcd(a,b)=1.
    # c = g^2 ab / (g(a+b)) = gab / (a+b).
    # So a+b must divide gab. Since gcd(a+b, a)=gcd(b,a)=1 and gcd(a+b, b)=1,
    # a+b must divide g.
    # Let g = k(a+b). Then a = k(a+b)a', b = k(a+b)b'.
    # This parameterization is complex. Simpler: just iterate a, b.
    
    forbidden = []  # list of (a, b, c)
    # We want to find a set S with no triple.
    # It's easier to just check all pairs (a,b) and see if c is an integer in range.
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                # We need distinct elements.
                # The equation is symmetric in b and c.
                # If c == a, then 1/a = 1/a + 1/c => 1/c = 0, impossible.
                # If c == b, then 1/a = 2/b => b = 2a. Then c=b=2a. Not distinct.
                # So if c is integer, it is either distinct or equal to one of them.
                # Check distinctness:
                if c != a and c != b:
                    # We can store the triple sorted to avoid duplicates in conflict checking?
                    # Or just store as is. The constraint is "no three distinct elements".
                    # So any permutation of (a,b,c) is the same constraint.
                    forbidden.append((a, b, c))
                    
    # We can reduce the number of triples by only storing each unique set {a,b,c} once.
    # But for the heuristic, having duplicates doesn't hurt correctness, just speed.
    # Let's unique them.
    unique_triples = set()
    for t in forbidden:
        unique_triples.add(tuple(sorted(t)))
    forbidden = list(unique_triples)
    
    # Build incidence lists
    elem_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(forbidden):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)

    def is_valid(in_set):
        """Check if in_set has any forbidden triple."""
        for a, b, c in forbidden:
            if in_set[a] and in_set[b] and in_set[c]:
                return False
        return True

    def polish(in_set, removed):
        """Try to add back removed elements if they don't cause conflicts."""
        changed = True
        while changed:
            changed = False
            # Shuffle removed to break symmetry?
            # The original code iterated list(removed).
            # To speed up, we can just try to add any removed element that fits.
            # We'll iterate through a shuffled copy of removed.
            for x in list(removed):
                if in_set[x]:
                    continue
                # Check if adding x creates a conflict
                can_add = True
                for idx in elem_to_triples[x]:
                    a, b, c = forbidden[idx]
                    others = [y for y in (a, b, c) if y != x]
                    if in_set[others[0]] and in_set[others[1]]:
                        can_add = False
                        break
                if can_add:
                    in_set[x] = True
                    removed.remove(x)
                    changed = True
        return in_set, removed

    def solve_once(seed):
        random.seed(seed)
        in_set = [False] * (n + 1)
        # Start with all elements
        for i in range(1, n + 1):
            in_set[i] = True
        removed = set()
        
        # Phase 1: Remove elements to satisfy constraints
        # We can do this by repeatedly finding a violated triple and removing one element from it.
        # To be faster, we can just iterate through triples and if violated, remove a random element.
        # This is O(T * K) roughly.
        # Let's do a randomized greedy removal.
        # Actually, the previous code's "greedy_solve" was removing the element with highest degree.
        # That is a good heuristic. Let's stick to a variant of that but faster.
        
        # Let's use the "highest degree" removal but with a time check.
        # Precompute degrees (number of triples containing element)
        # But degrees change as we remove elements.
        
        # Alternative: Randomized restarts of a simple local search.
        # 1. Start with random subset? No, start with all.
        # 2. While violated: pick random violated triple, remove random element from it.
        # 3. Polish.
        
        # This is very fast.
        
        # Let's try a mix:
        # 1. Start with all.
        # 2. While any triple is fully in set:
        #    - Pick a random such triple.
        #    - Remove a random element from it.
        
        # To make this efficient, we need to quickly find a violated triple.
        # We can just iterate through `forbidden` list.
        # N=180, T is small.
        
        while True:
            if time.time() > deadline:
                break
            violated = None
            for t in forbidden:
                a, b, c = t
                if in_set[a] and in_set[b] and in_set[c]:
                    violated = t
                    break
            if not violated:
                break
            a, b, c = violated
            # Remove one of them randomly
            x = random.choice([a, b, c])
            in_set[x] = False
            removed.add(x)
            
        # Phase 2: Polish
        in_set, removed = polish(in_set, removed)
        
        return [x for x in range(1, n + 1) if in_set[x]]

    best_result = None
    best_size = 0
    
    # Run multiple trials
    trials = 0
    while time.time() < deadline - 2:
        result = solve_once(trials)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        trials += 1
        
        # If we've done many trials, check time strictly
        if trials % 100 == 0 and time.time() > deadline - 5:
            break

    if not best_result:
        # Fallback: upper half of the range is usually safe?
        # 1/a = 1/b + 1/c. If a,b,c > 90, 1/a < 1/90.
        # 1/b + 1/c > 1/90 + 1/90 = 2/90 = 1/45.
        # So 1/a < 1/45 is possible.
        # Actually, just return a known valid set or the best found.
        # If no trials ran, return something valid.
        # [180] is valid.
        best_result = [180]
        
    return best_result
