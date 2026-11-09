import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    N = 180
    
    # Precompute all forbidden triples
    triple_set = set()
    triples = []
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    t = tuple(sorted([a, b, c]))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    # For each element, which triples does it belong to?
    elem_triples = {i: [] for i in range(1, N+1)}
    for t in triples:
        for x in t:
            elem_triples[x].append(t)
    
    def can_add(e, S):
        """Check if element e can be added to set S without creating a conflict."""
        for t in elem_triples[e]:
            others = [x for x in t if x != e]
            if all(x in S for x in others):
                return False
        return True
    
    best_size = 0
    best_set = set()
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Start with a large set: all elements from some threshold
        # Try different strategies for initialization
        
        if restart % 3 == 0:
            # All large numbers
            S = set(range(91, N+1))
        elif restart % 3 == 1:
            # Random subset
            S = set(random.sample(range(1, N+1), random.randint(80, 130)))
            # Fix conflicts
            changed = True
            while changed:
                changed = False
                for e in list(S):
                    for t in elem_triples[e]:
                        if all(x in S for x in t):
                            # Remove the largest element to minimize damage
                            S.remove(max(t))
                            changed = True
                            break
        else:
            # Greedy with random order
            elements = list(range(1, N+1))
            random.shuffle(elements)
            S = set()
            for e in elements:
                if can_add(e, S):
                    S.add(e)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try 1-additions
            for e in range(1, N+1):
                if e not in S and can_add(e, S):
                    S.add(e)
                    improved = True
            
            # Try 2-swaps: remove one, add one
            if not improved:
                S_list = list(S)
                random.shuffle(S_list)
                for e_remove in S_list:
                    if time.time() > deadline - 5:
                        break
                    S.remove(e_remove)
                    added = False
                    for e_add in range(1, N+1):
                        if e_add not in S and can_add(e_add, S):
                            S.add(e_add)
                            added = True
                            improved = True
                            break
                    if not added:
                        S.add(e_remove)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    return list(best_set)
