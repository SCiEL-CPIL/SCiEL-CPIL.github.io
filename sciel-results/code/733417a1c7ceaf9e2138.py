import random
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    
    def is_valid_addition(selected, candidate):
        """Check if adding candidate to selected maintains validity."""
        sel_set = set(selected)
        # For the new candidate: check if it divides sum of any 3 others in selected
        if len(selected) >= 3:
            for a, b, c in combinations(selected, 3):
                if (a + b + c) % candidate == 0:
                    return False
        # For each existing element a: check triples that include candidate
        # i.e., a | (b + c + candidate) for b,c in selected, b!=a, c!=a, b!=c
        for a in selected:
            others = [x for x in selected if x != a]
            if len(others) >= 2:
                for b, c in combinations(others, 2):
                    if (b + c + candidate) % a == 0:
                        return False
        return True

    def is_valid(selected):
        """Check if the entire set is valid."""
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def greedy_by_degree():
        selected = []
        sel_set = set()
        
        for _ in range(max_elements):
            # Precompute bad residues for all y not in selected
            # BadResidues[y] = set of residues r such that r % y == (-(b+c)) % y for some b,c in selected
            # If x % y is in BadResidues[y], then adding x makes y invalid (because y | b+c+x)
            bad_residues = {}
            for y in range(1, n + 1):
                if y in sel_set:
                    continue
                residues = set()
                if len(selected) >= 2:
                    for b, c in combinations(selected, 2):
                        residues.add((-(b + c)) % y)
                bad_residues[y] = residues
            
            # Find valid candidates
            valid_candidates = []
            for x in range(1, n + 1):
                if x in sel_set:
                    continue
                if is_valid_addition(selected, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
                
            # Score each valid candidate: number of y it kills
            best_x = None
            min_kills = float('inf')
            random.shuffle(valid_candidates) # Add randomness to break ties
            
            for x in valid_candidates:
                kills = 0
                for y in range(1, n + 1):
                    if y in sel_set or y == x:
                        continue
                    if x % y in bad_residues[y]:
                        kills += 1
                if kills < min_kills:
                    min_kills = kills
                    best_x = x
                    if min_kills == 0:
                        break # Can't do better than 0
            
            selected.append(best_x)
            sel_set.add(best_x)
            
        return selected

    def local_search(selected):
        selected = list(selected)
        sel_set = set(selected)
        improved = True
        while improved and len(selected) < max_elements:
            improved = False
            # Try adding
            for x in range(1, n + 1):
                if x not in sel_set:
                    if is_valid_addition(selected, x):
                        selected.append(x)
                        sel_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            
            # Try single swap to enable addition? 
            # If we can't add, maybe removing one allows adding two?
            # Or just try swapping to a "better" position for future?
            # Simple swap: remove one, try to add another, see if we can then add a third?
            # This is complex. Let's just do random restarts of greedy.
            break
            
        return selected

    best = greedy_by_degree()
    
    # Run multiple trials with different random seeds to handle ties/shuffling
    for trial in range(50):
        random.seed(1000 + trial)
        s = greedy_by_degree()
        # Simple local search to fill gaps
        sel_set = set(s)
        for x in range(1, n + 1):
            if x not in sel_set and len(s) < max_elements:
                if is_valid_addition(s, x):
                    s.append(x)
                    sel_set.add(x)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
    
    if not is_valid(best):
        # Fallback to simple greedy if something went wrong
        selected = []
        for x in range(1, n + 1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        best = selected
        
    return best[:max_elements]
