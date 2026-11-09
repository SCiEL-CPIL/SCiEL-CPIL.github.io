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

    def greedy_large_first():
        selected = []
        for x in range(n, 0, -1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected

    def greedy_degree_based():
        """
        Greedy selection based on minimizing 'conflict potential'.
        We prefer larger numbers as they are less likely to divide sums of others.
        We also add randomness to escape local optima.
        """
        selected = []
        available = list(range(1, n + 1))
        
        while len(selected) < max_elements and available:
            valid_candidates = []
            for x in available:
                if is_valid_addition(selected, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Heuristic: Pick the largest valid candidate, but with some randomness
            # Sort by value descending
            valid_candidates.sort(reverse=True)
            
            # To implement "degree" logic: 
            # A number with fewer multiples in the range [1, 576] is "safer".
            # Larger numbers have fewer multiples.
            # So picking largest is consistent with low-degree in conflict hypergraph.
            
            # Add randomness: pick from top 5 valid candidates
            top_k = valid_candidates[:min(5, len(valid_candidates))]
            x = random.choice(top_k)
            
            selected.append(x)
            available.remove(x)
            
        return selected

    def local_search(selected):
        selected = list(selected)
        best_size = len(selected)
        no_improve_count = 0
        max_no_improve = 50
        
        while no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try adding
            if len(selected) < max_elements:
                for x in range(1, n + 1):
                    if x not in selected:
                        if is_valid_addition(selected, x):
                            selected.append(x)
                            improved = True
                            break
            
            # 2. Try single swap
            if not improved and len(selected) < max_elements:
                # Try removing one and adding two? No, swap is remove 1 add 1.
                # But if we are at max capacity, we can't add.
                # The goal is to maximize cardinality.
                # If len < max, we should try to add.
                # If len == max, we can't improve cardinality by swap.
                # But maybe a swap allows adding later?
                # Let's focus on adding first.
                pass
            
            # 3. Try 2-swap (remove 2, add 2) if stuck
            # This is expensive O(N^4). Only do if len < max and we are stuck.
            if not improved and len(selected) < max_elements:
                # Try to find a pair (x, y) to add and pair (u, v) to remove
                # such that (selected - {u, v}) + {x, y} is valid.
                # This is complex. Let's stick to 1-swap for now but allow it to change the set structure.
                # Actually, if we can't add, maybe we should remove a "bad" element and add a "good" one to enable further additions?
                # This requires lookahead.
                
                # Simple 1-swap: Remove u, Add x.
                # This keeps size same. Doesn't help if we want to increase size.
                # Unless removing u allows adding x AND y later?
                
                # Let's try a different move: 
                # If we can't add any single element, try to swap one element for another 
                # that has lower "conflict degree" to allow future additions.
                
                # Calculate conflict score for each element in selected
                # Score(a) = number of other elements b such that a | (b+c+d) is "close" to happening?
                # Hard to define.
                
                # Let's just try random 1-swaps to escape local minima.
                for _ in range(10):
                    i = random.randint(0, len(selected) - 1)
                    u = selected[i]
                    temp = selected[:i] + selected[i+1:]
                    
                    # Try to add a random valid element to temp
                    candidates = [x for x in range(1, n+1) if x not in temp]
                    random.shuffle(candidates)
                    for x in candidates:
                        if is_valid_addition(temp, x):
                            new_set = temp + [x]
                            # Check if this new set allows adding more elements?
                            # For now, just accept if it's valid.
                            # But we need to ensure we don't degrade.
                            # Since size is same, we rely on randomness to find a better configuration.
                            selected = new_set
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
            else:
                no_improve_count += 1
        
        return selected

    best = greedy_large_first()
    
    for trial in range(50):
        random.seed(2000 + trial)
        s = greedy_degree_based()
        s = local_search(s)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
    
    if not is_valid(best):
        best = greedy_large_first()
    
    return best[:max_elements]
