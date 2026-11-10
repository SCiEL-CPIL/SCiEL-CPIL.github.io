import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    def is_valid(selected):
        """Check if selected list is valid."""
        n = len(selected)
        for i, a in enumerate(selected):
            others = [v for j, v in enumerate(selected) if j != i]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    def count_valid(selected):
        """Count how many elements are 'safe' - actually we need the whole set to be valid."""
        return is_valid(selected)
    
    # Greedy construction: try to build a large valid set
    def greedy_build():
        selected = []
        # Try adding numbers from high to low, but also try different orders
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            candidate = selected + [x]
            # Check if x can be added: x must not divide sum of any 3 in selected
            # AND no element in selected must divide sum of any 3 in (selected + [x]) that include x
            # Actually need to check the full condition for the new set
            # But that's expensive. Let's do a quick check:
            # 1. x doesn't divide sum of any 3 in selected
            safe_for_x = True
            for triple in combinations(selected, 3):
                if sum(triple) % x == 0:
                    safe_for_x = False
                    break
            if not safe_for_x:
                continue
            # 2. For each a in selected, a doesn't divide sum of any 3 in (selected + [x])
            #    The new triples involving x are: x + 2 from selected
            #    So for each a in selected, check if a divides (x + b + c) for any b,c in selected, b!=c, b!=a, c!=a
            safe_for_others = True
            for a in selected:
                for b, c in combinations([v for v in selected if v != a], 2):
                    if (x + b + c) % a == 0:
                        safe_for_others = False
                        break
                if not safe_for_others:
                    break
            if safe_for_others:
                selected.append(x)
        return selected
    
    # Start with greedy
    best = greedy_build()
    best_size = len(best)
    
    if time.time() > deadline:
        return best
    
    # Local search: try to improve by swapping or adding
    random.seed(42)
    
    def try_improve(current):
        """Try to improve current set by adding a number or swapping."""
        current_set = set(current)
        # Try adding a number not in current
        candidates = [x for x in range(192, 0, -1) if x not in current_set]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            new_set = current + [x]
            if is_valid(new_set):
                return new_set
        # Try swapping: remove one, add another
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for x in candidates:
                new_set = current[:i] + current[i+1:] + [x]
                if is_valid(new_set):
                    if len(new_set) >= len(current):
                        return new_set
        return None
    
    # Iterative improvement
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = try_improve(best)
        if improved is not None:
            best = improved
            best_size = len(best)
            if best_size >= 48:
                break
        else:
            # Try a random perturbation
            if len(best) > 1:
                i = random.randint(0, len(best) - 1)
                best = best[:i] + best[i+1:]
    
    return best
