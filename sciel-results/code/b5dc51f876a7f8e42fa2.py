"""Sidon set: small inclusion-maximal subset of [1,256]."""
import time
import random

N = 256
DEADLINE = time.time() + 25.0  # safety margin before 30s kill

def construct():
    start = time.time()
    best_size = N + 1
    best_set = None

    rng = random.Random(42)

    def is_valid_addition(x, selected, sums):
        """Check if x can be added to the current Sidon set."""
        for a in selected:
            if x + a in sums:
                return False
        if 2 * x in sums:
            return False
        # Check duplicates among new sums
        # New sums are {x+a for a in selected} U {2x}
        # We already checked x+a against old sums.
        # We need to check if any x+a == x+a' (implies a=a', impossible as a distinct)
        # and if x+a == 2x (implies a=x, impossible as x not in selected)
        # So we just need to ensure 2x is not equal to any x+a?
        # 2x = x+a => x=a. x is not in selected. So 2x is distinct from x+a.
        # So if x+a not in sums and 2x not in sums, it's valid.
        return True

    def greedy_construct(order):
        """Greedy: pick the first valid element from the given order."""
        selected = []
        sums = set()
        selected_set = set()
        
        for x in order:
            if time.time() > DEADLINE:
                break
            if x in selected_set:
                continue
            if is_valid_addition(x, selected, sums):
                selected.append(x)
                selected_set.add(x)
                for a in selected[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        
        # Check maximality
        all_sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                all_sums.add(selected[i] + selected[j])
        
        is_maximal = True
        for y in range(1, N + 1):
            if y in selected_set:
                continue
            can_add = True
            for a in selected:
                if y + a in all_sums:
                    can_add = False
                    break
            if can_add and 2 * y in all_sums:
                can_add = False
            # Also need to check if y+a conflicts with y+a' or 2y?
            # The check "y+a in all_sums" covers y+a = a1+a2.
            # If y+a = y+a' then a=a'.
            # If y+a = 2y then a=y.
            # So if y+a not in all_sums and 2y not in all_sums, is it sufficient?
            # We must also ensure that the NEW sums {y+a, 2y} don't collide with each other or each other's values?
            # y+a = 2y => a=y.
            # y+a = y+a' => a=a'.
            # So yes, checking against existing sums is sufficient for validity of addition.
            if can_add:
                is_maximal = False
                break
        
        if is_maximal:
            return selected
        return None

    # Try multiple random permutations and other orderings
    base = list(range(1, N + 1))
    
    # 1. Random shuffles
    while time.time() < DEADLINE - 1.0:
        perm = base[:]
        rng.shuffle(perm)
        result = greedy_construct(perm)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result

    # 2. Try specific orderings: Ascending, Descending
    for ordering in [base, base[::-1]]:
        if time.time() > DEADLINE - 1.0:
            break
        result = greedy_construct(ordering)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result

    if best_set is None:
        # Fallback: ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if is_valid_addition(x, selected, sums):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        best_set = selected
        best_size = len(selected)

    return best_set
