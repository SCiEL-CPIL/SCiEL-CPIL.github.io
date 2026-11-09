"""Sidon set: small inclusion-maximal subset of [1,256]."""
import time
import random

N = 256
DEADLINE = time.time() + 28.0  # safety margin before 30s kill

def construct():
    start = time.time()
    best_size = N + 1
    best_set = None

    rng = random.Random(42)

    def compute_sums(selected):
        """Compute all pair sums (including diagonal) for a Sidon set."""
        sums = set()
        n = len(selected)
        for i in range(n):
            si = selected[i]
            for j in range(i, n):
                sums.add(si + selected[j])
        return sums

    def is_valid_addition(x, selected, sums):
        """Check if x can be added to the current Sidon set."""
        for a in selected:
            if x + a in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def is_maximal_fast(selected, sums):
        """Check maximality: can any missing element be added?"""
        selected_set = set(selected)
        for y in range(1, N + 1):
            if y in selected_set:
                continue
            can_add = True
            for a in selected:
                if y + a in sums:
                    can_add = False
                    break
            if can_add and 2 * y in sums:
                can_add = False
            if can_add:
                return False
        return True

    def try_reduce(selected, rng):
        """Try to remove elements while maintaining maximality."""
        selected = list(selected)
        n_orig = len(selected)
        
        for _ in range(n_orig):
            if time.time() > DEADLINE - 0.5:
                break
            if len(selected) <= 1:
                break
            
            idx = rng.randint(0, len(selected) - 1)
            x = selected[idx]
            
            new_selected = selected[:idx] + selected[idx+1:]
            new_sums = compute_sums(new_selected)
            
            new_selected_set = set(new_selected)
            is_still_maximal = True
            for y in range(1, N + 1):
                if y in new_selected_set:
                    continue
                can_add = True
                for a in new_selected:
                    if y + a in new_sums:
                        can_add = False
                        break
                if can_add and 2 * y in new_sums:
                    can_add = False
                if can_add:
                    is_still_maximal = False
                    break
            
            if is_still_maximal:
                selected = new_selected
        
        return selected

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
        
        if not is_maximal_fast(selected, compute_sums(selected)):
            return None
        
        return selected

    base = list(range(1, N + 1))
    
    while time.time() < DEADLINE - 3.0:
        perm = base[:]
        rng.shuffle(perm)
        result = greedy_construct(perm)
        if result is not None:
            reduced = try_reduce(result, rng)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced

    for ordering in [base, base[::-1]]:
        if time.time() > DEADLINE - 1.0:
            break
        result = greedy_construct(ordering)
        if result is not None:
            reduced = try_reduce(result, rng)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced

    if best_set is None:
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
