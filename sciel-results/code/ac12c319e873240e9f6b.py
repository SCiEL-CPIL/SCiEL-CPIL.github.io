# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _build_forbidden(m, k):
    """Build the set of forbidden differences (nonzero k-th power residues mod m)."""
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    return forbidden


def _is_valid_subset(subset, forbidden, m):
    """Check if a subset has no forbidden differences."""
    s = set(subset)
    for a in s:
        for b in s:
            if a == b:
                continue
            if (a - b) % m in forbidden:
                return False
    return True


def _greedy_construct(forbidden, m, order=None):
    """Greedy construction: add elements in given order (or natural order) if valid."""
    if order is None:
        order = list(range(m))
    selected = []
    selected_set = set()
    for candidate in order:
        ok = True
        for s in selected:
            d1 = (candidate - s) % m
            d2 = (s - candidate) % m
            if d1 in forbidden or d2 in forbidden:
                ok = False
                break
        if ok:
            selected.append(candidate)
            selected_set.add(candidate)
    return selected


def _local_search_improve(subset, forbidden, m, max_iter=5000):
    """Try to improve subset by adding elements or swapping."""
    current = list(subset)
    current_set = set(current)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding each element not in current
        for candidate in range(m):
            if candidate in current_set:
                continue
            ok = True
            for s in current:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                current.append(candidate)
                current_set.add(candidate)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add another
            for i, s in enumerate(current):
                # Try to find a replacement
                for candidate in range(m):
                    if candidate in current_set:
                        continue
                    # Check if candidate is compatible with all except s
                    ok = True
                    for j, t in enumerate(current):
                        if i == j:
                            continue
                        d1 = (candidate - t) % m
                        d2 = (t - candidate) % m
                        if d1 in forbidden or d2 in forbidden:
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        current[i] = candidate
                        current_set.remove(s)
                        current_set.add(candidate)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return current


def _simulated_annealing(subset, forbidden, m, start_temp=1.0, end_temp=0.01, max_iter=10000, time_limit=30):
    """Simulated annealing to improve the subset."""
    start_time = time.time()
    current = list(subset)
    current_set = set(current)
    best = list(current)
    best_set = set(current)
    best_size = len(current)
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_subset = list(current)
        new_set = set(current)
        
        if move_type == 'add' and len(new_subset) < m:
            candidates = [x for x in range(m) if x not in new_set]
            if candidates:
                candidate = random.choice(candidates)
                new_subset.append(candidate)
                new_set.add(candidate)
        elif move_type == 'remove' and len(new_subset) > 1:
            idx = random.randint(0, len(new_subset) - 1)
            removed = new_subset.pop(idx)
            new_set.discard(removed)
        elif move_type == 'swap' and len(new_subset) >= 1:
            idx = random.randint(0, len(new_subset) - 1)
            removed = new_subset[idx]
            candidates = [x for x in range(m) if x not in new_set]
            if candidates:
                candidate = random.choice(candidates)
                new_subset[idx] = candidate
                new_set.discard(removed)
                new_set.add(candidate)
            else:
                continue
        
        # Evaluate: count forbidden pairs
        forbidden_count = 0
        for a in new_subset:
            for b in new_subset:
                if a != b and (a - b) % m in forbidden:
                    forbidden_count += 1
        
        # Objective: maximize size, minimize forbidden pairs
        old_obj = len(current) * 1000 - 0  # current is assumed valid
        new_obj = len(new_subset) * 1000 - forbidden_count * 10000
        
        if new_obj >= old_obj or random.random() < np.exp((new_obj - old_obj) / temp):
            current = new_subset
            current_set = new_set
            if len(current) > best_size and _is_valid_subset(current, forbidden, m):
                best = list(current)
                best_set = set(current)
                best_size = len(current)
        
        temp *= cooling
    
    # Return the best valid subset found
    if _is_valid_subset(best, forbidden, m):
        return best
    return current


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Uses greedy construction followed by local search and simulated annealing to find
    a large square-difference-free set.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    deadline = time.time() + 580  # 630s limit with safety margin
    
    forbidden = _build_forbidden(m, k)
    
    # Initial greedy construction
    best_subset = _greedy_construct(forbidden, m)
    
    # Try multiple random orderings for greedy
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        subset = _greedy_construct(forbidden, m, order)
        if len(subset) > len(best_subset):
            best_subset = subset
    
    # Local search improvement
    best_subset = _local_search_improve(best_subset, forbidden, m, max_iter=2000)
    
    # Simulated annealing
    if time.time() < deadline - 30:
        best_subset = _simulated_annealing(
            best_subset, forbidden, m,
            start_temp=1.0, end_temp=0.01,
            max_iter=5000,
            time_limit=min(30, max(0, deadline - time.time() - 5))
        )
    
    # Final local search
    if time.time() < deadline - 5:
        best_subset = _local_search_improve(best_subset, forbidden, m, max_iter=1000)
    
    # Verify validity
    if not _is_valid_subset(best_subset, forbidden, m):
        # Fallback to simple greedy
        best_subset = _greedy_construct(forbidden, m)
    
    return np.array(sorted(best_subset), dtype=np.int64)


# EVOLVE-BLOCK-END
