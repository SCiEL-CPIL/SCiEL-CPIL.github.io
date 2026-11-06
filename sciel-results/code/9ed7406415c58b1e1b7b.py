import time
import random

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _violations_count(selected):
    """Count total constraint violations in the set."""
    n = len(selected)
    if n < 4:
        return 0
    violations = 0
    for x in selected:
        # Case 1: x divides sum of 3 distinct others
        for i in range(n):
            if selected[i] == x:
                continue
            for j in range(i + 1, n):
                if selected[j] == x:
                    continue
                for k in range(j + 1, n):
                    if selected[k] == x:
                        continue
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        violations += 1
                        break
                if violations > 1000:
                    return violations
    # Case 2: x is a summand; check each a in selected if (x + b + c) % a == 0
    for x in selected:
        for idx_a, a in enumerate(selected):
            # find 2 distinct others b, c (not x, not a)
            others = [s for s in selected if s != x and s != a]
            if len(others) < 2:
                continue
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        violations += 1
                        break
                if violations > 1000:
                    return violations
    return violations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    if n < 3:
        return True

    # Case 1: x is the divisor.
    for i in range(n):
        a1 = selected[i]
        for j in range(i + 1, n):
            a2 = selected[j]
            s12 = a1 + a2
            for k in range(j + 1, n):
                if (s12 + selected[k]) % x == 0:
                    return False

    # Case 2: x is a summand.
    for idx_a, a in enumerate(selected):
        for i in range(n):
            if i == idx_a:
                continue
            a1 = selected[i]
            for j in range(i + 1, n):
                if j == idx_a:
                    continue
                if (a1 + selected[j] + x) % a == 0:
                    return False

    return True

def _greedy(order, deadline, selected_start=None):
    """Greedy construction following given order."""
    selected = list(selected_start) if selected_start else []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline, rng, max_iters=200):
    """Try to improve via removals and re-additions."""
    current = list(selected)
    current_size = len(current)
    all_numbers = set(range(1, 193))
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing one element and see if we can add two new ones
        if current_size >= 2:
            for idx in range(len(current)):
                candidate_remove = current[idx]
                remaining = current[:idx] + current[idx+1:]
                
                # Try to add two elements from remaining candidates
                candidates = [x for x in all_numbers if x not in remaining]
                rng.shuffle(candidates)
                
                added = []
                for x in candidates[:20]:
                    if _can_add(x, remaining + added):
                        added.append(x)
                        if len(added) >= 2:
                            break
                
                if len(added) >= 2:
                    new_set = remaining + added
                    if len(new_set) > current_size:
                        current = new_set
                        current_size = len(current)
                        improved = True
                        break
                elif len(added) == 1 and current_size > 1:
                    # Try just adding one if it helps
                    pass
        
        # Try swapping one element for another
        if current_size > 0:
            for idx in range(len(current)):
                candidate_remove = current[idx]
                remaining = current[:idx] + current[idx+1:]
                candidates = [x for x in all_numbers if x not in remaining]
                rng.shuffle(candidates)
                
                for x in candidates[:30]:
                    if x == candidate_remove:
                        continue
                    if _can_add(x, remaining):
                        new_set = remaining + [x]
                        # Only swap if it allows more additions later
                        # For now, just accept if same size (might help future adds)
                        if len(new_set) == current_size:
                            # Check if this swap enables more additions
                            further = []
                            for y in candidates:
                                if y in remaining or y == x:
                                    continue
                                if _can_add(y, new_set + further):
                                    further.append(y)
                                    if len(further) >= 2:
                                        current = new_set + further
                                        current_size = len(current)
                                        improved = True
                                        break
                            if improved:
                                break
            if improved:
                break
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    best = []
    best_size = 0
    
    all_numbers = list(range(1, 193))
    
    # Heuristic 1: Descending
    order1 = list(range(192, 0, -1))
    res = _greedy(order1, deadline)
    if len(res) > best_size:
        best = res
        best_size = len(res)
    
    # Heuristic 2: Ascending
    if time.time() < deadline:
        order2 = list(range(1, 193))
        res = _greedy(order2, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
    
    # Heuristic 3: Primes first (descending)
    if time.time() < deadline:
        primes = [x for x in range(192, 0, -1) if _is_prime(x)]
        non_primes = [x for x in range(192, 0, -1) if not _is_prime(x)]
        order3 = primes + non_primes
        res = _greedy(order3, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
    
    # Heuristic 4: Coprime to 6 first (descending)
    if time.time() < deadline:
        def coprime6(x):
            return x % 2 != 0 and x % 3 != 0
        c6 = [x for x in range(192, 0, -1) if coprime6(x)]
        rest = [x for x in range(192, 0, -1) if not coprime6(x)]
        order4 = c6 + rest
        res = _greedy(order4, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
    
    # Heuristic 5: Large numbers first, but exclude small divisors
    if time.time() < deadline:
        # Prefer numbers > 96 first, then 48-96, then 24-48, then 1-24
        order5 = (list(range(192, 96, -1)) + 
                  list(range(96, 48, -1)) + 
                  list(range(48, 24, -1)) + 
                  list(range(24, 0, -1)))
        res = _greedy(order5, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
    
    # Local search on best solution
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        improved = _local_search(best, deadline - 5.0, rng, max_iters=50)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Random Restarts
    rng = random.Random(42)
    
    while time.time() < deadline - 2.0:
        # Create a random order with bias toward larger numbers
        order = list(all_numbers)
        # Weight larger numbers more heavily in ordering
        weighted_order = []
        for x in order:
            weight = x / 192.0  # Larger numbers get higher weight
            weighted_order.append((weight, x))
        rng.shuffle(weighted_order)
        order = [x for _, x in sorted(weighted_order, key=lambda t: -t[0])]
        
        res = _greedy(order, deadline - 2.0)
        
        # Apply local search if promising
        if len(res) >= best_size and time.time() < deadline - 3.0:
            improved = _local_search(res, deadline - 3.0, rng, max_iters=30)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                if best_size >= 48:
                    return best
        elif len(res) > best_size:
            best = res
            best_size = len(best)
            if best_size >= 48:
                return best
    
    return best
