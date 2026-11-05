import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Known good solution: {66, ..., 100}, size 35
    best_set = set(range(66, 101))
    best_size = 35

    def is_valid(s):
        """Check if no subset of s sums to 200."""
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True

    def compute_reachable(s):
        """Compute all subset sums up to 200 for set s."""
        reachable = {0}
        for x in s:
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        return reachable

    def greedy_construct(order):
        """Greedy: add element if it doesn't create sum 200."""
        s = set()
        reachable = {0}
        for x in order:
            if (200 - x) not in reachable:
                s.add(x)
                new_sums = {y + x for y in reachable if y + x <= 200}
                reachable |= new_sums
        return s, reachable

    # Phase 1: Multiple greedy constructions with different orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 50, -1)) + list(range(50, 0, -1)),
        list(range(75, 0, -1)) + list(range(100, 75, -1)),
        list(range(25, 0, -1)) + list(range(100, 25, -1)),
    ]
    
    # Add random orders
    for _ in range(20):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        s, _ = greedy_construct(order)
        if len(s) > best_size and is_valid(s):
            best_size = len(s)
            best_set = s

    # Phase 2: Local search with multiple move types
    def local_search(initial_set, max_iterations=300):
        """Improve set via local search moves."""
        s = set(initial_set)
        reachable = compute_reachable(s)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1

            # Move 1: Add a single element
            for x in range(1, 101):
                if x not in s and (200 - x) not in reachable:
                    s.add(x)
                    new_sums = {y + x for y in reachable if y + x <= 200}
                    reachable |= new_sums
                    improved = True
                    break

            if improved:
                continue

            # Move 2: Remove one, add two (net +1)
            for x_out in sorted(s):
                if time.time() > deadline:
                    return s
                s.remove(x_out)
                reachable = compute_reachable(s)
                
                added = []
                # Try to find two elements to add
                for x1 in range(1, 101):
                    if x1 not in s and (200 - x1) not in reachable:
                        s.add(x1)
                        new_sums = {y + x1 for y in reachable if y + x1 <= 200}
                        reachable |= new_sums
                        added.append(x1)
                        break
                
                if len(added) == 1:
                    for x2 in range(1, 101):
                        if x2 not in s and x2 != added[0] and (200 - x2) not in reachable:
                            s.add(x2)
                            new_sums = {y + x2 for y in reachable if y + x2 <= 200}
                            reachable |= new_sums
                            added.append(x2)
                            break
                
                if len(added) == 2:
                    improved = True
                    break
                else:
                    for x_add in added:
                        s.discard(x_add)
                    reachable = compute_reachable(s)

            if improved:
                continue

            # Move 3: Remove two, add three (net +1)
            found = False
            for x1_out in sorted(s):
                if found or time.time() > deadline:
                    break
                for x2_out in sorted(s):
                    if x2_out <= x1_out:
                        continue
                    if time.time() > deadline:
                        return s
                    
                    s.discard(x1_out)
                    s.discard(x2_out)
                    reachable = compute_reachable(s)
                    
                    added = []
                    for x1 in range(1, 101):
                        if x1 not in s and (200 - x1) not in reachable:
                            s.add(x1)
                            new_sums = {y + x1 for y in reachable if y + x1 <= 200}
                            reachable |= new_sums
                            added.append(x1)
                            break
                    
                    if len(added) >= 1:
                        for x2 in range(1, 101):
                            if x2 not in s and x2 != added[0] and (200 - x2) not in reachable:
                                s.add(x2)
                                new_sums = {y + x2 for y in reachable if y + x2 <= 200}
                                reachable |= new_sums
                                added.append(x2)
                                break
                    
                    if len(added) >= 2:
                        for x3 in range(1, 101):
                            if x3 not in s and x3 not in added and (200 - x3) not in reachable:
                                s.add(x3)
                                new_sums = {y + x3 for y in reachable if y + x3 <= 200}
                                reachable |= new_sums
                                added.append(x3)
                                break
                    
                    if len(added) == 3:
                        found = True
                        improved = True
                        break
                    else:
                        for x_add in added:
                            s.discard(x_add)
                        reachable = compute_reachable(s)

            if not improved:
                # Random perturbation to escape local optima
                if random.random() < 0.3 and len(s) > 2:
                    # Remove 1-2 random elements
                    n_remove = random.randint(1, 2)
                    to_remove = random.sample(list(s), min(n_remove, len(s)))
                    for x in to_remove:
                        s.discard(x)
                    reachable = compute_reachable(s)

        return s

    # Apply local search to best set
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=400)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set

    # Phase 3: Shrink-and-regrow strategy
    # Remove k elements from best_set, then greedily re-add
    if time.time() < deadline:
        for k in range(1, 6):
            if time.time() > deadline:
                break
            # Try several random removals
            for trial in range(5):
                if time.time() > deadline:
                    break
                s = set(best_set)
                # Remove k random elements
                to_remove = random.sample(list(s), min(k, len(s)))
                for x in to_remove:
                    s.discard(x)
                
                # Greedily re-add all possible elements
                reachable = compute_reachable(s)
                for x in range(1, 101):
                    if x not in s and (200 - x) not in reachable:
                        s.add(x)
                        new_sums = {y + x for y in reachable if y + x <= 200}
                        reachable |= new_sums
                
                if len(s) > best_size and is_valid(s):
                    best_size = len(s)
                    best_set = s

    # Phase 4: Final local search on best found
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=200)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set

    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
