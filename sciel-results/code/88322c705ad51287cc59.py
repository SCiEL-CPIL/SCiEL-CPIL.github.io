"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from fractions import Fraction
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]

    best_indices = None
    best_deficit = Fraction(1, 1)  # start with deficit 1 (empty set is not allowed, but we need a valid start)

    # Greedy initialization: take largest possible fractions
    greedy_indices = []
    greedy_total = Fraction(0)
    for i, r in enumerate(recips):
        if greedy_total + r < 1:
            greedy_indices.append(i)
            greedy_total += r
    if greedy_indices:
        best_indices = greedy_indices
        best_deficit = 1 - greedy_total

    # Local search: try to swap in/out indices to improve deficit
    def get_deficit(indices):
        s = Fraction(0)
        for i in indices:
            s += recips[i]
        if s >= 1:
            return None
        return 1 - s

    def try_local_search(indices, max_iters=5000):
        nonlocal best_indices, best_deficit
        current = list(indices)
        current_deficit = get_deficit(current)
        if current_deficit is None:
            return
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_indices = current[:]
        
        iters = 0
        improved = True
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding one element
            for i in range(n):
                if i not in current:
                    new_set = current + [i]
                    d = get_deficit(new_set)
                    if d is not None and d < current_deficit:
                        current = new_set
                        current_deficit = d
                        if current_deficit < best_deficit:
                            best_deficit = current_deficit
                            best_indices = current[:]
                        improved = True
                        break
            if time.time() >= deadline:
                break
            # Try removing one element
            for i in current[:]:
                new_set = [x for x in current if x != i]
                if not new_set:
                    continue
                d = get_deficit(new_set)
                if d is not None and d < current_deficit:
                    current = new_set
                    current_deficit = d
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_indices = current[:]
                    improved = True
                    break
            if time.time() >= deadline:
                break
            # Try swapping one in and one out
            if time.time() >= deadline:
                break
            for i in range(n):
                if i in current:
                    continue
                for j in current:
                    new_set = [x for x in current if x != j] + [i]
                    d = get_deficit(new_set)
                    if d is not None and d < current_deficit:
                        current = new_set
                        current_deficit = d
                        if current_deficit < best_deficit:
                            best_deficit = current_deficit
                            best_indices = current[:]
                        improved = True
                        break
                if improved:
                    break

    # Run local search from greedy start
    try_local_search(greedy_indices, max_iters=20000)

    # Multiple random restarts
    rng = random.Random(42)
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random subset
        k = rng.randint(3, 12)
        indices = rng.sample(range(n), k)
        d = get_deficit(indices)
        if d is not None:
            if d < best_deficit:
                best_deficit = d
                best_indices = indices[:]
            try_local_search(indices, max_iters=5000)
        if time.time() >= deadline:
            break

    if best_indices is None or len(best_indices) == 0:
        # Fallback: single element
        best_indices = [0]
    
    return best_indices
# EVOLVE-BLOCK-END
