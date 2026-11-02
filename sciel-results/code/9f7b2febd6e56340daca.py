"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _score_set_fast(U):
    """Faster scoring using numpy-like approach for small sets."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(n):
            sums.add(U[i] + U[j])
            diffs.add(U[i] - U[j])
    n_sums = len(sums)
    n_diffs = len(diffs)
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _exhaustive_small(deadline):
    """Exhaustive search for sets of size 3-5 with reasonable bounds."""
    best_set = [0]
    best_score = -1
    
    # Size 2: [0, a]
    for a in range(1, 50):
        if time.time() > deadline - 5:
            break
        U = [0, a]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Size 3: [0, a, b]
    for a in range(1, 35):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, 70):
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Size 4: [0, a, b, c]
    for a in range(1, 20):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, 40):
            if time.time() > deadline - 5:
                break
            for c in range(b + 1, 70):
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Size 5: [0, a, b, c, d] - limited search
    for a in range(1, 12):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, 25):
            if time.time() > deadline - 5:
                break
            for c in range(b + 1, 45):
                if time.time() > deadline - 5:
                    break
                for d in range(c + 1, 70):
                    U = [0, a, b, c, d]
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
    
    return best_set, best_score

def _local_search(U, deadline, max_iter=10000):
    """Local search with incremental scoring."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        max_u = max(U)
        U_set = set(U)
        n = len(U)
        
        # Precompute sums and diffs
        sums_cache = set()
        diffs_cache = set()
        for a in U:
            for b in U:
                sums_cache.add(a + b)
                diffs_cache.add(a - b)
        
        # Try adding an element
        for new_elem in range(1, max_u + 30):
            if new_elem in U_set:
                continue
            new_sums = sums_cache.copy()
            new_diffs = diffs_cache.copy()
            for u in U:
                new_sums.add(new_elem + u)
                new_diffs.add(new_elem - u)
                new_diffs.add(u - new_elem)
            new_max = max(max_u, new_elem)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = sorted(U + [new_elem])
                            U_set = set(U)
                            current_score = s
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try removing an element
        for i in range(1, len(U)):
            if len(U) < 3:
                continue
            elem_to_remove = U[i]
            new_U = U[:i] + U[i+1:]
            new_sums = set()
            new_diffs = set()
            for a in new_U:
                for b in new_U:
                    new_sums.add(a + b)
                    new_diffs.add(a - b)
            new_max = max(new_U)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = new_U
                            U_set = set(U)
                            current_score = s
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try swapping an element
        for i in range(1, len(U)):
            old_elem = U[i]
            for new_elem in range(1, max_u + 40):
                if new_elem in U_set or new_elem == old_elem:
                    continue
                new_U = U[:i] + [new_elem] + U[i+1:]
                new_U_sorted = sorted(new_U)
                new_sums = set()
                new_diffs = set()
                for a in new_U_sorted:
                    for b in new_U_sorted:
                        new_sums.add(a + b)
                        new_diffs.add(a - b)
                new_max = max(new_U_sorted)
                n_sums = len(new_sums)
                n_diffs = len(new_diffs)
                if n_sums > 0 and n_diffs > 0:
                    ratio = n_diffs / n_sums
                    if ratio > 0:
                        denom = math.log(2 * new_max + 1)
                        if denom > 0:
                            s = 1 + math.log(ratio) / denom
                            if s > current_score + 1e-12:
                                U = new_U_sorted
                                U_set = set(U)
                                current_score = s
                                improved = True
                                break
            if improved:
                break
        
        if not improved:
            break
    
    return U, current_score

def _genetic_search(deadline, pop_size=30, generations=200):
    """Genetic search for better sets."""
    # Initialize population
    population = []
    for _ in range(pop_size):
        size = random.randint(3, 6)
        U = [0]
        for _ in range(size - 1):
            U.append(random.randint(1, 50))
        U = sorted(set(U))
        if len(U) >= 2:
            population.append(U)
    
    if not population:
        return [0, 1, 3], _score_set([0, 1, 3])
    
    best_set = population[0]
    best_score = _score_set(best_set)
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Evaluate population
        scored = []
        for U in population:
            s = _score_set(U)
            scored.append((s, U))
        scored.sort(key=lambda x: -x[0])
        
        # Update best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_set = scored[0][1][:]
        
        # Select parents (top half)
        n_parents = max(2, pop_size // 2)
        parents = [U for _, U in scored[:n_parents]]
        
        # Create next generation
        next_pop = [U[:] for U in scored[:pop_size // 3]]  # Elitism
        
        while len(next_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            # Crossover
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Simple: take elements from both
            child = [0]
            for x in p1:
                if x not in child and random.random() < 0.5:
                    child.append(x)
            for x in p2:
                if x not in child and random.random() < 0.5:
                    child.append(x)
            child = sorted(set(child))
            
            if len(child) < 2:
                child = p1[:]
            
            # Mutation
            if random.random() < 0.3:
                if child and random.random() < 0.5:
                    # Add element
                    child.append(random.randint(1, max(child) + 20))
                elif len(child) > 2:
                    # Remove element
                    idx = random.randint(1, len(child) - 1)
                    child = child[:idx] + child[idx+1:]
            
            # Mutate an element
            if random.random() < 0.4 and len(child) > 1:
                idx = random.randint(1, len(child) - 1)
                child[idx] = random.randint(1, max(child) + 20)
            
            child = sorted(set(child))
            if len(child) >= 2:
                next_pop.append(child)
        
        population = next_pop[:pop_size]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Exhaustive search for small sizes
    if time.time() < deadline - 30:
        ex_set, ex_score = _exhaustive_small(deadline)
        if ex_score > best_score:
            best_set = ex_set
            best_score = ex_score
    
    # Local search from best found
    if time.time() < deadline - 20:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=15000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 4, 9],
        [0, 2, 5, 11],
        [0, 1, 6, 10],
        [0, 3, 5, 8],
        [0, 2, 7, 13],
        [0, 3, 7, 11],
        [0, 1, 5, 12],
        [0, 1, 3, 6, 10],
        [0, 1, 4, 7, 11],
        [0, 2, 5, 8, 12],
        [0, 1, 3, 5, 8],
        [0, 2, 4, 6, 9],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 10:
            break
        s = _score_set(start)
        if s > best_score - 0.15:
            improved_set, improved_score = _local_search(start, deadline, max_iter=5000)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
    
    # Genetic search
    if time.time() < deadline - 15:
        gen_set, gen_score = _genetic_search(deadline, pop_size=40, generations=300)
        if gen_score > best_score:
            best_set = gen_set
            best_score = gen_score
    
    # Final local search polish
    if time.time() < deadline - 3:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=20000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
