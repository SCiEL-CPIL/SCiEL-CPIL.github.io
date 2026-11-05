"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _compute_sizes(A):
    """Compute |A+A| and |A*A| with equal summands/factors allowed."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(i, n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    return len(sum_set), len(prod_set)

def _score(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _compute_sizes(A)
    return max(s, p)

def _lattice_seed(max_val=10000, n=20):
    """Generate n distinct values of form 2^i * 3^j <= max_val."""
    candidates = set()
    for i in range(0, 14):  # 2^13 = 8192
        for j in range(0, 9):  # 3^8 = 6561
            val = (1 << i) * (3 ** j)
            if 1 <= val <= max_val:
                candidates.add(val)
    # Sort and pick n values spread across the range to minimize additive collisions
    cands = sorted(candidates)
    if len(cands) <= n:
        return cands
    # Pick evenly spaced from the sorted list for good spread
    step = len(cands) / n
    selected = []
    for k in range(n):
        idx = int(k * step)
        selected.append(cands[idx])
    return sorted(set(selected))

def _local_search(A, max_time=80.0, start_time=None):
    """Local search: swap elements to reduce max(|A+A|, |A*A|)."""
    if start_time is None:
        start_time = time.time()
    
    A = sorted(set(A))
    n = len(A)
    best_score = _score(A)
    best_A = list(A)
    
    deadline = start_time + max_time
    no_improve_count = 0
    max_no_improve = 50
    
    # Precompute for faster evaluation
    def evaluate_swap(A, i, new_val):
        """Evaluate the score if we replace A[i] with new_val."""
        A_new = list(A)
        A_new[i] = new_val
        A_new = sorted(set(A_new))
        if len(A_new) != n:
            return float('inf')
        if any(v > 10000 for v in A_new):
            return float('inf')
        return _score(A_new)
    
    while time.time() < deadline - 2.0:
        improved = False
        random.shuffle(A)
        
        for i in range(n):
            if time.time() >= deadline - 2.0:
                break
            # Try replacing A[i] with random values
            for _ in range(20):
                if time.time() >= deadline - 2.0:
                    break
                new_val = random.randint(1, 10000)
                if new_val in A:
                    continue
                s = evaluate_swap(A, i, new_val)
                if s < best_score:
                    A = list(A)
                    A[i] = new_val
                    A = sorted(set(A))
                    if len(A) != n:
                        A = best_A[:]
                        continue
                    best_score = s
                    best_A = list(A)
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    return best_A, best_score

def _genetic_algorithm(pop_size=30, max_time=80.0, start_time=None):
    """Genetic algorithm to find good sets."""
    if start_time is None:
        start_time = time.time()
    
    deadline = start_time + max_time
    n = 20
    
    # Initialize population
    pop = []
    # Seed with lattice
    lattice = _lattice_seed()
    if len(lattice) >= n:
        pop.append(lattice[:n])
    
    # Add some random and structured seeds
    for _ in range(pop_size - 1):
        # Mix of random and semi-structured
        if random.random() < 0.3:
            # Geometric-like progression
            base = random.randint(1, 5)
            ratio = random.uniform(1.1, 1.8)
            vals = []
            v = base
            while len(vals) < n and v <= 10000:
                vals.append(int(v))
                v *= ratio
            if len(vals) < n:
                # Fill with random
                while len(vals) < n:
                    vals.append(random.randint(1, 10000))
            pop.append(sorted(set(vals))[:n] if len(set(vals)) >= n else sorted(random.sample(range(1, 10001), n)))
        else:
            pop.append(sorted(random.sample(range(1, 10001), n)))
    
    # Evaluate initial population
    scored_pop = []
    for individual in pop:
        if len(set(individual)) < n or any(v > 10000 for v in individual):
            scored_pop.append((float('inf'), individual))
        else:
            scored_pop.append((_score(individual), individual))
    
    scored_pop.sort(key=lambda x: x[0])
    best_score = scored_pop[0][0]
    best_A = scored_pop[0][1][:]
    
    generation = 0
    while time.time() < deadline - 5.0:
        generation += 1
        new_pop = []
        
        # Elitism: keep top half
        new_pop.extend([ind for _, ind in scored_pop[:pop_size//2]])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() >= deadline - 5.0:
                break
            
            # Crossover: pick two parents
            p1 = random.choice(scored_pop[:pop_size//2])[1]
            p2 = random.choice(scored_pop[:pop_size//2])[1]
            
            # Uniform crossover
            child = []
            for i in range(n):
                if random.random() < 0.5:
                    child.append(p1[i])
                else:
                    child.append(p2[i])
            
            # Mutation
            for i in range(n):
                if random.random() < 0.3:
                    child[i] = random.randint(1, 10000)
            
            child = sorted(set(child))
            if len(child) < n:
                while len(child) < n:
                    child.append(random.randint(1, 10000))
                    child = sorted(set(child))
                if len(child) < n:
                    child = sorted(random.sample(range(1, 10001), n))
            elif len(child) > n:
                child = child[:n]
            
            new_pop.append(child)
        
        # Evaluate new population
        scored_pop = []
        for individual in new_pop:
            if len(set(individual)) < n or any(v > 10000 for v in individual):
                scored_pop.append((float('inf'), individual))
            else:
                scored_pop.append((_score(individual), individual))
        
        scored_pop.sort(key=lambda x: x[0])
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_A = scored_pop[0][1][:]
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Phase 1: Lattice seed + local search
    lattice = _lattice_seed()
    if len(lattice) < 20:
        lattice = sorted(random.sample(range(1, 10001), 20))
    
    best_A, best_score = _local_search(lattice[:20], max_time=40.0, start_time=start_time)
    
    if time.time() >= deadline - 10.0:
        return best_A
    
    # Phase 2: Genetic algorithm for additional exploration
    ga_A, ga_score = _genetic_algorithm(pop_size=25, max_time=35.0, start_time=start_time)
    
    if ga_score < best_score:
        best_A = ga_A
        best_score = ga_score
    
    # Phase 3: Final local search on best
    if time.time() < deadline - 5.0:
        final_A, final_score = _local_search(best_A, max_time=min(10.0, deadline - time.time() - 2.0), start_time=start_time)
        if final_score < best_score:
            best_A = final_A
    
    return best_A
# EVOLVE-BLOCK-END
