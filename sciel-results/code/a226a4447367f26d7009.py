import time
import random
import math
from collections import Counter
from itertools import combinations

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _row_to_diffs(row):
    """Convert a row to its list of consecutive differences."""
    return [row[i+1] - row[i] for i in range(len(row)-1)]

def _diffs_to_row(diffs):
    """Convert consecutive differences back to a row starting at 0."""
    row = [0]
    for d in diffs:
        row.append(row[-1] + d)
    return row

def _row_diff_set(row):
    """Get the set of all pairwise differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k, rng):
    """
    Greedy initialization: build each row by choosing the smallest available value.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _crossover(parent1_diffs, parent2_diffs, rng):
    """
    Uniform crossover on the difference vectors.
    """
    n = len(parent1_diffs)
    child = []
    for i in range(n):
        if rng.random() < 0.5:
            child.append(parent1_diffs[i])
        else:
            child.append(parent2_diffs[i])
    return child

def _mutate(diffs, rng, strength=1):
    """
    Mutate individual differences by small perturbations.
    """
    n = len(diffs)
    mutated = list(diffs)
    for i in range(n):
        if rng.random() < 0.3 * strength:
            delta = rng.randint(-3, 3)
            new_val = mutated[i] + delta
            if new_val > 0:
                mutated[i] = new_val
    return mutated

def _repair_row(diffs, used_diffs, rng, max_scope):
    """
    Repair a row of differences so that all pairwise differences are valid.
    Returns a valid row or None.
    """
    row = _diffs_to_row(diffs)
    if row[0] != 0:
        return None
    if not all(row[i] < row[i+1] for i in range(len(row)-1)):
        return None
    if row[-1] > max_scope:
        return None
    
    row_diffs = _row_diff_set(row)
    if row_diffs & used_diffs:
        return None
    
    return row

def _fitness(rows, k):
    """
    Fitness: scope + penalty for collisions.
    """
    scope = _get_scope(rows)
    all_diffs = []
    for row in rows:
        all_diffs.extend(_compute_diffs(row))
    counts = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in counts.values())
    return scope + 1000 * collisions

def _genetic_algorithm(rows, n_rows, k, deadline, rng):
    """
    Genetic algorithm operating on difference vectors.
    """
    population_size = 20
    best_rows = [list(r) for r in rows]
    best_fit = _fitness(rows, k)
    
    # Initialize population
    population = []
    for _ in range(population_size):
        pop_row = []
        for r in rows:
            if rng.random() < 0.3:
                # Mutate
                diffs = _row_to_diffs(r)
                diffs = _mutate(diffs, rng, strength=2)
                new_row = _diffs_to_row(diffs)
                if new_row[0] == 0 and all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                    pop_row.append(new_row)
                else:
                    pop_row.append(list(r))
            else:
                pop_row.append(list(r))
        population.append(pop_row)
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for pop in population:
            fit = _fitness(pop, k)
            scored.append((fit, pop))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_fit:
            best_fit = scored[0][0]
            best_rows = [list(r) for r in scored[0][1]]
        
        # Selection: keep top half
        survivors = [pop for _, pop in scored[:population_size//2]]
        
        # Create new population
        new_population = list(survivors)
        
        while len(new_population) < population_size:
            if rng.random() < 0.7:
                # Crossover
                p1 = rng.choice(survivors)
                p2 = rng.choice(survivors)
                child = []
                for i in range(n_rows):
                    d1 = _row_to_diffs(p1[i])
                    d2 = _row_to_diffs(p2[i])
                    child_diffs = _crossover(d1, d2, rng)
                    # Repair
                    child_diffs = _mutate(child_diffs, rng, strength=1)
                    child_row = _diffs_to_row(child_diffs)
                    if child_row[0] == 0 and all(child_row[j] < child_row[j+1] for j in range(len(child_row)-1)):
                        child.append(child_row)
                    else:
                        child.append(list(p1[i]))
                new_population.append(child)
            else:
                # Mutation
                parent = rng.choice(survivors)
                child = []
                for i in range(n_rows):
                    if rng.random() < 0.5:
                        diffs = _row_to_diffs(parent[i])
                        diffs = _mutate(diffs, rng, strength=2)
                        row = _diffs_to_row(diffs)
                        if row[0] == 0 and all(row[j] < row[j+1] for j in range(len(row)-1)):
                            child.append(row)
                        else:
                            child.append(list(parent[i]))
                    else:
                        child.append(list(parent[i]))
                new_population.append(child)
        
        population = new_population
        
        # Elitism: ensure best is in population
        if best_fit < _fitness(population[0], k):
            population[0] = [list(r) for r in best_rows]
    
    return best_rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search: try to improve by adjusting individual elements.
    """
    best_rows = [list(r) for r in rows]
    best_fit = _fitness(rows, k)
    current_rows = [list(r) for r in rows]
    current_fit = best_fit
    
    T = 5.0
    cooling = 0.999
    
    while T > 0.05 and time.time() < deadline - 5:
        # Pick a random row and position
        r_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        
        old_val = current_rows[r_idx][pos]
        
        # Try a new value
        if rng.random() < 0.5:
            new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
        else:
            new_val = rng.randint(1, old_val + 10)
        
        if new_val == old_val:
            continue
        if pos > 0 and new_val <= current_rows[r_idx][pos-1]:
            continue
        if pos < k and new_val >= current_rows[r_idx][pos+1]:
            continue
        
        current_rows[r_idx][pos] = new_val
        new_fit = _fitness(current_rows, k)
        
        if new_fit < current_fit or rng.random() < math.exp(-(new_fit - current_fit) / max(T, 0.05)):
            current_fit = new_fit
            if new_fit < best_fit:
                best_fit = new_fit
                best_rows = [list(r) for r in current_rows]
        else:
            current_rows[r_idx][pos] = old_val
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # Initial greedy solution
    rows = _greedy_init(n_rows, k, rng)
    
    best_rows = [list(r) for r in rows]
    best_fit = _fitness(rows, k)
    
    # Phase 1: Local search
    if time.time() < deadline - 30:
        rows = _local_search(rows, n_rows, k, deadline - 30, rng)
        fit = _fitness(rows, k)
        if fit < best_fit:
            best_fit = fit
            best_rows = [list(r) for r in rows]
    
    # Phase 2: Genetic algorithm
    if time.time() < deadline - 5:
        rows = _genetic_algorithm(rows, n_rows, k, deadline - 5, rng)
        fit = _fitness(rows, k)
        if fit < best_fit:
            best_fit = fit
            best_rows = [list(r) for r in rows]
    
    # Final local search polish
    if time.time() < deadline - 2:
        rows = _local_search(best_rows, n_rows, k, deadline - 2, rng)
        fit = _fitness(rows, k)
        if fit < best_fit:
            best_fit = fit
            best_rows = [list(r) for r in rows]
    
    # Verify validity
    if not _is_valid(best_rows, k):
        # Fallback to greedy
        return _greedy_init(n_rows, k, rng)
    
    return best_rows
