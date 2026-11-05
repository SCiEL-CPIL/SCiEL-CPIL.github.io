"""Constraint-satisfaction search for difference triangle sets."""
import time
import random
import numpy as np
from itertools import combinations

def _valid(rows):
    """Check that all positive intra-row differences are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows) if rows else 0

def _greedy_row(used, k):
    """Build one row greedily given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _row_diffs(row):
    """Compute all positive differences within a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _genetic_algorithm(rows, k, deadline, rng):
    """
    Genetic algorithm treating rows as rigid frames.
    - Crossover: swap row positions and try frame reassignment
    - Mutation: apply small integer shifts to individual entries
    - Repair: greedily resolve conflicts by rebuilding conflicting rows
    """
    n_rows = len(rows)
    pop_size = 12
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Precompute diff sets for each row
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    # Population: each individual is (rows, scope)
    population = []
    for i in range(pop_size):
        if i == 0:
            population.append(([list(r) for r in rows], best_scope))
        else:
            # Create diverse initial individuals
            used = set()
            new_rows = []
            for _ in range(n_rows):
                row = _greedy_row(used, k)
                new_rows.append(row)
            population.append((new_rows, _scope(new_rows)))
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate fitness and select parents (tournament selection)
        def tournament_select(pop, k_t=3):
            candidates = rng.sample(pop, min(k_t, len(pop)))
            return min(candidates, key=lambda x: x[1])[0]
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best
        best_in_pop = min(population, key=lambda x: x[1])
        new_population.append((best_in_pop[0], best_in_pop[1]))
        
        while len(new_population) < pop_size and time.time() < deadline:
            # Select parents
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: swap random subset of rows
            child_rows = [list(r) for r in parent1]
            if rng.random() < 0.7:
                # Swap 1-3 random rows between parents
                n_swap = rng.randint(1, min(3, n_rows))
                swap_indices = rng.sample(range(n_rows), n_swap)
                for idx in swap_indices:
                    child_rows[idx] = list(parent2[idx])
            
            # Mutation: apply small shifts to random entries
            if rng.random() < 0.6:
                n_mutations = rng.randint(1, 3)
                for _ in range(n_mutations):
                    row_idx = rng.randrange(n_rows)
                    row = child_rows[row_idx]
                    if len(row) < 2:
                        continue
                    pos = rng.randrange(1, len(row))
                    shift = rng.randint(-5, 5)
                    new_val = row[pos] + shift
                    # Check bounds
                    if pos > 0 and new_val <= row[pos-1]:
                        continue
                    if pos < len(row)-1 and new_val >= row[pos+1]:
                        continue
                    row[pos] = new_val
                    row.sort()
            
            # Repair: check validity, rebuild conflicting rows
            used = set()
            valid = True
            for i, row in enumerate(child_rows):
                for j in range(len(row)):
                    for m in range(j+1, len(row)):
                        d = row[m] - row[j]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                # Repair: rebuild rows that conflict
                # Find which rows have conflicts
                used = set()
                for i, row in enumerate(child_rows):
                    row_conflict = False
                    for j in range(len(row)):
                        for m in range(j+1, len(row)):
                            d = row[m] - row[j]
                            if d in used:
                                row_conflict = True
                                break
                            used.add(d)
                        if row_conflict:
                            break
                    if row_conflict:
                        # Rebuild this row
                        # Remove its diffs from used
                        for j in range(len(row)):
                            for m in range(j+1, len(row)):
                                used.discard(row[m] - row[j])
                        child_rows[i] = _greedy_row(used, k)
                        for j in range(len(child_rows[i])):
                            for m in range(j+1, len(child_rows[i])):
                                used.add(child_rows[i][m] - child_rows[i][j])
            
            # Final validation
            used = set()
            valid = True
            for row in child_rows:
                for j in range(len(row)):
                    for m in range(j+1, len(row)):
                        d = row[m] - row[j]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback: use parent1
                child_rows = [list(r) for r in parent1]
            
            child_scope = _scope(child_rows)
            new_population.append((child_rows, child_scope))
        
        population = new_population
        
        # Track best
        pop_best = min(population, key=lambda x: x[1])
        if pop_best[1] < best_scope:
            best_scope = pop_best[1]
            best_rows = pop_best[0]
        
        # Periodic refresh: add fresh greedy solutions
        if generation % 10 == 0 and time.time() < deadline:
            used = set()
            fresh_rows = []
            for _ in range(n_rows):
                row = _greedy_row(used, k)
                fresh_rows.append(row)
            fresh_scope = _scope(fresh_rows)
            if fresh_scope < population[-1][1]:
                population[-1] = (fresh_rows, fresh_scope)
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy initial solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Genetic algorithm
    while time.time() < deadline:
        candidate = _genetic_algorithm(best_rows, k, min(time.time() + 25, deadline), rng)
        scope = _scope(candidate)
        if scope < best_scope:
            best_scope = scope
            best_rows = candidate
        
        # Occasionally restart from a fresh greedy solution
        if rng.random() < 0.25:
            used = set()
            fresh_rows = []
            for _ in range(n_rows):
                row = _greedy_row(used, k)
                fresh_rows.append(row)
            fresh_scope = _scope(fresh_rows)
            if fresh_scope < best_scope:
                best_scope = fresh_scope
                best_rows = [list(r) for r in fresh_rows]
    
    # Final validation
    if _valid(best_rows):
        return best_rows
    
    # Fallback: return greedy solution (should always be valid)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows
