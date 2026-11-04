"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import collections

# EVOLVE-BLOCK-START

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    m = 0
    for r in rows:
        if r and r[-1] > m:
            m = r[-1]
    return m


def _diffs_of_row(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_used(rows):
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _greedy_row(used, k):
    """Greedily build one row of k+1 increasing ints starting at 0."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        rows.append(row)
        used |= new_diffs
    return rows


def _randomized_greedy_row(used, k, rng, pool_size=5):
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs


def _randomized_greedy(n_rows, k, rng, pool_size=5):
    best_rows = None
    best_scope = float('inf')
    deadline = time.time() + 100
    for _ in range(50):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                ok = False
                break
            rows.append(row)
            used |= new_diffs
        if ok:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _build_diff_counter(rows):
    """Build a Counter of all pairwise differences across all rows."""
    counter = collections.Counter()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                counter[r[j] - r[i]] += 1
    return counter


def _count_bad(counter):
    """Count how many differences have count != 1."""
    return sum(1 for v in counter.values() if v != 1)


def _is_valid(rows):
    counter = _build_diff_counter(rows)
    return _count_bad(counter) == 0


def _shrink_row(row, used_diffs, k):
    """Try to shrink a row by reducing elements from the end."""
    row = list(row)
    m = len(row)
    for pos in range(m - 1, 0, -1):
        for new_val in range(row[pos] - 1, row[pos - 1], -1):
            # Check if replacing row[pos] with new_val creates any duplicate diffs
            ok = True
            for i in range(m):
                if i == pos:
                    continue
                d = abs(row[i] - new_val)
                if d in used_diffs:
                    ok = False
                    break
            if ok:
                # Check internal consistency within the row
                for i in range(m):
                    if i == pos:
                        continue
                    for j in range(m):
                        if i == j or j == pos:
                            continue
                        if abs(row[i] - new_val) == abs(row[j] - row[i]) and i < j:
                            # This is a simplified check; full check done in caller
                            pass
                row[pos] = new_val
                break
    return row


def _difference_swap_mutation(rows, rng, max_swaps=20):
    """
    Perform difference-swap mutations: for each row, try to reduce elements
    by checking if the new differences don't conflict with others.
    Returns a new list of rows (copy).
    """
    rows = [list(r) for r in rows]
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    # Build global diff set for conflict checking
    all_diffs = set()
    for r in rows:
        all_diffs |= _diffs_of_row(r)
    
    # Try to reduce elements in each row, starting with the largest rows
    row_order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
    
    swaps_done = 0
    for _ in range(max_swaps):
        if swaps_done >= max_swaps:
            break
        for ri in row_order:
            row = rows[ri]
            m = len(row)
            # Try to reduce the last element first, then work backwards
            for pos in range(m - 1, 0, -1):
                if pos == 0:
                    continue
                old_val = row[pos]
                # Try reducing by small amounts
                for delta in range(1, 5):
                    new_val = old_val - delta
                    if new_val <= row[pos - 1]:
                        continue
                    # Check if this creates conflicts
                    # Remove old diffs involving position pos
                    old_diffs = set()
                    for i in range(m):
                        if i != pos:
                            old_diffs.add(abs(row[i] - old_val))
                    
                    # Add new diffs involving position pos
                    new_diffs = set()
                    for i in range(m):
                        if i != pos:
                            new_diffs.add(abs(row[i] - new_val))
                    
                    # Check: new_diffs should not intersect with (all_diffs - old_diffs)
                    # And new_diffs should not have internal duplicates
                    remaining = all_diffs - old_diffs
                    if new_diffs & remaining:
                        continue
                    if len(new_diffs) != len(new_diffs):  # always true, skip
                        continue
                    # Check internal consistency: all diffs in the row must be unique
                    # Build the full set of diffs for this row with the change
                    test_row = list(row)
                    test_row[pos] = new_val
                    row_diffs = set()
                    ok = True
                    for i in range(m):
                        for j in range(i + 1, m):
                            d = test_row[j] - test_row[i]
                            if d in row_diffs:
                                ok = False
                                break
                            row_diffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Check against other rows
                    other_diffs = set()
                    for oi in range(n_rows):
                        if oi == ri:
                            continue
                        other_diffs |= _diffs_of_row(rows[oi])
                    if row_diffs & other_diffs:
                        continue
                    
                    # Valid move!
                    row[pos] = new_val
                    # Update all_diffs
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    swaps_done += 1
                    break
                if swaps_done >= max_swaps:
                    break
            if swaps_done >= max_swaps:
                break
    
    return rows


def _partial_row_crossover(parent1, parent2, rng, k):
    """
    Crossover: for each row, take the first half from one parent and the 
    second half from the other, then repair by greedy completion.
    """
    n_rows = len(parent1)
    child = []
    used = set()
    
    for i in range(n_rows):
        # Decide which parent to take from for this row
        if rng.random() < 0.5:
            source = parent1[i]
            other = parent2[i]
        else:
            source = parent2[i]
            other = parent1[i]
        
        # Take first ceil(k/2) elements from source
        half = (k + 1) // 2
        partial = list(source[:half])
        
        # Complete the row greedily, avoiding used diffs and partial's own diffs
        partial_diffs = set()
        for i2 in range(len(partial)):
            for j2 in range(i2 + 1, len(partial)):
                partial_diffs.add(partial[j2] - partial[i2])
        
        row = list(partial)
        x = partial[-1] if partial else 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used) and not (new & partial_diffs):
                row.append(x)
                partial_diffs |= new
            else:
                # Try skipping
                continue
        
        # If we couldn't complete, fall back to greedy from scratch
        if len(row) < k + 1:
            row = [0]
            x = 0
            new_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & new_diffs):
                    row.append(x)
                    new_diffs |= new
            row_diffs = new_diffs
        else:
            row_diffs = partial_diffs
        
        child.append(row)
        used |= row_diffs
    
    return child


def _evolutionary_search(n_rows, k, rng, deadline, initial_rows):
    """
    Multi-objective evolutionary algorithm for difference triangles.
    Population of complete row sets, with difference-swap mutation and partial-row crossover.
    """
    pop_size = 12
    deadline_inner = deadline - 2
    
    # Initialize population
    population = []
    # First individual: the best initial rows
    population.append([list(r) for r in initial_rows])
    
    # Generate diverse initial population
    for _ in range(pop_size - 1):
        if time.time() > deadline_inner:
            break
        rg_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
        if rg_rows is not None:
            population.append(rg_rows)
    
    if len(population) < 2:
        return initial_rows
    
    best_scope = min(_total_scope(r) for r in population)
    best_individual = [list(r) for r in min(population, key=_total_scope)]
    
    generation = 0
    while time.time() < deadline_inner:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitnesses = []
        for ind in population:
            scope = _total_scope(ind)
            # Secondary objective: sum of all entries (to break ties and maintain diversity)
            total_sum = sum(sum(r) for r in ind)
            fitnesses.append((scope, total_sum))
        
        # Select parents using tournament selection on the primary objective (scope)
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = min(range(len(population)), key=lambda i: fitnesses[i][0])
        new_population.append([list(r) for r in population[best_idx]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline_inner:
                break
            
            # Tournament selection for two parents
            def tournament():
                idxs = rng.sample(range(len(population)), min(3, len(population)))
                return min(idxs, key=lambda i: fitnesses[i][0])
            
            p1_idx = tournament()
            p2_idx = tournament()
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover or mutation
            if rng.random() < 0.4:
                child = _partial_row_crossover(p1, p2, rng, k)
            else:
                # Pick one parent and mutate
                parent = p1 if rng.random() < 0.5 else p2
                child = _difference_swap_mutation(parent, rng, max_swaps=rng.randint(10, 30))
            
            # Validate and repair if needed
            if not _is_valid(child):
                # Try to repair by greedy reconstruction of invalid rows
                used = set()
                repaired = []
                for i in range(n_rows):
                    row = child[i]
                    row_diffs = _diffs_of_row(row)
                    if row_diffs & used:
                        # Rebuild this row greedily
                        row = [0]
                        new_diffs = set()
                        x = 0
                        while len(row) < k + 1:
                            x += 1
                            new = {x - a for a in row}
                            if not (new & used) and not (new & new_diffs):
                                row.append(x)
                                new_diffs |= new
                        row_diffs = new_diffs
                    repaired.append(row)
                    used |= row_diffs
                child = repaired
            
            if _is_valid(child):
                new_population.append(child)
        
        # Update population
        population = new_population[:pop_size]
        
        # Track best
        for ind in population:
            scope = _total_scope(ind)
            if scope < best_scope:
                best_scope = scope
                best_individual = [list(r) for r in ind]
        
        # If we've hit the theoretical lower bound, stop
        # Counting bound: need n_rows * (k+1)*k/2 distinct differences
        # Max diff in scope m is m, so we need m >= n_rows * (k+1)*k/2
        min_possible = n_rows * (k + 1) * k // 2
        if best_scope <= min_possible:
            break
    
    return best_individual


def _local_search(rows, k, rng, max_iterations=5000):
    """Pure-descent local search: shrink the largest elements greedily."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    for _ in range(max_iterations):
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        if len(row) < 2:
            break
        last = row[-1]
        found = False
        for new_last in range(last - 1, row[-2], -1):
            old_diffs = {last - a for a in row[:-1]}
            new_diffs = {new_last - a for a in row[:-1]}
            if not (new_diffs & (used - old_diffs)):
                row[-1] = new_last
                used = (used - old_diffs) | new_diffs
                found = True
                break
        if not found:
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                f2 = False
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        old_diffs.add(abs(row[i] - old_val))
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        new_diffs.add(abs(row[i] - new_val))
                    if not (new_diffs & (used - old_diffs)):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        f2 = True
                        break
                if f2:
                    break
            if not f2:
                break
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    Uses a multi-objective evolutionary algorithm with difference-swap mutations.
    """
    rng = random.Random(42)

    # Initial construction
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    deadline = time.time() + 145  # safety margin under the 150s kill
    best_rows = rows
    best_scope = _total_scope(rows)

    # Phase 1: Quick local search to improve initial solution
    if time.time() < deadline - 60:
        ls_rows = _local_search(best_rows, k, rng, max_iterations=3000)
        scope = _total_scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]

    # Phase 2: Evolutionary search (main structural change)
    if time.time() < deadline - 10:
        ea_rows = _evolutionary_search(n_rows, k, rng, deadline, best_rows)
        scope = _total_scope(ea_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ea_rows]

    # Phase 3: Final local search polish
    if time.time() < deadline - 3:
        ls_rows = _local_search(best_rows, k, rng, max_iterations=5000)
        scope = _total_scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]

    # Final validation
    if not _is_valid(best_rows):
        # Fallback to greedy if something went wrong
        best_rows = _greedy_construct(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
