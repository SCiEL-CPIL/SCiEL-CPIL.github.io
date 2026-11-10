import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set (n_rows, k) using Differential Evolution + Repair.
    Instead of greedy construction, we maintain a population of candidate sets
    and evolve them using DE crossover/mutation, with a repair function that
    ensures validity by greedily filling conflicts while minimizing the max entry.
    """
    deadline = time.time() + 120  # safety margin before 150s kill
    
    def valid(rows):
        if not rows:
            return False
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def score(rows):
        if rows is None:
            return float('inf')
        if not valid(rows):
            return float('inf')
        return max(max(row) for row in rows)

    def repair(row, used, max_val):
        """
        Greedily repair a row to be valid given the used differences.
        Returns the repaired row or None if impossible within max_val.
        """
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        return row

    def de_crossover(parent, mutant, cr):
        """Standard DE/1/bin crossover."""
        child = parent[:]
        j_rand = random.randint(0, len(parent) - 1)
        for i in range(len(parent)):
            if random.random() < cr or i == j_rand:
                child[i] = mutant[i]
        return child

    def de_mutation(x_r1, x_r2, x_r3, F):
        """Standard DE mutation: x_r1 + F * (x_r2 - x_r3)."""
        return [x_r1[i] + F * (x_r2[i] - x_r3[i]) for i in range(len(x_r1))]

    def flatten(rows):
        """Flatten list of rows into a single vector."""
        return [val for row in rows for val in row]

    def unflatten(vec):
        """Unflatten vector into list of rows."""
        rows = []
        for i in range(n_rows):
            rows.append(vec[i*(k+1):(i+1)*(k+1)])
        return rows

    def vector_to_rows(vec):
        """Convert a DE vector to valid rows using repair."""
        rows = []
        used = set()
        max_val = max(max(row) for row in unflatten(vec)) if vec else 100
        
        for i in range(n_rows):
            # Extract target row from vector
            target = vec[i*(k+1):(i+1)*(k+1)]
            # Ensure increasing and starts at 0
            target[0] = 0
            for j in range(1, len(target)):
                if target[j] <= target[j-1]:
                    target[j] = target[j-1] + 1
            
            # Repair this row
            repaired = repair(target, used, max_val + 10)
            if repaired is None:
                # Fallback: fully greedy repair
                repaired = repair([0] + [1000]*k, used, max_val + 100)
                if repaired is None:
                    return None
            rows.append(repaired)
        
        return rows

    # Initialize population
    pop_size = 20
    population = []
    
    # Seed with some greedy solutions
    for i in range(pop_size):
        rng = random.Random(i * 12345 + 678)
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
        population.append(flatten(rows))
    
    best_score = float('inf')
    best_rows = None
    
    # Get initial best
    for vec in population:
        rows = vector_to_rows(vec)
        if rows is not None and valid(rows):
            s = score(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
    
    # Differential Evolution
    F = 0.8  # Mutation scale
    CR = 0.5  # Crossover rate
    max_generations = 200
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        for i in range(pop_size):
            # Select 3 distinct individuals
            r1, r2, r3 = random.sample([j for j in range(pop_size) if j != i], 3)
            
            # Mutation
            mutant = de_mutation(population[r1], population[r2], population[r3], F)
            
            # Crossover
            trial = de_crossover(population[i], mutant, CR)
            
            # Evaluate trial
            trial_rows = vector_to_rows(trial)
            if trial_rows is not None and valid(trial_rows):
                trial_score = score(trial_rows)
                current_rows = vector_to_rows(population[i])
                current_score = score(current_rows) if current_rows else float('inf')
                
                if trial_score < current_score:
                    population[i] = trial
                    if trial_score < best_score:
                        best_score = trial_score
                        best_rows = [r[:] for r in trial_rows]
    
    # Local search on best solution
    if best_rows is not None and time.time() < deadline:
        rng = random.Random(42)
        current = [r[:] for r in best_rows]
        current_score = best_score
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Randomly pick a row and position to reduce
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            old_val = current[r_idx][pos]
            
            if old_val <= current[r_idx][pos-1] + 1:
                continue
            
            # Try to reduce
            for new_val in range(old_val - 1, current[r_idx][pos-1], -1):
                new_row = current[r_idx][:]
                new_row[pos] = new_val
                
                # Check validity
                used = set()
                valid_check = True
                for row in current:
                    if row is new_row:
                        row_to_check = new_row
                    else:
                        row_to_check = row
                    for i in range(len(row_to_check)):
                        for j in range(i + 1, len(row_to_check)):
                            d = row_to_check[j] - row_to_check[i]
                            if d in used:
                                valid_check = False
                                break
                            used.add(d)
                    if not valid_check:
                        break
                
                if valid_check:
                    new_score = max(max(row) for row in current)
                    if new_score < current_score:
                        current[r_idx][pos] = new_val
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [r[:] for r in current]
                        break
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: simple greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
