"""Program for Difference Triangle Set (4,7) - Genetic Algorithm with Bitmask Constraints."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_scope(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def compute_row_diffs(row):
        """Compute set of differences for a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_row_valid(row):
        """Check if a row is strictly increasing and starts at 0."""
        if not row or row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True

    def greedy_construct_row(max_val, forbidden_diffs, rng):
        """
        Construct a single row greedily, avoiding forbidden differences.
        Returns the row or None if impossible within max_val.
        """
        row = [0]
        used_in_row = set()
        
        for _pos in range(k):
            found = False
            next_val = row[-1] + 1
            # Limit search space to avoid infinite loops, but max_val is the hard cap
            while next_val <= max_val:
                valid = True
                for prev in row:
                    diff = next_val - prev
                    if diff in forbidden_diffs or diff in used_in_row:
                        valid = False
                        break
                if valid:
                    row.append(next_val)
                    for prev in row[:-1]:
                        used_in_row.add(next_val - prev)
                    found = True
                    break
                next_val += 1
            if not found:
                return None
        return row

    def random_row(max_val, forbidden_diffs, rng):
        """
        Generate a random valid row by picking random gaps, ensuring no forbidden diffs.
        This is slower but provides diversity.
        """
        attempts = 0
        while attempts < 100:
            row = [0]
            valid = True
            used_in_row = set()
            current_val = 0
            for _pos in range(k):
                # Pick a random gap
                # Max possible gap is max_val - current_val
                # Min gap is 1
                max_gap = max_val - current_val
                if max_gap <= 0:
                    valid = False
                    break
                gap = rng.randint(1, max(1, min(20, max_gap))) # Bias towards small gaps for feasibility
                next_val = current_val + gap
                # Check diffs
                for prev in row:
                    diff = next_val - prev
                    if diff in forbidden_diffs or diff in used_in_row:
                        valid = False
                        break
                if not valid:
                    break
                row.append(next_val)
                for prev in row[:-1]:
                    used_in_row.add(next_val - prev)
                current_val = next_val
            
            if valid and is_row_valid(row):
                return row
            attempts += 1
        return None

    def evaluate_population(population):
        """
        population: list of lists of rows.
        Returns list of (scope, valid_bool) for each individual.
        """
        results = []
        for ind in population:
            all_diffs = set()
            valid = True
            scope = 0
            for row in ind:
                if not is_row_valid(row):
                    valid = False
                    break
                diffs = compute_row_diffs(row)
                if len(diffs) != diffs_per_row:
                    valid = False
                    break
                # Check conflict with existing
                if all_diffs & diffs:
                    valid = False
                    break
                all_diffs |= diffs
                scope = max(scope, max(row))
            
            if valid:
                results.append((scope, True))
            else:
                results.append((get_scope(ind) if ind else float('inf'), False))
        return results

    def crossover(parent1, parent2, rng):
        """
        Crossover: For each row position, pick a row from one parent.
        Then, try to repair conflicts by regenerating rows that conflict.
        """
        child = []
        for i in range(n_rows):
            if rng.random() < 0.5:
                child.append(parent1[i][:])
            else:
                child.append(parent2[i][:])
        
        # Repair phase: Check for global conflicts
        # We need to ensure all diffs are unique.
        # Simple repair: iterate rows, if a row's diffs conflict with previous, regenerate it.
        # This is heuristic and might fail, but GA handles invalid individuals via fitness.
        
        return child

    def mutate(ind, max_val, rng):
        """
        Mutate: Pick a random row and regenerate it or tweak it.
        """
        if rng.random() < 0.1: # 10% chance to mutate
            row_idx = rng.randint(0, n_rows - 1)
            
            # Collect forbidden diffs from OTHER rows
            forbidden = set()
            for i in range(n_rows):
                if i != row_idx:
                    forbidden |= compute_row_diffs(ind[i])
            
            # Try to regenerate the row
            new_row = greedy_construct_row(max_val, forbidden, rng)
            if new_row is None:
                # Fallback: random
                new_row = random_row(max_val, forbidden, rng)
            
            if new_row is not None:
                ind[row_idx] = new_row
        return ind

    # --- Main Logic ---
    
    # Phase 1: Initialize Population
    # Start with a few greedy solutions to have a valid baseline
    initial_max = total_diffs + 50
    
    base_rows = []
    used_diffs_global = set()
    for _ in range(n_rows):
        # Greedy construction for initial valid set
        row = [0]
        for _pos in range(k):
            found = False
            next_val = row[-1] + 1
            while next_val <= initial_max:
                valid = True
                for prev in row:
                    diff = next_val - prev
                    if diff in used_diffs_global:
                        valid = False
                        break
                if valid:
                    row.append(next_val)
                    for prev in row[:-1]:
                        used_diffs_global.add(next_val - prev)
                    found = True
                    break
                next_val += 1
            if not found:
                # Fallback if greedy fails (shouldn't happen with large enough max)
                row = [0] + list(range(1, k+1)) * 100 # Dummy, will be fixed
                break
        base_rows.append(row)
    
    if any(len(r) != k+1 for r in base_rows):
        # Hard fallback
        base_rows = []
        for i in range(n_rows):
            base_rows.append([j * (i+1) for j in range(k+1)])

    best_scope = get_scope(base_rows)
    best_individual = [r[:] for r in base_rows]

    # Create initial population
    pop_size = 20
    population = [best_individual]
    
    # Add variations
    rng = random.Random(42)
    for _ in range(pop_size - 1):
        ind = [r[:] for r in base_rows]
        # Mutate a few rows
        for _ in range(2):
            row_idx = rng.randint(0, n_rows - 1)
            forbidden = set()
            for i in range(n_rows):
                if i != row_idx:
                    forbidden |= compute_row_diffs(ind[i])
            
            # Try greedy with random start bias? No, just greedy is fast.
            # To get diversity, we can perturb the greedy choice.
            # For now, just use greedy to ensure validity, then mutate later.
            # Actually, let's try random_row for diversity
            new_row = random_row(best_scope + 20, forbidden, rng)
            if new_row:
                ind[row_idx] = new_row
        population.append(ind)

    # GA Loop
    gen = 0
    no_improve_count = 0
    current_max_val = best_scope + 20
    
    while time.time() < deadline:
        gen += 1
        
        # Evaluate
        scores = evaluate_population(population)
        
        # Find best
        best_idx = -1
        best_score = float('inf')
        for i, (scope, valid) in enumerate(scores):
            if valid and scope < best_score:
                best_score = scope
                best_idx = i
        
        if best_idx != -1:
            if best_score < best_scope:
                best_scope = best_score
                best_individual = [r[:] for r in population[best_idx]]
                no_improve_count = 0
                # Tighten max_val for future mutations to encourage smaller values
                current_max_val = max(best_scope + 5, total_diffs) 
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Selection: Keep top half
        # Sort by score (lower is better), invalids last
        sorted_indices = sorted(range(pop_size), key=lambda i: (scores[i][1], scores[i][0]))
        # Note: valid=True is better than valid=False. 
        # Let's define a sort key: (0 if valid else 1, scope)
        sorted_indices = sorted(range(pop_size), key=lambda i: (0 if scores[i][1] else 1, scores[i][0]))
        
        survivors = [population[i] for i in sorted_indices[:pop_size//2]]
        
        # Crossover & Mutation to create next gen
        next_population = [ind[:] for ind in survivors] # Clone survivors
        
        while len(next_population) < pop_size:
            p1 = survivors[rng.randint(0, len(survivors)-1)]
            p2 = survivors[rng.randint(0, len(survivors)-1)]
            
            child = crossover(p1, p2, rng)
            child = mutate(child, current_max_val, rng)
            
            # Check if child is valid immediately? 
            # If not, it will be penalized in next eval.
            next_population.append(child)
        
        population = next_population
        
        # If we are stuck for a while, try a "hard" reset for some individuals
        if no_improve_count > 50 and time.time() < deadline - 10:
             # Inject random fresh individuals
             for i in range(5):
                 ind = [r[:] for r in base_rows]
                 row_idx = rng.randint(0, n_rows - 1)
                 forbidden = set()
                 for j in range(n_rows):
                     if j != row_idx:
                         forbidden |= compute_row_diffs(ind[j])
                 new_row = random_row(current_max_val, forbidden, rng)
                 if new_row:
                     ind[row_idx] = new_row
                 population[i] = ind

    return best_individual

# EVOLVE-BLOCK-END
