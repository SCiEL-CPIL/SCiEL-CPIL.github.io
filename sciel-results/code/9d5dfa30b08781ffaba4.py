"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0
    
    # 1. Generate a baseline greedy solution to have a valid fallback
    baseline = _greedy_construct(n_rows, k)
    best_rows = [row[:] for row in baseline]
    best_score = _max_entry(best_rows)
    
    if n_rows <= 2 and k <= 3:
        return best_rows

    # 2. Simulated Annealing Local Search
    # State: list of n_rows rows.
    # Move: Replace one row with a new randomly generated valid row.
    # Cost: max_entry(rows).
    # Acceptance: Metropolis criterion.
    
    current_rows = [row[:] for row in baseline]
    current_score = best_score
    
    # Temperature schedule
    start_temp = 20.0
    end_temp = 0.01
    # We want to run as many steps as possible within the time limit.
    # Estimate step cost: Generating a random row is O(k^2 * range). 
    # Checking validity is O(n_rows * k^2).
    # Let's aim for ~50,000 steps if possible, but guard with time.
    
    t = start_temp
    steps = 0
    
    # Pre-compute some random row templates? 
    # No, generating on the fly is better for diversity, but we need them to be "small" 
    # to have a chance of improving the max entry.
    # Strategy for random row generation:
    # Pick a random max_limit (e.g., current_score or slightly less).
    # Generate a random increasing sequence of length k+1 starting at 0 within [0, max_limit].
    # Check if its internal differences are unique (always true for a set? No, differences must be distinct *within* the row? 
    # Wait, the problem says "all differences between two entries of the same row are distinct over all rows".
    # This implies:
    # 1. Differences within a single row must be distinct? 
    #    "all differences between two entries of the same row are distinct over all rows"
    #    Usually, difference sets require distinct differences *within* the set as well (Sidon sets / Golomb rulers).
    #    Let's verify: If a row has {0, 1, 2}, diffs are 1, 2, 1. Not distinct.
    #    So each row must be a Golomb ruler (distinct pairwise differences).
    #    AND the differences across all rows must be distinct from each other.
    
    # So, a valid row is a Golomb ruler.
    # The random generator must produce Golomb rulers.
    
    while time.time() < deadline - 5:
        # Decay temperature
        # Simple exponential decay based on steps or time? 
        # Let's use a fixed number of "epochs" or just linear decay over total expected steps.
        # Better: Decay based on time remaining or step count.
        # Let's try a simple linear decay from start_temp to end_temp over 10,000 steps.
        if steps < 10000:
            t = start_temp * (1 - steps / 10000.0) + end_temp
        else:
            t = end_temp
            
        steps += 1
        
        # Pick a random row to perturb
        row_idx = random.randint(0, n_rows - 1)
        
        # Generate a candidate row
        # We want a row with max entry <= current_score (ideally smaller).
        # Let's try to generate a row with max entry in [0, current_score].
        # If we can't find a valid Golomb ruler in that range, we might have to accept a larger one?
        # No, we want to minimize the global max. So we should only consider rows with max <= current_score.
        # If we replace a row that had max > new_max, the global max might drop.
        # If we replace a row that had max == global_max, and the new row has max < global_max, global max drops.
        
        candidate_row = _generate_random_golomb_ruler(k, max_val=current_score)
        
        if candidate_row is None:
            continue
            
        # Construct new state
        new_rows = current_rows[:]
        new_rows[row_idx] = candidate_row
        
        # Check global validity: 
        # 1. Each row is a Golomb ruler (guaranteed by generator).
        # 2. No difference conflicts between any pair of rows.
        
        # Optimization: Instead of checking all pairs, we can maintain a set of used differences.
        # But since we are replacing one row, we can:
        # - Remove differences of old row from global set.
        # - Check if new row's differences conflict with remaining global set.
        # - If no conflict, add new row's differences.
        
        # For simplicity and correctness in this short script, let's just check the whole set.
        # n_rows=7, k=5. Pairs in row = 15. Total diffs = 105.
        # Checking 105 ints for duplicates is fast.
        
        if not _is_valid(new_rows, n_rows, k):
            continue
            
        new_score = _max_entry(new_rows)
        
        # Calculate delta
        delta = new_score - current_score
        
        # Metropolis Acceptance
        if delta < 0 or (t > 0 and random.random() < math.exp(-delta / t)):
            current_rows = new_rows
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_rows = current_rows[:]
                
                # Early termination if we hit the known lower bound or a very good score
                if best_score <= 105:
                    break
                if best_score <= 110:
                    # Keep searching a bit more, but we are doing great
                    pass

    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _generate_random_golomb_ruler(k, max_val):
    """
    Generates a random Golomb ruler of length k+1 (k+1 marks) starting at 0,
    with max entry <= max_val.
    Returns None if no such ruler is found within a few tries.
    """
    # A Golomb ruler requires all pairwise differences to be distinct.
    # For k=5 (6 marks), we need 15 distinct differences.
    # The minimum max_val for a Golomb ruler of 6 marks is 17 (0, 1, 4, 10, 12, 17).
    # If max_val is too small, it's impossible.
    
    # Try a few random permutations or random subsets
    for _ in range(20):
        # Randomly pick k distinct integers from 1 to max_val
        # Ensure 0 is the first.
        if max_val < 17: # Heuristic minimum for 6 marks
            return None
            
        # Sample k-1 intermediate points and 1 final point?
        # Simpler: Sample k distinct points from 1..max_val, sort them, prepend 0.
        # But they must be distinct.
        candidates = random.sample(range(1, max_val + 1), k)
        candidates.sort()
        row = [0] + candidates
        
        # Check if it's a Golomb ruler
        diffs = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if valid:
            return row
            
    return None

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check internal differences (Golomb property)
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False
                row_diffs.add(d)
        
        # Check global uniqueness
        for d in row_diffs:
            if d in used:
                return False
            used.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)
# EVOLVE-BLOCK-END
