"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
from collections import defaultdict

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_cost(rows, k):
    """
    Compute a tuple (scope, collisions) for ranking.
    Collisions is the number of times a difference is repeated.
    """
    scope = _get_scope(rows)
    diff_counts = defaultdict(int)
    for row in rows:
        for d in _row_diffs(row):
            diff_counts[d] += 1
    
    collisions = sum(c - 1 for c in diff_counts.values() if c > 1)
    return (scope, collisions)

def _greedy_init(n_rows, k):
    """Greedy initialization: build rows one by one with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _repair_row(row, used, k, max_val, deadline):
    """
    Attempt to repair a row to fit within max_val and avoid 'used' diffs.
    Uses a simple backtracking search.
    """
    n = k + 1
    new_row = [0]
    
    def backtrack(pos, current_max):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        if max_val - current_max < (n - pos - 1):
            return False
        
        start = current_max + 1
        end = min(max_val, max_val - (n - pos - 1))
        
        # Randomize order to escape local traps
        candidates = list(range(start, end + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            ok = True
            for i in range(pos):
                d = v - new_row[i]
                if d in used:
                    ok = False
                    break
            if not ok:
                continue
            new_row.append(v)
            if backtrack(pos + 1, v):
                return True
            new_row.pop()
        return False
    
    if backtrack(1, 0):
        return new_row
    return None

def _local_search(rows, n_rows, k, deadline):
    """
    Fast local search using simulated annealing logic to minimize (scope, collisions).
    """
    current_scope, current_collisions = _compute_cost(rows, k)
    best_rows = [list(r) for r in rows]
    best_cost = (current_scope, current_collisions)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Build used set for quick diff checks
    used = set()
    for row in rows:
        used |= _row_diffs(row)
        
    while T > min_T and time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Propose a new value
        # Bias towards smaller values to reduce scope
        if random.random() < 0.6:
            new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
        else:
            # Allow some expansion to escape local minima
            new_val = random.randint(1, old_val + 5)
            
        if new_val == old_val:
            T *= cooling
            continue
            
        if pos > 0 and new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            T *= cooling
            continue
            
        # Compute old diffs
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - old_val)
            
        # Compute new diffs
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)
            
        # Check collision change
        # A diff is "free" if it was in old_diffs and is not in new_diffs and not in used (outside this row)
        # A diff is "colliding" if it is in new_diffs and in used (outside this row) or in new_diffs multiple times (internal)
        
        # Simplified check: 
        # Remove old_diffs from used, add new_diffs. Count collisions.
        # This is expensive. Let's approximate or do exact if fast.
        # For small k, exact is fine.
        
        # To speed up, we can just check if new_diffs intersects with (used - old_diffs)
        # If yes, collisions might increase.
        # If no, collisions might decrease or stay same.
        
        # Let's just apply and compute cost if it looks promising or random.
        # Actually, computing full cost is O(N*K^2). With N=7, K=6, it's very fast.
        
        row[pos] = new_val
        used = (used - old_diffs) | new_diffs
        new_scope, new_collisions = _compute_cost(rows, k)
        new_cost = (new_scope, new_collisions)
        
        # Acceptance criteria
        accept = False
        if new_cost < best_cost:
            accept = True
        else:
            # Calculate delta
            # We want to minimize scope primarily, collisions secondarily.
            # Delta scope = new_scope - current_scope
            # Delta collisions = new_collisions - current_collisions
            # Composite delta: 100 * delta_scope + delta_collisions
            delta = 100 * (new_scope - current_scope) + (new_collisions - current_collisions)
            if delta <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta / T):
                    accept = True
        
        if accept:
            current_scope, current_collisions = new_scope, new_collisions
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in rows]
        else:
            row[pos] = old_val
            used = (used - new_diffs) | old_diffs
            
        T *= cooling
        
    return best_rows

def _ea_optimize(rows, n_rows, k, deadline):
    """
    Evolutionary Algorithm optimization.
    """
    pop_size = 20
    # Initialize population
    population = [rows]
    # Add some mutated versions of the initial solution
    for _ in range(pop_size - 1):
        new_pop = [list(r) for r in rows]
        # Random mutations
        for _ in range(5):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(new_pop[r_idx]) - 1)
            new_val = random.randint(1, _get_scope(new_pop))
            if pos > 0 and new_val > new_pop[r_idx][pos-1] and (pos == len(new_pop[r_idx])-1 or new_val < new_pop[r_idx][pos+1]):
                new_pop[r_idx][pos] = new_val
        population.append(new_pop)
        
    best_rows = [list(r) for r in rows]
    best_cost = _compute_cost(rows, k)
    
    # Sort population
    population.sort(key=lambda r: _compute_cost(r, k))
    
    while time.time() < deadline:
        # Selection and Crossover
        parent1 = population[0]
        parent2 = population[random.randint(0, min(5, len(population)-1))]
        
        # Crossover: Mix rows
        child = []
        for i in range(n_rows):
            if random.random() < 0.5:
                child.append(list(parent1[i]))
            else:
                child.append(list(parent2[i]))
                
        # Mutation
        for _ in range(10):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(child[r_idx]) - 1)
            old_val = child[r_idx][pos]
            new_val = random.randint(1, _get_scope(child) + 10)
            if pos > 0 and new_val > child[r_idx][pos-1] and (pos == len(child[r_idx])-1 or new_val < child[r_idx][pos+1]):
                child[r_idx][pos] = new_val
                
        # Repair if invalid? 
        # Check validity
        if not _is_valid(child):
            # Try to repair by local search or rebuild
            # For simplicity, discard if invalid and rely on mutation to find valid ones
            # Or, try to repair the specific row with conflicts
            # Let's just discard for now to keep EA simple, or use local search to fix.
            # Local search is better.
            pass # Skip repair for now, let local search handle it on valid ones? 
            # Actually, EA works best if we only keep valid individuals or penalize heavily.
            # Let's use the cost function which penalizes collisions.
            # So we keep it.
            
        # Local search on the child
        if time.time() < deadline:
            child = _local_search(child, n_rows, k, deadline)
            
        child_cost = _compute_cost(child, k)
        
        # Insert into population
        population.append(child)
        population.sort(key=lambda r: _compute_cost(r, k))
        
        # Keep only best pop_size
        population = population[:pop_size]
        
        # Update best
        if population[0][0] if isinstance(population[0], tuple) else _compute_cost(population[0], k) < best_cost:
             # Note: population contains rows, not costs.
             pass
        current_best_cost = _compute_cost(population[0], k)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_rows = [list(r) for r in population[0]]
            
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with greedy solution
    rows = _greedy_init(n_rows, k)
    
    # Verify validity
    if not _is_valid(rows):
        return rows
        
    # Phase 1: Local Search to polish the greedy solution
    rows = _local_search(rows, n_rows, k, deadline)
    
    # Phase 2: Evolutionary Algorithm for global exploration
    if time.time() < deadline - 10:
        rows = _ea_optimize(rows, n_rows, k, deadline)
        
    # Final validation
    if not _is_valid(rows):
        return _greedy_init(n_rows, k)
    
    return rows
# EVOLVE-BLOCK-END
