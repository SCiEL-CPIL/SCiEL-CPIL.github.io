import time
import random
import sys

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 entries each."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Number of differences per row
    num_diffs = k * (k + 1) // 2  # for k=5, this is 15
    total_diffs = n_rows * num_diffs  # 7 * 15 = 105
    
    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Fast greedy to get initial solution
    baseline = _greedy_fast(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)
    
    # Phase 2: Interleaved greedy with randomization
    for attempt in range(300):
        if time.time() > deadline - 20:
            break
        rows = _interleaved_greedy(n_rows, k, seed=attempt)
        m = max(max(r) for r in rows)
        if m < best_max:
            best_max = m
            best_rows = rows
    
    # Phase 3: Local search / hill climbing on individual rows
    if time.time() < deadline - 5:
        best_rows, best_max = _local_search_improved(best_rows, n_rows, k, best_max, deadline)
    
    return best_rows


def _greedy_fast(n_rows, k):
    """Fast standard greedy construction."""
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


def _interleaved_greedy(n_rows, k, seed=0):
    """
    Interleaved greedy: build all rows simultaneously, one element at a time.
    At each step, for each row, find the smallest valid next element.
    This can lead to more balanced usage of the difference space.
    """
    rng = random.Random(seed)
    used = set()
    rows = [[0] for _ in range(n_rows)]
    positions = [0] * n_rows  # current last element in each row
    
    # We need to add k more elements to each row
    for step in range(k):
        # For each row, find the smallest valid next element
        # Process rows in a random order to add diversity
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for r_idx in order:
            row = rows[r_idx]
            last = row[-1]
            x = last
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows


def _local_search_improved(rows, n_rows, k, best_max, deadline):
    """
    Local search: try to reduce the scope by adjusting individual elements.
    For each row, try shifting elements to reduce the maximum while maintaining validity.
    """
    current = [list(r) for r in rows]
    current_max = max(max(r) for r in current)
    rng = random.Random(123)
    
    # Compute all used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used = get_used_diffs(current)
    
    # Try multiple iterations of local search
    for iteration in range(500):
        if time.time() > deadline - 2:
            break
        
        # Pick a random row to try to optimize
        r_idx = rng.randint(0, n_rows - 1)
        row = current[r_idx]
        
        # Try to reduce the last element (which is the max of this row)
        # by trying to find a smaller valid value
        old_last = row[-1]
        if old_last <= 1:
            continue
        
        # Try values from old_last-1 down to some minimum
        # But we need to maintain strictly increasing order
        min_possible = row[-2] + 1 if len(row) > 1 else 1
        
        # Remove the last element's differences from used
        old_diffs = set()
        for a in row[:-1]:
            old_diffs.add(old_last - a)
        used -= old_diffs
        
        # Try smaller values
        found = False
        for new_val in range(old_last - 1, min_possible - 1, -1):
            new_diffs = {new_val - a for a in row[:-1]}
            if not new_diffs & used:
                row[-1] = new_val
                used |= new_diffs
                found = True
                break
        
        if not found:
            # Restore
            row[-1] = old_last
            used |= old_diffs
            continue
        
        # Check if overall max decreased
        new_max = max(max(r) for r in current)
        if new_max < current_max:
            current_max = new_max
    
    return current, current_max


def _greedy_with_lookahead(n_rows, k, seed=0):
    """
    Greedy with lookahead: when choosing the next element for a row,
    consider how it affects the ability to complete all rows.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for r_idx in range(n_rows):
        row = [0]
        x = 0
        remaining_rows = n_rows - r_idx - 1  # rows left to build after this one
        
        while len(row) < k + 1:
            # Find the smallest valid x
            best_x = None
            for trial in range(x + 1, x + 50):  # look ahead up to 50
                new_diffs = {trial - a for a in row}
                if not new_diffs & used:
                    # Check if this leaves enough room for remaining rows
                    # (heuristic: just check if it's valid)
                    best_x = trial
                    break
            
            if best_x is None:
                # Fallback: search further
                trial = x
                while True:
                    trial += 1
                    new_diffs = {trial - a for a in row}
                    if not new_diffs & used:
                        best_x = trial
                        break
            
            x = best_x
            row.append(x)
            used |= {x - a for a in row[:-1]}
        
        rows.append(row)
    
    return rows


def _genetic_search(n_rows, k, seed=0, deadline=None):
    """
    Genetic algorithm approach: represent each row as a set of differences,
    and evolve assignments to minimize the scope.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    rng = random.Random(seed)
    num_diffs = k * (k + 1) // 2
    
    # Start with a greedy solution
    population = [_greedy_fast(n_rows, k)]
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        best = min(population, key=lambda rows: max(max(r) for r in rows))
        best_max = max(max(r) for r in best)
        
        # Create mutants
        for _ in range(3):
            mutant = [list(r) for r in best]
            # Pick a random row and try to modify it
            r_idx = rng.randint(0, n_rows - 1)
            row = mutant[r_idx]
            
            # Try to shift some elements
            if len(row) > 1:
                # Try to reduce the last element
                old_last = row[-1]
                # Remove diffs
                used = set()
                for r in mutant:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            used.add(r[j] - r[i])
                
                old_diffs = {old_last - a for a in row[:-1]}
                used -= old_diffs
                
                # Try smaller values
                for new_val in range(old_last - 1, row[-2], -1):
                    new_diffs = {new_val - a for a in row[:-1]}
                    if not new_diffs & used:
                        row[-1] = new_val
                        used |= new_diffs
                        break
                else:
                    row[-1] = old_last
                    used |= old_diffs
            
            m = max(max(r) for r in mutant)
            if m < best_max:
                population = [mutant]
            else:
                population.append(mutant)
        
        # Keep only the best few
        population.sort(key=lambda rows: max(max(r) for r in rows))
        population = population[:5]
    
    return min(population, key=lambda rows: max(max(r) for r in rows))


def _backtrack_search(n_rows, k, max_scope, deadline):
    """
    Backtracking search to find a solution with scope <= max_scope.
    """
    used = set()
    rows = []
    
    def backtrack(row_idx, row, current_max):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if len(row) == k + 1:
            rows.append(row)
            return backtrack(row_idx + 1, [0], current_max)
        
        last = row[-1]
        for x in range(last + 1, max_scope + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                if backtrack(row_idx, row, max(current_max, x)):
                    return True
                used -= new_diffs
                row.pop()
        
        return False
    
    if backtrack(0, [0], 0):
        return rows
    return None
