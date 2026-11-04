"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows.
    Minimize the largest entry."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Each row has k+1 elements, starting at 0
    # Number of differences per row: (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope from counting: need at least total_diffs distinct positive diffs
    # The smallest total_diffs positive diffs are 1..total_diffs
    # But they must be partitionable into valid difference sets
    count_bound = total_diffs  # m >= this
    
    def get_diffs(row):
        """Get all pairwise differences of a row (list of k+1 ints starting at 0)."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    def is_valid_config(config):
        """Check if config (list of n_rows rows) is valid."""
        used = set()
        for row in config:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = get_diffs(row)
            if len(set(d)) != len(d):
                return False  # duplicate diffs within row
            if used & set(d):
                return False
            used |= set(d)
        return True
    
    def scope_of(config):
        return max(max(r) for r in config)
    
    def compress_row(row, used_global, row_idx):
        """Try to compress a single row by pulling elements down.
        Returns (new_row, success)."""
        new_row = list(row)
        # Try to reduce each element from the end
        for pos in range(len(new_row) - 1, 0, -1):
            old_val = new_row[pos]
            # Find the smallest valid value for this position
            # Must be > new_row[pos-1] and all diffs must not conflict
            for candidate in range(new_row[pos - 1] + 1, old_val):
                # Check if changing new_row[pos] to candidate works
                # Remove old diffs involving pos
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - new_row[i])
                for i in range(pos + 1, len(new_row)):
                    old_diffs.add(new_row[i] - old_val)
                
                # Add new diffs
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(candidate - new_row[i])
                for i in range(pos + 1, len(new_row)):
                    new_diffs.add(new_row[i] - candidate)
                
                # Check: new_diffs must not conflict with (used_global - old_diffs)
                # Also new_diffs must not have internal duplicates
                if len(new_diffs) != len(new_diffs):
                    continue
                conflict = new_diffs & (used_global - old_diffs)
                if not conflict:
                    new_row[pos] = candidate
                    # Update used_global
                    used_global -= old_diffs
                    used_global |= new_diffs
                    break
        
        return new_row
    
    def compress_all(config, used):
        """Compress all rows iteratively."""
        for _ in range(3):  # Multiple passes
            changed = False
            for ridx in range(n_rows):
                # Compute used without this row
                row_diffs = set(get_diffs(config[ridx]))
                used_wo_row = used - row_diffs
                new_row = compress_row(config[ridx], used_wo_row, ridx)
                if new_row != config[ridx]:
                    used -= row_diffs
                    used |= set(get_diffs(new_row))
                    config[ridx] = new_row
                    changed = True
            if not changed:
                break
        return config
    
    def initial_greedy():
        """Greedy construction: build rows one by one, picking smallest valid values."""
        used = set()
        config = []
        for ridx in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                x = row[-1] + 1
                while True:
                    new_diffs = set(x - a for a in row)
                    if new_diffs & used:
                        x += 1
                        continue
                    # Also check internal consistency (should be fine since increasing)
                    row.append(x)
                    used |= new_diffs
                    break
            config.append(row)
        return config
    
    def initial_random(max_scope):
        """Random initial configuration within max_scope."""
        used = set()
        config = []
        for ridx in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                # Pick a random value that's valid
                found = False
                for _ in range(100):
                    x = random.randint(row[-1] + 1, max_scope)
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: find smallest valid
                    x = row[-1] + 1
                    while True:
                        new_diffs = set(x - a for a in row)
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                    break
            config.append(row)
        return config
    
    def simulated_annealing(config, used, max_iters, deadline):
        """Simulated annealing on the configuration."""
        current_config = [list(r) for r in config]
        current_used = set(used)
        current_scope = scope_of(current_config)
        best_config = [list(r) for r in current_config]
        best_scope = current_scope
        best_used = set(current_used)
        
        T = float(current_scope)  # Initial temperature
        T_min = 1.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and position to perturb
            ridx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Not the first element (which is 0)
            
            row = current_config[ridx]
            old_val = row[pos]
            
            # Compute old diffs involving this position
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            # Try a random new value
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos + 1 < len(row) else current_scope + 5
            
            if low >= high:
                continue
            
            new_val = random.randint(low, high)
            if new_val == old_val:
                continue
            
            # Compute new diffs
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check validity
            # The diffs that remain: current_used - old_diffs
            # We need new_diffs to be disjoint from (current_used - old_diffs)
            # And new_diffs must not have internal duplicates
            if len(new_diffs) != len(new_diffs):
                continue
            
            remaining = current_used - old_diffs
            if new_diffs & remaining:
                # This move is invalid (creates conflict)
                # But in simulated annealing, we can still accept with probability
                # based on how many conflicts there are
                n_conflicts = len(new_diffs & remaining)
                delta = n_conflicts * 10  # Penalty for conflicts
                if random.random() < pow(2.718281828, -delta / max(T, 0.01)):
                    # Accept even though invalid (will fix later)
                    row[pos] = new_val
                    current_used -= old_diffs
                    current_used |= new_diffs
            else:
                # Valid move
                row[pos] = new_val
                current_used -= old_diffs
                current_used |= new_diffs
                
                # Check if scope improved
                new_scope = scope_of(current_config)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_config = [list(r) for r in current_config]
                    best_used = set(current_used)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_config, best_used, best_scope
    
    # Phase 1: Greedy baseline
    config = initial_greedy()
    used = set()
    for r in config:
        used |= set(get_diffs(r))
    config = compress_all(config, used)
    # Recompute used after compression
    used = set()
    for r in config:
        used |= set(get_diffs(r))
    best_config = [list(r) for r in config]
    best_used = set(used)
    best_scope = scope_of(best_config)
    
    # Phase 2: Simulated annealing with multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        
        # Create a perturbed starting point
        if n_restarts == 1:
            start_config = [list(r) for r in best_config]
            start_used = set(best_used)
        else:
            # Start from best and perturb
            start_config = [list(r) for r in best_config]
            start_used = set(best_used)
            # Apply some random perturbations
            for _ in range(random.randint(3, 8)):
                ridx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                row = start_config[ridx]
                old_val = row[pos]
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - row[i])
                for i in range(pos + 1, len(row)):
                    old_diffs.add(row[i] - old_val)
                
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos + 1 < len(row) else scope_of(start_config) + 5
                if low < high:
                    new_val = random.randint(low, high)
                    if new_val != old_val:
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        # Only accept if valid
                        remaining = start_used - old_diffs
                        if not (new_diffs & remaining) and len(new_diffs) == len(new_diffs):
                            row[pos] = new_val
                            start_used -= old_diffs
                            start_used |= new_diffs
        
        # Run annealing
        iters = max(1000, int(50000 / max(1, n_restarts)))
        result_config, result_used, result_scope = simulated_annealing(
            start_config, start_used, iters, deadline - 5.0
        )
        
        if result_scope < best_scope:
            best_config = result_config
            best_used = result_used
            best_scope = result_scope
            
            # Compress after improvement
            best_config = compress_all(best_config, best_used)
            best_used = set()
            for r in best_config:
                best_used |= set(get_diffs(r))
            best_scope = scope_of(best_config)
        
        # If we've hit the counting bound, stop
        if best_scope <= count_bound:
            break
    
    # Final compression
    best_config = compress_all(best_config, best_used)
    best_used = set()
    for r in best_config:
        best_used |= set(get_diffs(r))
    best_scope = scope_of(best_config)
    
    # Verify validity
    if not is_valid_config(best_config):
        # Fallback to greedy
        best_config = initial_greedy()
    
    return best_config

# EVOLVE-BLOCK-END
