"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    # Precompute max possible difference to use for bitset size if needed, 
    # but Python ints are arbitrary precision, so we just use bitwise ops.
    
    def get_diffs_bitset(row):
        """Calculate the bitset of positive differences for a row."""
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits

    def get_diffs_list(row):
        """Calculate the list of positive differences for a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(r) for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def energy(rows):
        """
        Objective function: primarily max entry, secondarily sum of entries.
        We encode this into a single score for SA.
        """
        m = max_entry(rows)
        s = sum_entries(rows)
        # We want to minimize m, then s.
        # Scale s so it doesn't dominate m unless m is equal.
        # Max s for m=112, n=7, k=5 is roughly 7 * 5 * 112 / 2 ~ 2000.
        # So if we add s/1000, it's small.
        return m * 1000 + s

    def greedy_row(used_bits, max_val, randomize=False, skip_prob=0.0):
        """
        Build a row greedily using bitset checks.
        """
        row = [0]
        local_bits = 0
        x = 0
        limit = max_val
        
        while len(row) < k + 1:
            found = False
            # Search for next element
            # We can randomize the start or the selection
            start_x = x + 1
            # If randomizing, we might jump ahead or skip
            candidates = []
            for candidate_x in range(start_x, limit + 1):
                # Check if candidate_x creates conflicts
                # New diffs are candidate_x - a for a in row
                new_diffs_bits = 0
                for a in row:
                    d = candidate_x - a
                    new_diffs_bits |= (1 << d)
                
                if (new_diffs_bits & used_bits) == 0 and (new_diffs_bits & local_bits) == 0:
                    candidates.append(candidate_x)
                    # If not randomizing, take the first one
                    if not randomize:
                        found = True
                        break
                        # If randomizing, we collect a few or all? 
                        # Collecting all is expensive if limit is high.
                        # Let's just pick randomly from valid ones in a small window
                        # or just the first one if not randomizing.
            
            if not found and not randomize:
                return None, None

            if randomize:
                if not candidates:
                    return None, None
                # Pick a random candidate, maybe biased towards smaller
                # To keep it simple, pick the first one if skip_prob is low, else random
                if random.random() < skip_prob and len(candidates) > 1:
                    # Skip the first few? Or pick random?
                    # Let's pick a random one from the first 10 valid candidates to keep local
                    pick_idx = random.randint(0, min(9, len(candidates)-1))
                    x = candidates[pick_idx]
                else:
                    x = candidates[0]
            else:
                x = candidates[0]
            
            row.append(x)
            # Update local bits
            for a in row[:-1]:
                d = x - a
                local_bits |= (1 << d)
            
            # Optimization: if we found a candidate in the loop, we break.
            # But the loop above only breaks if not randomize.
            # If randomize, it finishes the loop up to limit. This is slow.
            # Let's optimize the randomize case to stop early.
            
        return row, local_bits

    # Refactored Greedy for speed
    def greedy_row_fast(used_bits, max_val, randomize=False, skip_prob=0.0):
        row = [0]
        local_bits = 0
        current_max = 0
        
        while len(row) < k + 1:
            found = False
            # We scan from current_max + 1
            start = current_max + 1
            valid_candidates = []
            
            for x in range(start, max_val + 1):
                # Check conflicts
                conflict = False
                new_bits = 0
                for a in row:
                    d = x - a
                    if d == 0: continue # x > a always
                    bit = 1 << d
                    if (used_bits & bit) or (local_bits & bit):
                        conflict = True
                        break
                    new_bits |= bit
                
                if not conflict:
                    valid_candidates.append((x, new_bits))
                    if not randomize:
                        found = True
                        break
                    else:
                        # Collect a few valid candidates to allow randomization
                        if len(valid_candidates) >= 10:
                            break
            
            if not valid_candidates:
                return None, None
            
            if not randomize:
                x, new_bits = valid_candidates[0]
            else:
                # Pick randomly from valid candidates
                # Bias towards smaller x?
                # Uniform pick for now
                idx = random.randint(0, len(valid_candidates) - 1)
                x, new_bits = valid_candidates[idx]
            
            row.append(x)
            local_bits |= new_bits
            current_max = x
            
        return row, local_bits

    # 1. Initial Construction
    best_rows = None
    best_max = float('inf')
    best_energy = float('inf')
    
    # Try a few greedy strategies
    for strategy in range(5):
        used = 0
        rows = []
        valid = True
        # Strategy 0: Pure Greedy
        # Strategy 1+: Randomized Greedy
        skip_prob = 0.0 if strategy == 0 else 0.2 * strategy
        
        for i in range(n_rows):
            row, row_bits = greedy_row_fast(used, 200, randomize=(strategy > 0), skip_prob=skip_prob)
            if row is None:
                valid = False
                break
            rows.append(row)
            used |= row_bits
            
        if valid:
            cur_max = max_entry(rows)
            cur_energy = energy(rows)
            if cur_max < best_max or (cur_max == best_max and cur_energy < best_energy):
                best_max = cur_max
                best_energy = cur_energy
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback
        used = 0
        rows = []
        for i in range(n_rows):
            row, row_bits = greedy_row_fast(used, 1000)
            rows.append(row)
            used |= row_bits
        return rows

    # 2. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_bits_list = [get_diffs_bitset(r) for r in current_rows]
    used_bits = 0
    for b in current_bits_list:
        used_bits |= b
        
    current_energy = energy(current_rows)
    best_rows_sa = [r[:] for r in current_rows]
    best_energy_sa = current_energy
    
    start_time = time.time()
    time_limit = 110
    
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a move
        # 1. Pick a row to modify
        i = random.randint(0, n_rows - 1)
        
        # 2. Remove its bits
        used_bits_excl_i = used_bits & ~current_bits_list[i]
        
        # 3. Generate new row for i
        # Use randomization to explore
        new_row, new_bits = greedy_row_fast(used_bits_excl_i, current_max_entry(current_rows) + 5, randomize=True, skip_prob=0.3)
        
        if new_row is None:
            T *= cooling_rate
            continue
            
        # Calculate new energy
        # Temporarily swap
        old_row = current_rows[i]
        old_bits = current_bits_list[i]
        
        current_rows[i] = new_row
        current_bits_list[i] = new_bits
        used_bits = used_bits_excl_i | new_bits
        
        new_energy = energy(current_rows)
        delta_E = new_energy - current_energy
        
        # Acceptance criterion
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy_sa:
                best_energy_sa = new_energy
                best_rows_sa = [r[:] for r in current_rows]
        else:
            # Reject, revert
            current_rows[i] = old_row
            current_bits_list[i] = old_bits
            used_bits = used_bits_excl_i | old_bits
            
        T *= cooling_rate

    return best_rows_sa

def current_max_entry(rows):
    return max(max(r) for r in rows)

# EVOLVE-BLOCK-END
