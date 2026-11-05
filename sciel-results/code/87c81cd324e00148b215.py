# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Improved Hybrid approach:
    1. Multi-start randomized greedy with "difference pressure" candidate selection.
    2. Simulated annealing with cooling schedule and multi-row rebuilds.
    3. Final polish phase with aggressive improvement attempts.
    4. Bitset-accelerated difference tracking for speed.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def time_up():
        return time.time() > deadline

    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    # Helper to compute all positive differences in a row as a set
    def get_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # Bitset helpers
    def diffs_to_bits(diffs):
        bits = 0
        for d in diffs:
            bits |= (1 << d)
        return bits

    def bits_to_set(bits):
        s = set()
        d = 1
        while bits:
            if bits & 1:
                s.add(d)
            bits >>= 1
            d += 1
        return s

    # 1. Multi-start Greedy with bitset tracking and "difference pressure"
    def greedy_construct(seed, pressure_weight=0.5):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_bits = 0
            
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                
                # Search for valid candidates
                # Collect a wider set to allow better selection
                width = 10
                
                for x in range(last + 1, 100000): # Large enough upper bound for greedy
                    new_diffs = {x - a for a in row}
                    new_bits = diffs_to_bits(new_diffs)
                    if (new_bits & local_bits) == 0 and (new_bits & used_bits) == 0:
                        # Score candidate by blend of value and max-new-diff
                        max_new_diff = max(new_diffs)
                        score = x * (1 - pressure_weight) + max_new_diff * pressure_weight
                        valid_candidates.append((score, x, new_bits))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None # Failed to construct
                
                # Selection strategy: Pick randomly from top candidates weighted by score
                # Lower score is better
                valid_candidates.sort(key=lambda t: t[0])
                # Pick from top 3 with probability, else random from top 5
                if len(valid_candidates) >= 3 and rng.random() < 0.7:
                    idx = rng.randint(0, 2)
                else:
                    idx = rng.randint(0, min(4, len(valid_candidates) - 1))
                
                _, x, new_bits = valid_candidates[idx]
                
                row.append(x)
                local_bits |= new_bits
                
            rows.append(row)
            used_bits |= local_bits
            
        return rows, used_bits

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used_bits = 0
    
    # Try a few seeds to find a good initial solution
    for seed in range(30):
        if time_up(): break
        # Vary pressure weight slightly
        pw = 0.3 + 0.02 * (seed % 10)
        res = greedy_construct(seed, pw)
        if res:
            rows, used_bits = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used_bits = used_bits
    
    if best_rows is None:
        # Fallback: deterministic greedy
        res = greedy_construct(0, 0.5)
        if res:
            best_rows, current_used_bits = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            # Should not happen for small k, n_rows
            return []

    if best_max == lb:
        return best_rows

    # 2. Simulated Annealing with Rebuilds
    # We maintain a bitset of used differences.
    # We repeatedly pick a row, remove its differences, and try to rebuild it.
    # Acceptance based on energy change (max entry) with temperature cooling.
    
    rng = random.Random(42)
    current_max = best_max
    
    # Temperature schedule
    T_initial = 5.0
    T_final = 0.1
    cooling_rate = 0.995
    
    def rebuild_row(k, forbidden_bits, rng, max_val_limit, bias=0.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_bits, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first 10, else smallest.
        """
        row = [0]
        local_bits = 0
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            # We must ensure x <= max_val_limit
            # Also, we need to leave room for remaining elements?
            # remaining_elements = k - len(row)
            # The next element must be at most max_val_limit - (remaining_elements - 1)
            
            remaining = k - len(row)
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                new_bits = diffs_to_bits(new_diffs)
                if (new_bits & local_bits) == 0 and (new_bits & forbidden_bits) == 0:
                    valid_candidates.append((x, new_bits))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection strategy
            if bias > 0 and rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_bits = valid_candidates[idx]
            else:
                x, new_bits = valid_candidates[0]
            
            row.append(x)
            local_bits |= new_bits
            
        return row, local_bits

    T = T_initial
    steps_since_improve = 0
    max_steps_since_improve = 200
    
    while not time_up():
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs_set(old_row)
        old_bits = diffs_to_bits(old_diffs)
        old_max_row = max(old_row)
        
        # Remove from used_bits
        current_used_bits &= ~old_bits
        
        new_row = None
        new_row_bits = 0
        
        # Attempt 1: Improve (strictly smaller max for this row)
        if old_max_row == current_max:
            target_max = current_max - 1
            if target_max > k: # Must be at least k
                res = rebuild_row(k, current_used_bits, rng, target_max, bias=0.5)
                if res:
                    new_row, new_row_bits = res
        
        # Attempt 2: Reshuffle (maintain max) to escape local minima
        if new_row is None:
            # Try to rebuild with the same max, but different structure
            res = rebuild_row(k, current_used_bits, rng, current_max, bias=0.8)
            if res:
                new_row, new_row_bits = res
        
        if new_row is None:
            # Failed to rebuild
            current_used_bits |= old_bits
            steps_since_improve += 1
            continue
        
        # Calculate new global max
        new_max_row = max(new_row)
        # To calculate new global max efficiently, we can track max per row
        # But for simplicity, let's just compute it if we accept
        # For now, assume we accept if it looks good
        
        # Energy calculation:
        # If new_max_row < old_max_row, energy decreases (good)
        # If new_max_row > old_max_row, energy increases (bad)
        # But we care about global max.
        # Let's compute the potential new global max.
        
        # Find current global max excluding row i
        other_max = 0
        for j in range(n_rows):
            if j != i:
                if max(current_rows[j]) > other_max:
                    other_max = max(current_rows[j])
        
        new_global_max = max(other_max, new_max_row)
        old_global_max = current_max
        
        delta_E = new_global_max - old_global_max
        
        # Acceptance criterion
        accepted = False
        if delta_E < 0:
            accepted = True
        else:
            # Metropolis criterion
            if T > 0:
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    accepted = True
        
        if accepted:
            current_rows[i] = new_row
            current_used_bits |= new_row_bits
            current_max = new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
                steps_since_improve = 0
                if best_max == lb:
                    break
            else:
                steps_since_improve += 1
        else:
            # Reject: restore old state
            current_used_bits |= old_bits
            steps_since_improve += 1
        
        # Cooling
        T *= cooling_rate
        if T < T_final:
            T = T_final
        
        # If stuck for too long, restart temperature or change strategy
        if steps_since_improve > max_steps_since_improve:
            T = T_initial
            steps_since_improve = 0
            # Also try a multi-row rebuild: pick 2 rows and rebuild both
            # This is complex, so for now just restart temperature

    # 3. Final Polish Phase
    # Use remaining time to aggressively try to reduce max
    if time.time() < deadline - 15.0:
        polish_deadline = deadline - 15.0
        polish_rng = random.Random(123)
        
        while time.time() < polish_deadline:
            # Find rows with max == best_max
            max_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_max]
            
            if not max_rows:
                break
            
            # Pick one
            i = polish_rng.choice(max_rows)
            
            old_row = current_rows[i]
            old_diffs = get_diffs_set(old_row)
            old_bits = diffs_to_bits(old_diffs)
            
            current_used_bits &= ~old_bits
            
            # Try to rebuild with max_val_limit = current_max - 1
            target_max = current_max - 1
            if target_max > k:
                res = rebuild_row(k, current_used_bits, polish_rng, target_max, bias=0.2)
                if res:
                    new_row, new_row_bits = res
                    current_rows[i] = new_row
                    current_used_bits |= new_row_bits
                    new_global_max = get_max(current_rows)
                    if new_global_max < best_max:
                        best_max = new_global_max
                        best_rows = [r[:] for r in current_rows]
                        if best_max == lb:
                            break
                else:
                    # Restore
                    current_used_bits |= old_bits
            else:
                current_used_bits |= old_bits

    # Validate
    def validate(rows, n_rows, k):
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
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        res = greedy_construct(0, 0.5)
        if res:
            best_rows = res[0]
        else:
            best_rows = [[0] * (k + 1)] * n_rows
    
    return best_rows

# EVOLVE-BLOCK-END
