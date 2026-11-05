import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    L = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    
    def get_diffs_from_gaps(gaps):
        """
        gaps: list of k integers
        returns: list of all differences x_j - x_i
        """
        diffs = []
        # x_0 = 0
        # x_j = sum(gaps[0..j-1])
        # diffs are x_j - x_i
        # It's easier to compute x first
        x = [0] * (k + 1)
        for i in range(1, k + 1):
            x[i] = x[i-1] + gaps[i-1]
        
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diffs.append(x[j] - x[i])
        return diffs

    def check_validity(gaps_list, max_scope):
        """
        gaps_list: list of n_rows lists of gaps
        Returns (is_valid, max_entry, collision_count, overflow_count)
        """
        all_diffs = []
        max_entry = 0
        for gaps in gaps_list:
            diffs = get_diffs_from_gaps(gaps)
            all_diffs.extend(diffs)
            # max entry is sum of gaps
            entry = sum(gaps)
            if entry > max_entry:
                max_entry = entry
        
        if max_entry > max_scope:
            # Count overflows
            overflow_count = 0
            # We need to check which diffs are > max_scope? 
            # Actually, the constraint is on the entries (x_k), not just diffs?
            # "largest entry as small as possible".
            # The differences are derived from entries.
            # If an entry > max_scope, it's invalid.
            # Also, differences must be distinct.
            # Do we care if a difference > max_scope? 
            # If x_k <= max_scope, then all diffs x_j - x_i <= x_k <= max_scope.
            # So if we ensure max_entry <= max_scope, we ensure all diffs <= max_scope.
            pass
            
        # Check distinctness
        # Using a set is O(N). 
        # For SA, we need fast collision counting.
        # Let's just use a set for now, it's O(N) per evaluation which might be slow for many steps.
        # N = n_rows * num_diffs_per_row.
        # For n=4, k=7, N=112. Very fast.
        
        unique_diffs = set(all_diffs)
        collision_count = len(all_diffs) - len(unique_diffs)
        
        is_valid = (collision_count == 0) and (max_entry <= max_scope)
        return is_valid, max_entry, collision_count

    def simulate_annealing(max_scope, time_limit):
        """
        Try to find a configuration with max entry <= max_scope and 0 collisions.
        """
        # Initialize gaps
        # Random gaps, biased towards small values to keep sums small
        # But we need to explore.
        # Let's initialize with random integers in [1, max_scope // k]
        max_gap_init = max(1, max_scope // k)
        
        gaps_list = []
        for _ in range(n_rows):
            # Start with 1s to ensure small sums, then randomize?
            # Or just random.
            gaps = [random.randint(1, max_gap_init) for _ in range(k)]
            gaps_list.append(gaps)
            
        # Calculate initial energy
        _, max_entry, collisions = check_validity(gaps_list, max_scope)
        # Energy = collisions + max(0, max_entry - max_scope) * 10
        # We want to minimize this.
        
        def energy(gaps_list):
            _, me, coll = check_validity(gaps_list, max_scope)
            overflow = max(0, me - max_scope)
            return coll + overflow * 10

        E = energy(gaps_list)
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        best_gaps = [g[:] for g in gaps_list]
        best_E = E
        
        while time.time() < time_limit:
            # Pick a move
            r = random.randrange(n_rows)
            j = random.randrange(k)
            old_val = gaps_list[r][j]
            
            # New value: random walk
            # Bias towards values that keep sum small?
            # Just random in a range
            # Range [1, max_scope] is too big.
            # Maybe [1, max_scope // 2]
            
            # Let's try a local change
            # Delta
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            if new_val < 1:
                new_val = 1
            # Cap new_val to avoid huge sums immediately
            # If sum of other gaps + new_val > max_scope * 2, it's bad.
            
            gaps_list[r][j] = new_val
            
            new_E = energy(gaps_list)
            
            # Acceptance
            if new_E < E or random.random() < math.exp(-(new_E - E) / T):
                E = new_E
                if E < best_E:
                    best_E = E
                    best_gaps = [g[:] for g in gaps_list]
                    if E == 0:
                        # Found valid solution
                        # Convert gaps to rows
                        rows = []
                        for g in best_gaps:
                            row = [0]
                            cur = 0
                            for gap in g:
                                cur += gap
                                row.append(cur)
                            rows.append(row)
                        return rows
            else:
                # Reject
                gaps_list[r][j] = old_val
            
            T = max(T_min, T * cooling)
            
        # Return best found so far (might not be valid)
        rows = []
        for g in best_gaps:
            row = [0]
            cur = 0
            for gap in g:
                cur += gap
                row.append(cur)
            rows.append(row)
        return rows

    # Search for smallest M
    # Lower bound
    M_lb = 112 # From prompt
    # Upper bound from greedy?
    # Let's run a quick greedy to get an upper bound
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    greedy_rows = pure_greedy()
    M_ub = max(max(r) for r in greedy_rows)
    
    # If greedy is already good, we can start from there?
    # But SA might find better.
    # Let's try M from M_lb to M_ub.
    
    # To save time, we can jump.
    # If M_lb fails, try M_lb + 5, etc.
    
    # Actually, let's just run SA for a fixed M for a while.
    # If it finds a solution, great.
    # If not, increase M.
    
    current_M = M_lb
    while current_M <= M_ub and time.time() < deadline - 10:
        # Allocate time for this M
        # If we are close to deadline, use remaining time
        time_alloc = min(10.0, deadline - 10 - time.time())
        if time_alloc <= 0: break
        
        rows = simulate_annealing(current_M, time.time() + time_alloc)
        
        # Check if rows are valid for current_M
        is_valid, max_entry, coll = check_validity(rows, current_M)
        if is_valid:
            return rows
        
        # If not valid, increase M
        # How much?
        # If we got close, increase by 1.
        # If far, increase more.
        # Let's just increase by 1 for now, SA is fast.
        current_M += 1
        
    # If we ran out of time or M, return the best greedy or last SA result
    # The last SA result might have lower max_entry than greedy?
    # We should keep the best rows found (lowest max_entry, valid or not? No, must be valid).
    # If no valid solution found, return greedy.
    
    # Wait, the SA might return invalid rows.
    # We should only return valid rows.
    # If SA didn't find a valid one, we fall back to greedy.
    
    return greedy_rows
