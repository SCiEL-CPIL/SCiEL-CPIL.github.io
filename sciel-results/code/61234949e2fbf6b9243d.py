"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Total number of differences that must be distinct
    # Each row has (k+1) choose 2 differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on max entry: must be at least total_diffs
    # Because differences are distinct integers in [1, M]
    lower_bound = total_diffs

    best_rows = None
    best_max = float('inf')

    def backtracking_search(seed, max_limit=None):
        """
        Attempt to construct a valid set using randomized backtracking.
        Returns (rows, max_entry) or (None, inf) if failed.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        if max_limit is None:
            max_limit = int(math.sqrt(total_diffs) * 2) + 20

        def build_row(target_len):
            """
            Build a single row of length target_len using backtracking.
            """
            row = [0]
            
            def backtrack(pos, current_max):
                if time.time() > deadline:
                    return False
                
                if pos == target_len:
                    return True
                
                # Determine the range of candidates for this position
                # Start from current_max + 1
                # We want to keep numbers small, so try smaller ones first
                # But add some randomness
                
                start = current_max + 1
                # Limit the search to avoid infinite loops
                # The maximum possible value is bounded by max_limit
                end = min(start + 50, max_limit + 1)
                
                # Generate candidates in a slightly randomized order
                # But bias towards smaller values
                candidates = list(range(start, end))
                # Shuffle the first few to add randomness, but keep the rest ordered
                if len(candidates) > 3:
                    rng.shuffle(candidates[:min(5, len(candidates))])
                    # Re-sort to maintain increasing order for the rest? 
                    # Actually, we just want to try a few small ones randomly, then larger ones.
                    # Let's just shuffle all and sort by a random key? No, that's expensive.
                    # Let's just try the first 5 randomly, then the rest in order.
                    # Or simpler: just shuffle the whole list but with a bias.
                    # For now, let's just shuffle the first 10.
                    shuffle_len = min(10, len(candidates))
                    first_part = candidates[:shuffle_len]
                    rng.shuffle(first_part)
                    candidates = first_part + candidates[shuffle_len:]
                
                for x in candidates:
                    # Check if x is valid
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Try this candidate
                        row.append(x)
                        used.update(new_diffs)
                        
                        if backtrack(pos + 1, x):
                            return True
                        
                        # Backtrack
                        row.pop()
                        used.difference_update(new_diffs)
                
                return False
            
            if backtrack(1, 0):
                return row
            return None

        # Main loop for rows
        for _ in range(n_rows):
            if time.time() > deadline:
                return None, float('inf')
            
            row = build_row(k + 1)
            if row is None:
                return None, float('inf')
            rows.append(row)
        
        max_entry = max(max(r) for r in rows)
        if max_entry > max_limit:
            return None, float('inf')
            
        return rows, max_entry

    def pure_greedy():
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

    # Run pure greedy first to have a baseline
    baseline_rows = pure_greedy()
    best_rows = baseline_rows
    best_max = max(max(r) for r in baseline_rows)

    # Now try randomized improvements with backtracking
    current_seed = 1
    
    while time.time() < deadline:
        seed = current_seed
        current_seed += 1
        
        # Use a tighter max_limit as we find better solutions
        # Start with the current best_max and try to improve
        max_limit = best_max - 1
        
        rows, max_val = backtracking_search(seed, max_limit)
        if rows is not None and max_val < best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows]
            
            # If we hit the theoretical lower bound, we can stop
            if best_max == lower_bound:
                break

    return best_rows
# EVOLVE-BLOCK-END
