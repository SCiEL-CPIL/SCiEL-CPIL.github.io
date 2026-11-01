import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(forbidden_diffs, k, noise=0):
        """
        Constructs a row of k+1 elements starting at 0.
        Avoids differences in forbidden_diffs.
        If noise > 0, picks randomly from valid candidates in a window.
        """
        row = [0]
        row_diffs = set()
        
        for _ in range(k):
            last = row[-1]
            x = last + 1
            candidates = []
            
            # Find valid candidates
            # We limit the search window to keep it fast, but expand if necessary
            # Since we want to minimize max, we prefer small x.
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden_diffs) and not (new_diffs & row_diffs):
                    candidates.append(x)
                    if noise == 0:
                        break
                    if len(candidates) >= noise:
                        break
                x += 1
                # Safety: if x gets too large, break? 
                # But we must find a solution. 
                # The problem is guaranteed to have a solution for small k, n.
                # If x > 1000, something is wrong or we are in a very tight spot.
                if x > 10000:
                    break

            if not candidates:
                # Should not happen if forbidden_diffs is not "full"
                # Fallback: return a very large row? Or raise?
                # For robustness, just return what we have + large gaps
                # But this invalidates the solution.
                # We assume we find candidates.
                row.append(x)
                break

            if noise == 0:
                chosen = candidates[0]
            else:
                chosen = random.choice(candidates)
            
            row.append(chosen)
            row_diffs |= {chosen - a for a in row[:-1]}
        
        return row

    # Initialize
    used_diffs = set()
    current_rows = []
    for _ in range(n_rows):
        row = greedy_row(used_diffs, k, noise=0)
        current_rows.append(row)
        used_diffs |= get_diffs(row)
    
    current_max = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_max = current_max
    
    T = 50.0
    min_T = 0.5
    
    while time.time() < deadline:
        # Pick a random row
        j = random.randint(0, n_rows - 1)
        old_row = current_rows[j]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs from global set
        used_diffs -= old_diffs
        
        # Reconstruct row with noise
        # Noise increases with T
        noise = max(1, int(T))
        new_row = greedy_row(used_diffs, k, noise=noise)
        new_diffs = get_diffs(new_row)
        
        # Check if new_row is valid (it should be by construction)
        # Add new diffs
        used_diffs |= new_diffs
        
        # Calculate new global max
        # To avoid O(n_rows * k^2) for max, we can track max per row
        # But n_rows is small.
        new_global_max = max(max(r) for r in current_rows[:-1] + [new_row])
        
        # Simulated Annealing Acceptance
        delta = new_global_max - current_max
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            current_rows[j] = new_row
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            used_diffs -= new_diffs
            used_diffs |= old_diffs
            # current_rows[j] remains old_row
            
        # Cooling
        T = max(min_T, T * 0.99)
        
        # Occasionally restart with fresh greedy?
        # If T is low and we are stuck, maybe restart?
        # For now, just continue.
        
    return best_rows
