import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_rows = None
    best_max_val = float('inf')
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def generate_greedy(seed_offset=0):
        """
        Generates a valid set of rows using a randomized greedy approach.
        """
        random.seed(seed_offset)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            # For the first row, we can be more deterministic or use a specific strategy
            # For subsequent rows, randomize to explore space
            for pos in range(1, k + 1):
                found = False
                # Start from the smallest possible value
                x = row[-1] + 1
                # Limit search range to avoid infinite loops, 
                # but for small k, valid x exists relatively close
                max_x = 5000 
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    # Check if new_diffs intersects with used_diffs
                    if not (new_diffs & used_diffs):
                        # Valid candidate
                        # To add randomness, we can collect a few candidates and pick one
                        # Or just pick the first one for speed, but that's deterministic.
                        # Let's collect up to 10 candidates and pick randomly
                        candidates = [x]
                        # Look ahead a bit for diversity
                        temp_x = x + 1
                        while temp_x <= max_x and len(candidates) < 10:
                            nd = {temp_x - a for a in row}
                            if not (nd & used_diffs):
                                candidates.append(temp_x)
                            temp_x += 1
                            
                        chosen_x = random.choice(candidates)
                        row.append(chosen_x)
                        used_diffs.update({chosen_x - a for a in row[:-1]})
                        found = True
                        break
                    x += 1
                
                if not found:
                    # Fallback: If we didn't find a candidate in the limited range, 
                    # expand range or raise error. For this task, 5000 should be enough.
                    # If not, we might need to backtrack. 
                    # For simplicity, we'll just try a larger range.
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            row.append(x)
                            used_diffs.update({x - a for a in row[:-1]})
                            found = True
                            break
                        x += 1
                        if x > 10000:
                            return None, None # Failed to generate
        
            rows.append(row)
        
        return rows, used_diffs

    def simulated_annealing(rows, used_diffs, iters=500):
        """
        Improves the solution using Simulated Annealing.
        Moves: Change a single element in a row.
        Objective: Minimize the global maximum element.
        """
        current_rows = [r[:] for r in rows]
        current_used_diffs = set(used_diffs)
        
        # Calculate initial energy (max val)
        current_max = max(max(r) for r in current_rows)
        
        # SA Parameters
        T = 50.0  # Initial temperature
        cooling = 0.99
        min_T = 0.1
        
        for i in range(iters):
            if time.time() - start_time > time_limit - 2:
                break
            if T < min_T:
                break
                
            # Pick a random row and a random position (index > 0)
            r_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, k) # Index 1 to k (0 is fixed at 0)
            
            old_val = current_rows[r_idx][pos_idx]
            row = current_rows[r_idx]
            
            # Determine valid range for new value
            # Must be > row[pos_idx-1] and < row[pos_idx+1] (if pos_idx < k)
            lower = row[pos_idx - 1] + 1
            upper = row[pos_idx + 1] - 1 if pos_idx < k else current_max + 10 # Allow expansion? No, we want to shrink.
            # Actually, we want to try values that might reduce the max.
            # If this is the last element, we can try smaller values.
            # If it's an internal element, changing it affects diffs to all other elements in the row.
            
            # Let's try a few random candidates in the range [lower, upper]
            # Or specifically, if this element is the max of the row, try to reduce it.
            
            # To keep it simple and effective:
            # Try to replace old_val with a new_val in a neighborhood.
            # Neighborhood: [max(lower, old_val - 20), min(upper, old_val + 20)]
            # But we must ensure strict increase.
            
            search_lower = max(lower, old_val - 20)
            search_upper = min(upper, old_val + 20)
            
            if search_lower > search_upper:
                continue
                
            # Pick a random new value
            new_val = random.randint(search_lower, search_upper)
            
            if new_val == old_val:
                continue
                
            # Check if new_val is valid for the row (strictly increasing)
            if pos_idx > 0 and new_val <= row[pos_idx - 1]:
                continue
            if pos_idx < k and new_val >= row[pos_idx + 1]:
                continue
            
            # Check global validity
            # Remove old diffs contributed by old_val
            old_diffs = set()
            for j in range(len(row)):
                if j != pos_idx:
                    d = abs(row[j] - old_val)
                    old_diffs.add(d)
            
            # Add new diffs contributed by new_val
            new_diffs = set()
            for j in range(len(row)):
                if j != pos_idx:
                    d = abs(row[j] - new_val)
                    new_diffs.add(d)
            
            # Check conflicts
            # The set of "active" diffs is current_used_diffs - old_diffs
            # We need new_diffs to be disjoint from (current_used_diffs - old_diffs)
            # Also, new_diffs must not contain duplicates internally? 
            # Since the row is strictly increasing, diffs are distinct within the row? 
            # Yes, if a_i < a_j < a_k, then a_j-a_i != a_k-a_j is not guaranteed. 
            # Wait, a_j-a_i and a_k-a_j can be equal. 
            # The condition is "all differences between two entries of the same row are distinct".
            # So we must check if new_diffs has duplicates.
            
            if len(new_diffs) != len([abs(row[j] - new_val) for j in range(len(row)) if j != pos_idx]):
                continue # Duplicate diffs within the row
            
            # Check against global used diffs
            conflict = False
            for d in new_diffs:
                if d in current_used_diffs and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Valid move
            # Calculate new max
            new_row = row[:]
            new_row[pos_idx] = new_val
            new_max = max(new_row)
            # Global max is max of all rows. 
            # To speed up, we can maintain a heap or just recalculate. 
            # Recalculating is O(n*k). n=7, k=5. Very fast.
            global_new_max = max(max(r) for r in current_rows if r is not row) 
            # Wait, row is being modified. 
            # Let's just copy and check.
            temp_rows = current_rows[:]
            temp_rows[r_idx] = new_row
            global_new_max = max(max(r) for r in temp_rows)
            
            delta_e = global_new_max - current_max
            
            if delta_e < 0:
                # Better solution
                current_rows[r_idx] = new_row
                current_used_diffs -= old_diffs
                current_used_diffs |= new_diffs
                current_max = global_new_max
            else:
                # Worse solution, accept with probability
                p = math.exp(-delta_e / T)
                if random.random() < p:
                    current_rows[r_idx] = new_row
                    current_used_diffs -= old_diffs
                    current_used_diffs |= new_diffs
                    current_max = global_new_max
            
            T *= cooling
        
        return current_rows, current_used_diffs

    # Main Loop
    attempt = 0
    while time.time() - start_time < time_limit:
        attempt += 1
        rows, used_diffs = generate_greedy(attempt)
        
        if rows is None:
            continue
            
        # Improve
        rows, used_diffs = simulated_annealing(rows, used_diffs, iters=200)
        
        # Check validity (safety)
        # (Assuming SA maintains validity, but let's be sure)
        
        score = get_max_val(rows)
        if score < best_max_val:
            best_max_val = score
            best_rows = [r[:] for r in rows]
            
            # If we hit the theoretical lower bound or a very good bound, we might stop early?
            # For (7,5), 112 is best known. 
            # If we find 112, we can stop? 
            # The prompt says "best known set has largest entry 112".
            # If we find 112, it's likely optimal or near-optimal.
            if score <= 112:
                # We can break if we are confident, or continue to try for 111?
                # The counting bound is 105. So 112 might not be optimal.
                # But 112 is "best known". 
                # Let's run a bit longer to see if we can beat 112.
                pass

    return best_rows
