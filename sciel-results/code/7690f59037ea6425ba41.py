import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate an initial valid solution using a simple greedy algorithm.
    # This ensures we always have a valid starting point.
    # Greedy: for each row, pick the smallest possible next integer that doesn't 
    # conflict with existing differences.
    
    def generate_greedy(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = generate_greedy(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing to minimize scope.
    # State: list of rows.
    # Move: Pick a random row r and a random index i (1 <= i <= k), 
    #        change rows[r][i] by delta (delta is small, e.g., -1, 0, +1, or random jump).
    #        Actually, changing one element changes all differences involving that element.
    #        We need to check validity of the whole set.
        
    # To speed up validity check, we maintain a set of all differences.
    # When we change an element, we remove old diffs involving it, add new ones, 
    # and check for conflicts.
    
    # However, maintaining the global set dynamically with backtracking in SA is tricky 
    # because a "move" might make the solution invalid temporarily.
    # Standard SA accepts invalid states with a penalty.
    
    # Let's define the cost function:
    # Cost = Scope + Penalty * (Number of Duplicate Differences)
    # If Penalty is large enough, the optimizer will try to eliminate duplicates first, 
    # then minimize scope.
    
    # Since we start with a valid solution (0 duplicates), we can use a simpler cost:
    # Cost = Scope + 1000 * (Max Duplicate Count)
    # But actually, we want to stay valid if possible.
    # Let's use a penalty for any duplicate difference.
    
    def calculate_cost(rows):
        # Returns (scope, num_duplicates)
        # We need to count how many times each difference appears.
        # If any difference appears > 1, it's a duplicate.
        # For efficiency in the inner loop, we might just check validity.
        # But SA needs a smooth gradient.
        
        # Let's stick to a hard constraint approach first: 
        # Only accept moves that result in a valid solution.
        # This is "Local Search with Hard Constraints".
        # If stuck, we might need to allow invalid states (Soft Constraints).
        
        # Given the small size (k=7, n_rows=4), checking validity is O(n_rows * k^2).
        # 4 * 49 = 196 operations. Very fast.
        
        # Let's try Hard Constraint Local Search first.
        # Move: Shift one element up or down.
        # If the new state is valid, calculate new scope.
        # Accept if new_scope < current_scope.
        # To escape local minima, we need to allow moves that increase scope slightly?
        # Or we can use "kick" moves: randomize a few elements, then greedily fix?
        
        # Let's implement a randomized hill-climbing with restarts.
        # 1. Start with greedy.
        # 2. Loop:
        #    - Try to improve by shifting random elements.
        #    - If no improvement after X steps, do a "kick" (randomize one row slightly) 
        #      and re-optimize.
        
        pass

    # Actually, let's implement a proper SA with a penalty for duplicates.
    # This allows navigating through invalid states to find better valid states.
    
    def get_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.append(r[j] - r[i])
        return diffs

    def evaluate(rows):
        # Returns (scope, penalty)
        # Penalty is high if there are duplicates.
        diffs = get_diffs(rows)
        scope = max(max(r) for r in rows)
        
        # Count duplicates
        # Use a set to track seen diffs.
        seen = set()
        dup_count = 0
        for d in diffs:
            if d in seen:
                dup_count += 1
            else:
                seen.add(d)
        
        # Penalty: We want to minimize duplicates. 
        # If dup_count > 0, we are invalid.
        # Cost = scope + 1000 * dup_count
        return scope, dup_count

    # Optimization Loop
    current_rows = [r[:] for r in best_rows]
    current_scope, current_dups = evaluate(current_rows)
    
    # If greedy is already optimal (unlikely), we are done.
    # But we assume we can do better.
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    
    # We will run until time limit or T < T_min
    # To make it faster, we can group moves.
    
    # Precompute indices for moves
    # A move is defined by (row_idx, elem_idx, delta)
    # elem_idx is from 1 to k (index 0 is fixed at 0)
    
    steps = 0
    while time.time() < deadline:
        # Random move
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, k)  # 1 to k inclusive
        
        # Choose a delta
        # Small deltas are good for refinement. Large deltas for exploration.
        if random.random() < 0.8:
            delta = random.choice([-1, 1])
        else:
            delta = random.randint(-5, 5)
        
        old_val = current_rows[r_idx][e_idx]
        new_val = old_val + delta
        
        # Basic bounds check:
        # Must be > previous element
        # Must be < next element
        # Must be > 0 (since index 0 is 0 and strictly increasing)
        
        if e_idx > 1:
            if new_val <= current_rows[r_idx][e_idx - 1]:
                continue
        if e_idx < k:
            if new_val >= current_rows[r_idx][e_idx + 1]:
                continue
        if new_val <= 0:
            continue
            
        # Apply move
        current_rows[r_idx][e_idx] = new_val
        
        # Evaluate
        new_scope, new_dups = evaluate(current_rows)
        
        # Cost function
        # We prioritize validity. 
        # If current is valid (dups=0) and new is invalid (dups>0), 
        # we should only accept if T is high enough to allow this "jump".
        # If current is invalid and new is valid, always accept.
        
        # Let's define Cost = scope + 100 * dups
        old_cost = current_scope + 100 * current_dups
        new_cost = new_scope + 100 * new_dups
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0:
            # Improvement
            current_scope = new_scope
            current_dups = new_dups
            # Update best if valid
            if new_dups == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Accept with probability exp(-delta_cost / T)
            if random.random() < math.exp(-delta_cost / T):
                current_scope = new_scope
                current_dups = new_dups
                # Update best if valid
                if new_dups == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject move
                current_rows[r_idx][e_idx] = old_val
        
        # Cool down
        T *= alpha
        
        steps += 1
        # If we hit a low T and are stuck, reheat?
        # For simplicity, just let it run. The initial T should be high enough.
        
        # Optimization: If we are valid and have been valid for a long time, 
        # we can increase the penalty weight to force stricter validity?
        # Or just keep it simple.
        
        # Check if we can break early if T is very low and no improvements in last N steps?
        # Let's just run until deadline.
        
    # Final check: ensure best_rows is valid
    # The SA might end in an invalid state if it didn't find a valid one with lower scope.
    # But we always keep best_rows which was valid when updated.
    # The initial greedy solution is valid. So best_rows is always valid.
    
    return best_rows
