import time
import random
import math

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin
    
    # Max possible sum is 256 * 256 = 65536
    MAX_SUM = 65536 + 1
    
    # Helper to check validity and get penalty
    # Returns (is_valid, penalty)
    # penalty is number of duplicate sums
    def check_validity(seq):
        if not seq:
            return True, 0
        
        # Compute all consecutive sums
        # seq is sorted list of ints
        n = len(seq)
        used_sums = set()
        duplicates = 0
        
        # Efficiently compute sums using prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
            
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in used_sums:
                    duplicates += 1
                else:
                    used_sums.add(s)
                    
        return duplicates == 0, duplicates

    # Greedy initialization to get a good starting point
    def greedy_init():
        values = []
        used_sums = bytearray(MAX_SUM)
        # Try to build a sequence
        # We can try multiple random orders or just standard 1..N
        # Standard greedy is deterministic and fast
        for x in range(1, N + 1):
            # Calculate new suffix sums
            # Suffix sums of new seq: x, x + last, x + last + second_last, ...
            s = x
            valid = True
            if used_sums[x]:
                valid = False
            else:
                for a in reversed(values):
                    s += a
                    if used_sums[s]:
                        valid = False
                        break
            
            if valid:
                values.append(x)
                # Update used_sums
                s = x
                used_sums[x] = 1
                for a in reversed(values[:-1]):
                    s += a
                    used_sums[s] = 1
        return values

    # Get initial solution
    current_seq = greedy_init()
    current_len = len(current_seq)
    current_penalty, _ = check_validity(current_seq)
    
    best_seq = current_seq[:]
    best_len = current_len
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    # Precompute some structure for faster updates? 
    # For simplicity and robustness, we recompute validity.
    # To speed up, we can maintain the set of sums and counts.
    
    # Let's maintain a dictionary of sum -> count
    # And the current sequence
    # Moves:
    # 1. Add a random x not in seq
    # 2. Remove a random x from seq
    # 3. Swap a random x in seq with a random y not in seq
    
    def get_sums_and_counts(seq):
        counts = {}
        n = len(seq)
        if n == 0:
            return counts
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                counts[s] = counts.get(s, 0) + 1
        return counts

    # Current state
    current_seq_set = set(current_seq)
    current_sums_counts = get_sums_and_counts(current_seq)
    current_penalty = sum(1 for c in current_sums_counts.values() if c > 1)
    # Actually, penalty should be sum(c-1 for c in counts.values() if c>1) to reflect total duplicates
    # Let's define penalty = number of pairs of identical sums? Or just count of duplicates?
    # Simpler: penalty = sum(c-1 for c in current_sums_counts.values() if c > 1)
    
    def calc_penalty(counts):
        return sum(c - 1 for c in counts.values() if c > 1)
    
    current_penalty = calc_penalty(current_sums_counts)
    
    def update_sums_counts(seq, counts, x, is_add):
        """
        Update sums counts when adding or removing x from a sorted seq.
        This is tricky because x's position changes all subsequent prefix sums.
        For simplicity in SA, we might just recompute if N is small enough or if moves are sparse.
        However, N=256, len~100-150. Recomputing is O(L^2).
        L=100 -> 10k ops. 1000 steps -> 10M ops. Should be fine in Python if optimized?
        Let's try to just recompute for simplicity and correctness first.
        """
        pass # Placeholder

    # Since efficient incremental update is complex, we will rely on the fact that
    # L is ~100-150, so L^2 is ~15k operations.
    # 55 seconds / 15k ops ~ 3600 steps. Might be too few.
    # Let's try to optimize the check.
    
    # Alternative: Use a bytearray for sums in the check?
    # The sums are up to 65536.
    # We can use a list of ints for counts, size 65537.
    # This allows O(1) increment/decrement if we knew which sums changed.
    # But again, adding x changes all sums that include x.
    # If we add x to the end, only new sums ending at x are added.
    # If we insert x in the middle, sums containing x change.
    
    # Strategy: Only allow "Add to End" and "Remove from End" and "Swap End with Non-End"?
    # No, that restricts the search space too much.
    
    # Let's stick to full recompute but optimize the inner loop.
    # Use numpy? No, standard library is safer for this specific small array unless we import it.
    # Let's use a list for counts.
    
    count_size = MAX_SUM
    current_counts = [0] * count_size
    
    def recompute_counts(seq):
        nonlocal current_counts
        # Reset
        for i in range(count_size):
            current_counts[i] = 0
            
        n = len(seq)
        if n == 0:
            return 0
            
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
            
        penalty = 0
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if current_counts[s] > 0:
                    penalty += 1
                current_counts[s] += 1
        return penalty

    current_penalty = recompute_counts(current_seq)
    
    # Main SA Loop
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate Move
        move_type = random.random()
        
        if move_type < 0.4 and current_len < N:
            # Add
            candidates = [x for x in range(1, N+1) if x not in current_seq_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            # Insert x into sorted list
            # Find position
            lo, hi = 0, len(current_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if current_seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            new_seq = current_seq[:lo] + [x] + current_seq[lo:]
            
            # Calculate New Penalty
            old_penalty = current_penalty
            new_penalty = recompute_counts(new_seq)
            
            delta_penalty = new_penalty - old_penalty
            delta_len = 1
            
            # Accept?
            if delta_penalty < 0 or (delta_penalty == 0 and delta_len > 0):
                # Better or equal penalty but longer
                current_seq = new_seq
                current_seq_set.add(x)
                current_len += 1
                current_penalty = new_penalty
                
                if current_len > best_len and current_penalty == 0:
                    best_len = current_len
                    best_seq = current_seq[:]
            elif T > 0:
                # Metropolis criterion
                # Energy = penalty
                # We want to minimize penalty, maximize length.
                # Let's define cost = penalty * 1000 - length
                # delta_cost = (new_penalty - old_penalty) * 1000 - 1
                delta_cost = delta_penalty * 1000 - 1
                if delta_cost < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta_cost / T)
                    
                if accept:
                    current_seq = new_seq
                    current_seq_set.add(x)
                    current_len += 1
                    current_penalty = new_penalty
                    
        elif move_type < 0.7 and current_len > 1:
            # Remove
            idx = random.randint(0, current_len - 1)
            x = current_seq[idx]
            new_seq = current_seq[:idx] + current_seq[idx+1:]
            
            old_penalty = current_penalty
            new_penalty = recompute_counts(new_seq)
            
            delta_penalty = new_penalty - old_penalty
            delta_len = -1
            
            if delta_penalty < 0:
                # Better
                current_seq = new_seq
                current_seq_set.remove(x)
                current_len -= 1
                current_penalty = new_penalty
            elif T > 0:
                delta_cost = delta_penalty * 1000 - (-1) # -1 is delta_len
                # Wait, cost = penalty * 1000 - length
                # delta_cost = delta_penalty * 1000 - delta_len
                delta_cost = delta_penalty * 1000 - (-1)
                if delta_cost < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta_cost / T)
                    
                if accept:
                    current_seq = new_seq
                    current_seq_set.remove(x)
                    current_len -= 1
                    current_penalty = new_penalty
                    
        else:
            # Swap
            if current_len < 1 or current_len >= N:
                continue
                
            # Pick one to remove
            idx = random.randint(0, current_len - 1)
            x = current_seq[idx]
            
            # Pick one to add
            candidates = [y for y in range(1, N+1) if y not in current_seq_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Remove x, add y
            # This is effectively remove then add, but we can do it in one step
            # Since we recompute anyway, just build new_seq
            temp_seq = current_seq[:idx] + current_seq[idx+1:]
            
            # Insert y into temp_seq
            lo, hi = 0, len(temp_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if temp_seq[mid] < y:
                    lo = mid + 1
                else:
                    hi = mid
            new_seq = temp_seq[:lo] + [y] + temp_seq[lo:]
            
            old_penalty = current_penalty
            new_penalty = recompute_counts(new_seq)
            
            delta_penalty = new_penalty - old_penalty
            delta_len = 0
            
            if delta_penalty < 0:
                current_seq = new_seq
                current_seq_set.remove(x)
                current_seq_set.add(y)
                current_penalty = new_penalty
            elif T > 0:
                delta_cost = delta_penalty * 1000 - 0
                if delta_cost < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta_cost / T)
                    
                if accept:
                    current_seq = new_seq
                    current_seq_set.remove(x)
                    current_seq_set.add(y)
                    current_penalty = new_penalty
        
        # Cool down
        T *= cooling_rate
        
        # If we have a valid solution, update best
        if current_penalty == 0 and current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

    # Final check: ensure best_seq is valid
    # It should be, but just in case
    is_valid, _ = check_validity(best_seq)
    if not is_valid:
        # Fallback to greedy
        return greedy_init()
        
    return best_seq
