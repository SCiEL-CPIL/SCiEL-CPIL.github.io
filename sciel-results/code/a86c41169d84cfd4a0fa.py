import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    def get_block_sums(seq):
        """Get all consecutive block sums of seq."""
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return sums
    
    def count_conflicts(sums):
        """Count the number of duplicate sums (each duplicate pair counts once)."""
        if not sums:
            return 0
        s = set()
        conflicts = 0
        for v in sums:
            if v in s:
                conflicts += 1
            s.add(v)
        return conflicts
    
    def is_valid(seq):
        """Check if seq is strictly increasing in [1,256] with distinct block sums."""
        if not seq:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
        sums = get_block_sums(seq)
        return len(sums) == len(set(sums))
    
    # Phase 1: Build a good initial sequence using greedy with lookahead
    def greedy_build():
        seq = []
        used_sums = set()
        for x in range(1, 257):
            # Check if adding x creates conflicts
            new_sums = [x]
            s = x
            for a in reversed(seq):
                s += a
                new_sums.append(s)
            conflict = False
            for ns in new_sums:
                if ns in used_sums:
                    conflict = True
                    break
            if not conflict:
                seq.append(x)
                used_sums.update(new_sums)
        return seq
    
    initial_seq = greedy_build()
    best_seq = initial_seq[:]
    best_conflicts = count_conflicts(get_block_sums(initial_seq))
    
    if best_conflicts == 0:
        best_len = len(best_seq)
    else:
        best_len = 0
    
    # Phase 2: Simulated annealing on the sequence
    # We'll work with sequences of various lengths, trying to reduce conflicts
    
    def random_seq_of_length(n):
        """Generate a random strictly increasing sequence of length n in [1,256]."""
        if n > 256:
            return None
        # Choose n distinct values from [1,256]
        vals = random.sample(range(1, 257), n)
        vals.sort()
        return vals
    
    # Start from the greedy sequence and try to extend it
    # Also try different starting points
    
    current_seq = initial_seq[:]
    current_sums = get_block_sums(current_seq)
    current_conflicts = count_conflicts(current_sums)
    
    # Try to extend the sequence
    def try_extend(seq):
        """Try to extend seq by adding elements at the end."""
        seq = seq[:]
        used_sums = set(get_block_sums(seq))
        extended = False
        for x in range(seq[-1] + 1, 257):
            new_sums = [x]
            s = x
            for a in reversed(seq):
                s += a
                new_sums.append(s)
            conflict = False
            for ns in new_sums:
                if ns in used_sums:
                    conflict = True
                    break
            if not conflict:
                seq.append(x)
                used_sums.update(new_sums)
                extended = True
        return seq
    
    current_seq = try_extend(current_seq)
    current_sums = get_block_sums(current_seq)
    current_conflicts = count_conflicts(current_sums)
    if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(current_seq) > best_len):
        best_seq = current_seq[:]
        best_conflicts = current_conflicts
        best_len = len(best_seq) if best_conflicts == 0 else 0
    
    # Simulated annealing
    temperature = 50.0
    cooling = 0.999
    min_temp = 0.01
    
    while time.time() - start_time < time_limit and temperature > min_temp:
        # Generate a perturbation
        new_seq = current_seq[:]
        
        # Random move: pick a random position and change the value
        n = len(new_seq)
        if n == 0:
            break
        
        move_type = random.random()
        
        if move_type < 0.3 and n < 256:
            # Try to insert a new element
            pos = random.randint(0, n)
            if pos == 0:
                lo, hi = 1, new_seq[0] - 1
            elif pos == n:
                lo, hi = new_seq[-1] + 1, 256
            else:
                lo, hi = new_seq[pos-1] + 1, new_seq[pos] - 1
            if lo <= hi:
                x = random.randint(lo, hi)
                new_seq.insert(pos, x)
        elif move_type < 0.6 and n > 1:
            # Try to remove an element
            pos = random.randint(0, n - 1)
            new_seq.pop(pos)
        elif move_type < 0.9 and n > 0:
            # Try to change a value
            pos = random.randint(0, n - 1)
            if pos == 0:
                lo, hi = 1, (new_seq[1] - 1 if n > 1 else 256)
            elif pos == n - 1:
                lo, hi = (new_seq[pos-1] + 1 if pos > 0 else 1), 256
            else:
                lo, hi = new_seq[pos-1] + 1, new_seq[pos+1] - 1
            if lo <= hi:
                x = random.randint(lo, hi)
                if x != new_seq[pos]:
                    new_seq[pos] = x
            # Ensure strictly increasing
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i+1]:
                    # Fix by adjusting
                    if i == 0:
                        new_seq[0] = max(1, new_seq[1] - 1)
                    else:
                        new_seq[i] = new_seq[i-1] + 1
                    if new_seq[i] >= new_seq[i+1]:
                        # Invalid, revert
                        new_seq = current_seq[:]
        else:
            # Try to swap two adjacent elements (but must maintain increasing order, so skip)
            # Instead, try a small random perturbation of a random element
            if n > 0:
                pos = random.randint(0, n - 1)
                delta = random.choice([-2, -1, 1, 2])
                new_val = new_seq[pos] + delta
                if new_val < 1:
                    new_val = 1
                if new_val > 256:
                    new_val = 256
                # Check bounds
                if pos > 0 and new_val <= new_seq[pos-1]:
                    new_val = new_seq[pos-1] + 1
                if pos < n - 1 and new_val >= new_seq[pos+1]:
                    new_val = new_seq[pos+1] - 1
                if 1 <= new_val <= 256:
                    if pos > 0 and new_val <= new_seq[pos-1]:
                        new_seq = current_seq[:]
                    elif pos < n - 1 and new_val >= new_seq[pos+1]:
                        new_seq = current_seq[:]
                    else:
                        new_seq[pos] = new_val
        
        # Compute new conflicts
        new_sums = get_block_sums(new_seq)
        new_conflicts = count_conflicts(new_sums)
        
        # Acceptance criterion
        delta_conflicts = new_conflicts - current_conflicts
        delta_len = len(new_seq) - len(current_seq)
        
        # We prefer fewer conflicts, and for equal conflicts, prefer longer sequences
        if new_conflicts < current_conflicts:
            accept = True
        elif new_conflicts == current_conflicts and len(new_seq) > len(current_seq):
            accept = True
        elif new_conflicts == current_conflicts and len(new_seq) == len(current_seq):
            accept = True
        else:
            # Accept with probability exp(-delta_effective / temperature)
            # delta_effective combines conflicts and length
            if new_conflicts > current_conflicts:
                effective_delta = new_conflicts - current_conflicts
            else:
                effective_delta = 0
            if len(new_seq) < len(current_seq):
                effective_delta += (len(current_seq) - len(new_seq)) * 0.5
            
            if effective_delta <= 0:
                accept = True
            else:
                import math
                try:
                    prob = math.exp(-effective_delta / temperature)
                except OverflowError:
                    prob = 0.0
                accept = random.random() < prob
        
        if accept:
            current_seq = new_seq
            current_sums = new_sums
            current_conflicts = new_conflicts
        
        # Update best
        if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(current_seq) > best_len):
            best_seq = current_seq[:]
            best_conflicts = current_conflicts
            if best_conflicts == 0:
                best_len = len(best_seq)
        
        temperature *= cooling
    
    # Phase 3: If we have a valid sequence, try to extend it greedily
    if best_conflicts == 0:
        final_seq = try_extend(best_seq)
        if len(final_seq) > best_len:
            best_seq = final_seq
            best_len = len(best_seq)
    
    # Phase 4: Try multiple random restarts if time remains
    if time.time() - start_time < time_limit - 5:
        for _ in range(10):
            if time.time() - start_time > time_limit - 2:
                break
            # Try a different initial sequence
            # Use a random subset approach
            n_try = random.randint(5, min(30, 256))
            rand_seq = random_seq_of_length(n_try)
            if rand_seq is None:
                continue
            rand_seq = try_extend(rand_seq)
            rc = count_conflicts(get_block_sums(rand_seq))
            if rc < best_conflicts or (rc == best_conflicts and len(rand_seq) > best_len):
                best_seq = rand_seq[:]
                best_conflicts = rc
                if rc == 0:
                    best_len = len(best_seq)
    
    # Final check
    if best_conflicts != 0:
        # Return the greedy sequence as fallback
        best_seq = greedy_build()
    
    return best_seq
