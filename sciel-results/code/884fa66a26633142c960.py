"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Use simulated annealing on the set of selected indices to find a strictly
    increasing sequence of integers in [1,256] such that all nonempty consecutive
    block sums are distinct. Maximize sequence length.
    """
    
    def compute_consecutive_sums(seq):
        """Compute all nonempty consecutive block sums for a strictly increasing sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def is_valid(seq):
        """Check if sequence is strictly increasing and all consecutive block sums are distinct."""
        if len(seq) == 0:
            return True
        # Check strictly increasing
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        # Check all values in [1, 256]
        for v in seq:
            if v < 1 or v > 256:
                return False
        # Check consecutive block sums are distinct
        sums = compute_consecutive_sums(seq)
        n = len(seq)
        expected = n * (n + 1) // 2
        if len(sums) != expected:
            return False
        return True
    
    def count_consecutive_block_sums(seq):
        """Return the number of distinct consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return len(sums)
    
    # Start with a random valid subset
    def random_valid_subset(min_len=5, max_len=15):
        """Generate a random strictly increasing sequence with valid consecutive block sums."""
        attempts = 0
        while attempts < 1000:
            length = random.randint(min_len, max_len)
            # Pick random distinct values from [1, 256]
            candidates = random.sample(range(1, 257), length)
            candidates.sort()
            if is_valid(candidates):
                return candidates
            attempts += 1
        # Fallback: use the greedy approach to get a valid starting point
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        # Take a random subset of the greedy solution
        if len(values) > 10:
            length = random.randint(8, 12)
            idx = random.sample(range(len(values)), length)
            idx.sort()
            return [values[i] for i in idx]
        return values
    
    # Simulated annealing
    def simulated_annealing(initial_seq, max_iters=50000, start_temp=100.0, end_temp=0.1, cooling=0.9998):
        """Simulated annealing to maximize sequence length while maintaining validity."""
        current = list(initial_seq)
        current_len = len(current)
        best = list(current)
        best_len = current_len
        
        temp = start_temp
        
        # Precompute all possible values
        all_values = list(range(1, 257))
        
        for iteration in range(max_iters):
            temp = max(end_temp, temp * cooling)
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add an element
                # Pick a random position to insert
                if len(current) == 0:
                    new_val = random.choice(all_values)
                    new_seq = [new_val]
                else:
                    # Try to insert a new value at a random position
                    pos = random.randint(0, len(current))
                    # Find a candidate value that fits
                    low = current[pos-1] + 1 if pos > 0 else 1
                    high = current[pos] - 1 if pos < len(current) else 256
                    if low <= high:
                        new_val = random.randint(low, high)
                        new_seq = current[:pos] + [new_val] + current[pos:]
                    else:
                        continue
            elif move_type < 0.7:
                # Try to remove an element
                if len(current) <= 1:
                    continue
                pos = random.randint(0, len(current) - 1)
                new_seq = current[:pos] + current[pos+1:]
            elif move_type < 0.85:
                # Try to swap an element with a different value
                if len(current) == 0:
                    continue
                pos = random.randint(0, len(current) - 1)
                low = current[pos-1] + 1 if pos > 0 else 1
                high = current[pos+1] - 1 if pos < len(current) - 1 else 256
                if low < high:
                    new_val = random.randint(low, high)
                    if new_val != current[pos]:
                        new_seq = current[:pos] + [new_val] + current[pos+1:]
                    else:
                        continue
                else:
                    continue
            else:
                # Try to move an element to a different position
                if len(current) <= 1:
                    continue
                pos1 = random.randint(0, len(current) - 1)
                pos2 = random.randint(0, len(current) - 1)
                if pos1 == pos2:
                    continue
                val = current[pos1]
                new_seq = current[:]
                new_seq.pop(pos1)
                # Insert at pos2 (adjust for removal)
                if pos1 < pos2:
                    insert_pos = pos2 - 1
                else:
                    insert_pos = pos2
                # Check if it fits
                if insert_pos > 0 and val <= new_seq[insert_pos-1]:
                    continue
                if insert_pos < len(new_seq) and val >= new_seq[insert_pos]:
                    continue
                new_seq.insert(insert_pos, val)
            
            # Check if new_seq is valid
            if not is_valid(new_seq):
                continue
            
            new_len = len(new_seq)
            delta_len = new_len - len(current)
            
            # Accept if better, or with some probability if worse
            if delta_len >= 0:
                current = new_seq
                if new_len > best_len:
                    best = new_seq
                    best_len = new_len
            else:
                # Accept worse with probability exp(delta_len / temp)
                prob = np.exp(delta_len / temp)
                if random.random() < prob:
                    current = new_seq
        
        return best
    
    # Run simulated annealing
    best_overall = []
    best_overall_len = 0
    
    # Run multiple restarts
    random.seed(42)
    for restart in range(5):
        initial = random_valid_subset(min_len=5, max_len=12)
        if len(initial) == 0:
            continue
        
        result = simulated_annealing(initial, max_iters=30000, start_temp=50.0, end_temp=0.05, cooling=0.9999)
        
        if len(result) > best_overall_len:
            best_overall = result
            best_overall_len = len(result)
    
    # Also try the greedy solution as a baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    if len(values) > best_overall_len:
        best_overall = values
        best_overall_len = len(values)
    
    return best_overall
# EVOLVE-BLOCK-END
