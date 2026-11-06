import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit with safety margin for 90s kill
    
    def get_all_block_sums(seq):
        """Get all non-empty consecutive block sums of a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        sums = set()
        for i in range(len(seq)):
            total = 0
            for j in range(i, len(seq)):
                total += seq[j]
                if total in sums:
                    return False
                sums.add(total)
        return True

    def greedy_build(start_val=1):
        values = []
        sums = set()
        for x in range(start_val, 257):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def local_search_add_only(seq):
        """Try to insert elements anywhere in the sequence."""
        improved = True
        while improved:
            improved = False
            seq_set = set(seq)
            for x in range(1, 257):
                if x in seq_set:
                    continue
                for i in range(len(seq) + 1):
                    if i > 0 and seq[i-1] >= x:
                        continue
                    if i < len(seq) and seq[i] <= x:
                        continue
                    new_seq = seq[:i] + [x] + seq[i:]
                    if is_valid(new_seq):
                        seq = new_seq
                        seq_set.add(x)
                        improved = True
                        break
                if improved:
                    break
                if time.time() > deadline:
                    return seq
        return seq

    def randomized_local_search(initial_seq, time_budget):
        """
        Perform a random walk of Add/Remove/Swap moves.
        State is a sorted list of integers.
        """
        current_seq = list(initial_seq)
        best_seq = list(initial_seq)
        best_len = len(current_seq)
        
        all_vals = set(range(1, 257))
        
        def try_add(x):
            if x in current_seq_set:
                return False
            # Find insertion point
            i = 0
            while i < len(current_seq) and current_seq[i] < x:
                i += 1
            if i > 0 and current_seq[i-1] >= x:
                return False
            if i < len(current_seq) and current_seq[i] <= x:
                return False
            
            new_seq = current_seq[:i] + [x] + current_seq[i:]
            if is_valid(new_seq):
                current_seq[:] = new_seq
                current_seq_set.add(x)
                return True
            return False
        
        def try_remove(x):
            if x not in current_seq_set:
                return False
            for i, val in enumerate(current_seq):
                if val == x:
                    current_seq.pop(i)
                    current_seq_set.remove(x)
                    return True
            return False

        def try_swap(x_out, x_in):
            if x_out not in current_seq_set or x_in in current_seq_set:
                return False
            # Remove x_out
            idx_out = -1
            for i, val in enumerate(current_seq):
                if val == x_out:
                    idx_out = i
                    current_seq.pop(i)
                    current_seq_set.remove(x_out)
                    break
            
            # Add x_in
            j = 0
            while j < len(current_seq) and current_seq[j] < x_in:
                j += 1
            if j > 0 and current_seq[j-1] >= x_in:
                # Revert
                current_seq.insert(idx_out, x_out)
                current_seq_set.add(x_out)
                return False
            if j < len(current_seq) and current_seq[j] <= x_in:
                current_seq.insert(idx_out, x_out)
                current_seq_set.add(x_out)
                return False
            
            new_seq = current_seq[:j] + [x_in] + current_seq[j:]
            if is_valid(new_seq):
                current_seq[:] = new_seq
                current_seq_set.add(x_in)
                return True
            else:
                # Revert
                current_seq.insert(idx_out, x_out)
                current_seq_set.add(x_out)
                return False

        current_seq_set = set(current_seq)
        
        while time.time() < start_time + time_budget:
            r = random.random()
            
            if r < 0.6: # Try Add
                avail = list(all_vals - current_seq_set)
                if not avail:
                    break
                x = random.choice(avail)
                if try_add(x):
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
            elif r < 0.8: # Try Remove
                if len(current_seq) > best_len + 2:
                    x = random.choice(current_seq)
                    try_remove(x)
            else: # Try Swap
                if len(current_seq) > 0:
                    x_out = random.choice(current_seq)
                    avail = list(all_vals - current_seq_set)
                    if avail:
                        x_in = random.choice(avail)
                        try_swap(x_out, x_in)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
            
        return best_seq

    # Phase 1: Greedy to get a good initial solution
    best_seq = []
    max_len = 0
    
    for start_val in [1, 2, 3, 5, 10]:
        if time.time() > deadline - 10:
            break
        seq = greedy_build(start_val)
        seq = local_search_add_only(seq)
        if len(seq) > max_len:
            max_len = len(seq)
            best_seq = seq
    
    if not best_seq:
        return [1]

    # Phase 2: Randomized Local Search
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed
    
    num_restarts = 5
    time_per_restart = remaining_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 1:
            break
        
        perturbed = list(best_seq)
        if len(perturbed) > 2:
            idx = random.randint(1, len(perturbed) - 2)
            perturbed.pop(idx)
        
        result = randomized_local_search(perturbed, time_per_restart)
        if len(result) > max_len:
            max_len = len(result)
            best_seq = result
            
    return best_seq
