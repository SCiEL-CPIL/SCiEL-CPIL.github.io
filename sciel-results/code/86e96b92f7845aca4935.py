"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Greedy construction of a valid sequence
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    # If we already used all 256 values, return immediately
    if len(values) == 256:
        return values

    best_sequence = list(values)
    best_len = len(best_sequence)

    # Try to improve by finding better starting points or using local search
    # Strategy: try different greedy orders and local search

    def compute_block_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return sums

    def count_duplicates(sums_list):
        """Count number of duplicate sums (collisions)."""
        return len(sums_list) - len(set(sums_list))

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block sums are distinct."""
        if not seq:
            return True
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        sums = compute_block_sums(seq)
        return len(sums) == len(set(sums))

    # Local search: perturb the sequence and try to reduce duplicates
    def local_search(seq, max_iters=5000):
        """Local search to find a longer valid sequence."""
        current = list(seq)
        current_sums = set(compute_block_sums(current))
        current_dupes = count_duplicates(list(current_sums))
        
        best_local = list(current)
        best_local_dupes = current_dupes
        best_local_len = len(current)
        
        temp = 10.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random perturbation: pick a random position and try to insert/remove/modify
            move_type = random.random()
            
            if move_type < 0.4 and len(current) < 256:
                # Try to insert a new value at a random position
                pos = random.randint(0, len(current))
                # Try a few candidate values
                candidates = []
                for v in range(1, 257):
                    if v not in current:
                        candidates.append(v)
                if candidates:
                    v = random.choice(candidates)
                    new_seq = current[:pos] + [v] + current[pos:]
                    # Check if strictly increasing
                    valid_incr = all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1))
                    if valid_incr:
                        new_sums = compute_block_sums(new_seq)
                        new_dupes = count_duplicates(new_sums)
                        delta = new_dupes - current_dupes
                        if delta < 0 or (delta == 0 and random.random() < 0.1) or (delta > 0 and random.random() < 0.01 * temp / (temp + 1)):
                            current = new_seq
                            current_sums = set(new_sums)
                            current_dupes = new_dupes
                            if current_dupes == 0 and len(current) > best_local_len:
                                best_local = list(current)
                                best_local_dupes = 0
                                best_local_len = len(current)
            elif move_type < 0.7 and len(current) > 1:
                # Try to remove a value
                pos = random.randint(0, len(current) - 1)
                new_seq = current[:pos] + current[pos+1:]
                new_sums = compute_block_sums(new_seq)
                new_dupes = count_duplicates(new_sums)
                delta = new_dupes - current_dupes
                if delta <= 0 or random.random() < 0.05:
                    current = new_seq
                    current_sums = set(new_sums)
                    current_dupes = new_dupes
            else:
                # Try to swap two adjacent values (if it maintains increasing order)
                if len(current) > 1:
                    pos = random.randint(0, len(current) - 2)
                    if current[pos] + 1 < current[pos+1]:
                        new_seq = list(current)
                        new_seq[pos], new_seq[pos+1] = new_seq[pos+1], new_seq[pos]
                        if all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1)):
                            new_sums = compute_block_sums(new_seq)
                            new_dupes = count_duplicates(new_sums)
                            delta = new_dupes - current_dupes
                            if delta < 0 or random.random() < 0.05:
                                current = new_seq
                                current_sums = set(new_sums)
                                current_dupes = new_dupes
                                if current_dupes == 0 and len(current) > best_local_len:
                                    best_local = list(current)
                                    best_local_dupes = 0
                                    best_local_len = len(current)
            
            temp *= cooling
        
        return best_local

    # Try multiple restarts with different random seeds
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a perturbed initial sequence
        seq = list(best_sequence)
        
        # Randomly shuffle some elements or try different constructions
        if random.random() < 0.5:
            # Try a fresh greedy with different starting point
            values2, sums2 = [], set()
            # Start with a random subset
            start_val = random.randint(1, 50)
            for x in range(start_val, 257):
                tails, total = [x], x
                for a in reversed(values2):
                    total += a
                    tails.append(total)
                if not sums2.intersection(tails):
                    values2.append(x)
                    sums2.update(tails)
            if len(values2) > len(seq):
                seq = values2
        
        # Local search
        result = local_search(seq, max_iters=2000)
        if len(result) > len(best_sequence) and is_valid(result):
            best_sequence = result
            best_len = len(result)
        
        # Try to extend the best sequence further
        if best_len < 256:
            # Try adding values at the end
            for x in range(1, 257):
                if x in best_sequence:
                    continue
                # Check if we can add x
                new_seq = best_sequence + [x]
                if not all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1)):
                    continue
                sums_new = compute_block_sums(new_seq)
                if len(sums_new) == len(set(sums_new)):
                    best_sequence = new_seq
                    best_len = len(best_sequence)
                    break

    # Final verification
    if not is_valid(best_sequence):
        # Fall back to the original greedy result
        return values
    
    return best_sequence
# EVOLVE-BLOCK-END
