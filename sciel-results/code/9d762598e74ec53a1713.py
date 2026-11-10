import time
import sys
import random

def construct():
    start_time = time.time()
    time_limit = 80.0  # Leave 10s margin for 90s limit

    best_seq = []
    best_len = 0

    # Helper: compute all consecutive block sums for a sequence
    def all_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    # Helper: check if appending x to seq creates any duplicate block sums
    # Only need to check new blocks ending at x
    def check_append(seq, x, existing_sums):
        """
        Returns True if x can be appended without creating duplicate block sums.
        Also returns the set of new sums if valid.
        """
        n = len(seq)
        new_sums = set()
        total = x
        new_sums.add(total)
        for i in range(n - 1, -1, -1):
            total += seq[i]
            new_sums.add(total)
        
        # Check for collisions
        for s in new_sums:
            if s in existing_sums:
                return False, None
        return True, new_sums

    # Helper: count number of duplicate block sums in a sequence
    def count_duplicates(seq):
        """
        Returns the number of duplicate block sums (counting each duplicate pair once).
        """
        from collections import Counter
        n = len(seq)
        sums = []
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.append(total)
        c = Counter(sums)
        duplicates = 0
        for s, cnt in c.items():
            if cnt > 1:
                duplicates += cnt - 1
        return duplicates

    # Phase 1: Deterministic greedy to get a good starting point
    def greedy_deterministic():
        values = []
        sums = set()
        for x in range(1, 257):
            valid, new_sums = check_append(values, x, sums)
            if valid:
                values.append(x)
                sums.update(new_sums)
        return values

    # Phase 2: Randomized local search using difference vectors
    def local_search(initial_seq, deadline):
        """
        Local search that perturbs the sequence and accepts moves based on
        the count of duplicate block sums.
        """
        rng = random.Random(int(start_time * 1000) % 100000)
        
        # Start from initial sequence
        current = list(initial_seq)
        current_sums = all_block_sums(current)
        
        # Track best valid sequence found
        best_local = list(current)
        best_local_len = len(current)
        
        # If current is valid, that's our best
        if len(current_sums) == len(current) * (len(current) + 1) // 2:
            best_local = list(current)
            best_local_len = len(current)
        
        # Simulated annealing parameters
        temperature = 5.0
        cooling = 0.995
        
        while time.time() < deadline:
            if temperature < 0.01:
                break
            
            # Generate a random perturbation
            # Option 1: Remove a random element
            # Option 2: Change a random element
            # Option 3: Insert a random element at a random position
            
            n = len(current)
            move_type = rng.random()
            
            if move_type < 0.3 and n > 0:
                # Remove a random element
                idx = rng.randint(0, n - 1)
                new_seq = current[:idx] + current[idx+1:]
            elif move_type < 0.6:
                # Change a random element to a nearby value
                if n == 0:
                    continue
                idx = rng.randint(0, n - 1)
                old_val = current[idx]
                # Try to change to a value that maintains strict increase
                if idx > 0:
                    min_val = current[idx-1] + 1
                else:
                    min_val = 1
                if idx < n - 1:
                    max_val = current[idx+1] - 1
                else:
                    max_val = 256
                
                if min_val <= max_val:
                    new_val = rng.randint(min_val, max_val)
                    new_seq = current[:idx] + [new_val] + current[idx+1:]
                else:
                    continue
            else:
                # Insert a random element at a random position
                # Try to insert a value that maintains strict increase
                insert_pos = rng.randint(0, n)
                if insert_pos == 0:
                    min_val = 1
                    max_val = (current[0] - 1) if n > 0 else 256
                elif insert_pos == n:
                    min_val = current[-1] + 1
                    max_val = 256
                else:
                    min_val = current[insert_pos - 1] + 1
                    max_val = current[insert_pos] - 1
                
                if min_val <= max_val:
                    new_val = rng.randint(min_val, max_val)
                    new_seq = current[:insert_pos] + [new_val] + current[insert_pos:]
                else:
                    continue
            
            # Check if new sequence is strictly increasing
            valid_increase = True
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i+1]:
                    valid_increase = False
                    break
            
            if not valid_increase:
                continue
            
            # Compute duplicate count
            new_sums = all_block_sums(new_seq)
            expected_sums = len(new_seq) * (len(new_seq) + 1) // 2
            new_dups = expected_sums - len(new_sums)
            
            # Compute current duplicate count
            current_dups = len(current) * (len(current) + 1) // 2 - len(current_sums)
            
            # Acceptance criterion
            if new_dups < current_dups:
                # Better, always accept
                current = new_seq
                current_sums = new_sums
                if len(current) > best_local_len and new_dups == 0:
                    best_local = list(current)
                    best_local_len = len(current)
            elif new_dups == current_dups:
                # Same quality, accept with some probability
                if rng.random() < 0.3:
                    current = new_seq
                    current_sums = new_sums
                    if len(current) > best_local_len and new_dups == 0:
                        best_local = list(current)
                        best_local_len = len(current)
            else:
                # Worse, accept with probability based on temperature
                delta = new_dups - current_dups
                prob = min(1.0, temperature / max(1, delta))
                if rng.random() < prob:
                    current = new_seq
                    current_sums = new_sums
            
            temperature *= cooling
        
        return best_local

    # Phase 3: Try to extend a valid sequence by searching for the next element
    def extend_sequence(seq, sums, deadline):
        """
        Try to extend the sequence by finding valid next elements.
        """
        current = list(seq)
        current_sums = set(sums)
        
        while time.time() < deadline:
            last = current[-1] if current else 0
            found = False
            
            # Try elements in order from last+1 to 256
            for x in range(last + 1, 257):
                valid, new_sums = check_append(current, x, current_sums)
                if valid:
                    current.append(x)
                    current_sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                break
        
        return current

    # Main execution
    # Step 1: Get initial greedy solution
    greedy_result = greedy_deterministic()
    if len(greedy_result) > best_len:
        best_seq = list(greedy_result)
        best_len = len(greedy_result)
    
    greedy_sums = all_block_sums(greedy_result)
    
    # Step 2: Try to extend the greedy result
    extended = extend_sequence(greedy_result, greedy_sums, start_time + time_limit * 0.3)
    if len(extended) > best_len:
        best_seq = list(extended)
        best_len = len(extended)
    
    # Step 3: Run local search from multiple starting points
    # Use the greedy result and some variants
    starting_points = [greedy_result]
    
    # Create some perturbed starting points
    for _ in range(5):
        if time.time() > start_time + time_limit * 0.4:
            break
        perturbed = list(greedy_result)
        # Remove 1-3 random elements
        num_removals = random.randint(1, min(3, len(perturbed) - 1))
        indices_to_remove = random.sample(range(len(perturbed)), num_removals)
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            perturbed.pop(idx)
        if len(perturbed) > 0:
            starting_points.append(perturbed)
    
    # Run local search on each starting point
    for sp in starting_points:
        if time.time() > start_time + time_limit * 0.8:
            break
        
        result = local_search(sp, start_time + time_limit * 0.8)
        if len(result) > best_len:
            # Verify the result is valid
            result_sums = all_block_sums(result)
            expected = len(result) * (len(result) + 1) // 2
            if len(result_sums) == expected:
                best_seq = list(result)
                best_len = len(result)
    
    # Final: try to extend the best sequence found
    if time.time() < start_time + time_limit * 0.95:
        best_sums = all_block_sums(best_seq)
        extended_best = extend_sequence(best_seq, best_sums, start_time + time_limit * 0.95)
        if len(extended_best) > best_len:
            # Verify
            ext_sums = all_block_sums(extended_best)
            expected = len(extended_best) * (len(extended_best) + 1) // 2
            if len(ext_sums) == expected:
                best_seq = list(extended_best)
                best_len = len(extended_best)
    
    return best_seq

if __name__ == "__main__":
    print(construct())
