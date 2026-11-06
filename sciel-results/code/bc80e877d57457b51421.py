import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill
    
    best_seq = []
    best_len = 0
    
    def is_valid_candidate(x, prefix_sums, forbidden_sums):
        """
        Checks if adding x to the current sequence (ending at prefix_sums[-1])
        results in a valid sequence.
        """
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d in forbidden_sums:
                return False
        return True

    def greedy_construct(seed=None, prob_smallest=0.8, use_largest=False):
        if seed is not None:
            random.seed(seed)
        
        seq = []
        prefix_sums = [0]
        forbidden_sums = set()
        last = 0
        
        for _ in range(256):
            valid_candidates = []
            # Optimization: iterate and collect valid candidates
            for x in range(last + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Selection Strategy
            if use_largest:
                x = valid_candidates[-1]
            elif random.random() < prob_smallest:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            # Commit x
            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            
            # Update forbidden_sums
            for p in prefix_sums[:-1]:
                forbidden_sums.add(new_prefix - p)
            
        return seq

    def is_valid_sequence(seq):
        if not seq:
            return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 1 or x > 256 for x in seq):
            return False
        p = [0]
        for x in seq:
            p.append(p[-1] + x)
        sums = set()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                s = p[j] - p[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def local_search_with_insertion(seq, time_limit, seed=None):
        """
        Attempts to improve the sequence by:
        1. Removing elements and re-extending
        2. Trying to insert elements between existing ones
        """
        if seed is not None:
            random.seed(seed)
        
        current_seq = seq[:]
        best_local_len = len(current_seq)
        best_local_seq = current_seq[:]
        
        # Strategy 1: Remove element at position i and re-greedy extend
        for i in range(len(current_seq)):
            if time.time() > time_limit:
                break
            
            # Remove element at i
            temp_seq = current_seq[:i] + current_seq[i+1:]
            
            # Check validity of temp_seq
            if not is_valid_sequence(temp_seq):
                continue
                
            # Now, can we add more elements to temp_seq?
            # We extend from the end.
            extended_seq = temp_seq[:]
            prefix_sums = [0]
            forbidden_sums = set()
            for x in extended_seq:
                new_prefix = prefix_sums[-1] + x
                for p in prefix_sums:
                    forbidden_sums.add(new_prefix - p)
                prefix_sums.append(new_prefix)
            
            # Greedy extend
            last_val = extended_seq[-1] if extended_seq else 0
            for _ in range(256 - len(extended_seq)):
                valid_candidates = []
                for x in range(last_val + 1, 257):
                    if is_valid_candidate(x, prefix_sums, forbidden_sums):
                        valid_candidates.append(x)
                if not valid_candidates:
                    break
                x = valid_candidates[0]
                extended_seq.append(x)
                last_val = x
                new_p = prefix_sums[-1] + x
                for p in prefix_sums:
                    forbidden_sums.add(new_p - p)
                prefix_sums.append(new_p)
                
            if len(extended_seq) > best_local_len:
                best_local_len = len(extended_seq)
                best_local_seq = extended_seq[:]
                
        # Strategy 2: Try inserting an element between existing elements
        if time.time() > time_limit:
            return best_local_seq
            
        base_seq = best_local_seq[:]
        for insert_pos in range(1, len(base_seq)):
            if time.time() > time_limit:
                break
                
            lower = base_seq[insert_pos - 1]
            upper = base_seq[insert_pos]
            
            # Try inserting values between lower and upper
            for x in range(lower + 1, upper):
                test_seq = base_seq[:insert_pos] + [x] + base_seq[insert_pos:]
                if is_valid_sequence(test_seq):
                    if len(test_seq) > best_local_len:
                        best_local_len = len(test_seq)
                        best_local_seq = test_seq[:]
                        # Try to extend further
                        prefix_sums = [0]
                        forbidden_sums = set()
                        for val in best_local_seq:
                            new_prefix = prefix_sums[-1] + val
                            for p in prefix_sums:
                                forbidden_sums.add(new_prefix - p)
                            prefix_sums.append(new_prefix)
                        
                        last_val = best_local_seq[-1]
                        for _ in range(256 - len(best_local_seq)):
                            valid_candidates = []
                            for x2 in range(last_val + 1, 257):
                                if is_valid_candidate(x2, prefix_sums, forbidden_sums):
                                    valid_candidates.append(x2)
                            if not valid_candidates:
                                break
                            x2 = valid_candidates[0]
                            best_local_seq.append(x2)
                            last_val = x2
                            new_p = prefix_sums[-1] + x2
                            for p in prefix_sums:
                                forbidden_sums.add(new_p - p)
                            prefix_sums.append(new_p)
                            
                            if len(best_local_seq) > best_local_len:
                                best_local_len = len(best_local_seq)
                            
                        break  # Only try first valid insertion at this position
        
        return best_local_seq

    # Phase 1: Randomized Greedy with multiple strategies
    attempt_count = 0
    # Run a large number of greedy attempts
    greedy_deadline = start_time + time_limit - 15
    while time.time() < greedy_deadline:
        # Strategy A: Mostly smallest
        seq = greedy_construct(seed=attempt_count, prob_smallest=0.85)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        # Strategy B: Balanced
        if attempt_count % 3 == 0:
            seq2 = greedy_construct(seed=attempt_count + 1000, prob_smallest=0.5)
            if len(seq2) > best_len:
                best_len = len(seq2)
                best_seq = seq2
        
        # Strategy C: Prefer larger values (sometimes helps)
        if attempt_count % 5 == 0:
            seq3 = greedy_construct(seed=attempt_count + 2000, use_largest=True)
            if len(seq3) > best_len:
                best_len = len(seq3)
                best_seq = seq3
        
        # Strategy D: Random selection
        if attempt_count % 7 == 0:
            seq4 = greedy_construct(seed=attempt_count + 3000, prob_smallest=0.3)
            if len(seq4) > best_len:
                best_len = len(seq4)
                best_seq = seq4
                
        attempt_count += 1
        
    # Phase 2: Local Search on Best Sequence
    if best_seq and time.time() < start_time + time_limit - 5:
        # Run local search with insertion
        improved_seq = local_search_with_insertion(best_seq, start_time + time_limit - 5, seed=42)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq
        
        # Try another local search with different seed
        if time.time() < start_time + time_limit - 2:
            improved_seq2 = local_search_with_insertion(best_seq, start_time + time_limit - 2, seed=123)
            if len(improved_seq2) > best_len:
                best_len = len(improved_seq2)
                best_seq = improved_seq2
            
    return best_seq
