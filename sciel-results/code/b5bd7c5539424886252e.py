import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s kill
    
    best_seq = []
    best_len = 0
    
    # Precompute bit masks for prefix sums up to a reasonable max
    # Max possible sum: 256 * 256 = 65536
    MAX_SUM = 65536
    
    def update_state(seq):
        """
        Rebuilds prefix_sums and forbidden_sums from a sequence.
        """
        prefix_sums = [0]
        forbidden_mask = 0  # bitset of forbidden sums
        for x in seq:
            new_prefix = prefix_sums[-1] + x
            for p in prefix_sums:
                d = new_prefix - p
                if d >= 0 and d < MAX_SUM:
                    forbidden_mask |= (1 << d)
            prefix_sums.append(new_prefix)
        return prefix_sums, forbidden_mask

    def is_valid_candidate(x, prefix_sums, forbidden_mask):
        """
        Checks if adding x to the current sequence (ending at prefix_sums[-1])
        results in a valid sequence.
        """
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d >= MAX_SUM or (forbidden_mask >> d) & 1:
                return False
        return True

    def greedy_construct(seed=None, prob_smallest=0.8):
        if seed is not None:
            random.seed(seed)
        
        seq = []
        prefix_sums = [0]
        forbidden_mask = 0
        last = 0
        
        for _ in range(256):
            valid_candidates = []
            for x in range(last + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_mask):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if random.random() < prob_smallest:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            
            for p in prefix_sums[:-1]:
                d = new_prefix - p
                if d >= 0 and d < MAX_SUM:
                    forbidden_mask |= (1 << d)
            
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

    # Phase 1: Randomized Greedy to get a strong baseline
    attempt_count = 0
    greedy_deadline = start_time + 8.0  # Reserve 8s for greedy
    while time.time() < greedy_deadline:
        # Vary the probability to explore different structures
        prob = 0.3 + 0.6 * random.random()
        seq = greedy_construct(seed=attempt_count, prob_smallest=prob)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        attempt_count += 1
        if attempt_count > 200:
            break

    def extend_sequence(seq, target_len, deadline, random_order=True):
        """
        Tries to extend 'seq' to length 'target_len' using DFS with bitset optimization.
        """
        if len(seq) >= target_len:
            return seq[:]
        
        prefix_sums, forbidden_mask = update_state(seq)
        last_val = seq[-1] if seq else 0
        current_seq = seq[:]
        
        def _dfs_ext(depth, last_val, prefix_sums, forbidden_mask, current_seq):
            if time.time() > deadline:
                return None
            
            # Pruning: Can we reach target_len?
            if depth + (256 - last_val) < target_len:
                return None
            
            if depth == target_len:
                return current_seq[:]
            
            candidates = []
            for x in range(last_val + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_mask):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            if random_order:
                random.shuffle(candidates)
            else:
                # Try smaller values first
                candidates.sort()
            
            for x in candidates:
                new_prefix = prefix_sums[-1] + x
                new_diffs = []
                valid = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d >= MAX_SUM or (forbidden_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Commit
                prefix_sums.append(new_prefix)
                new_mask = forbidden_mask
                for d in new_diffs:
                    new_mask |= (1 << d)
                current_seq.append(x)
                
                result = _dfs_ext(depth + 1, x, prefix_sums, new_mask, current_seq)
                
                if result is not None:
                    return result
                
                # Backtrack
                current_seq.pop()
                prefix_sums.pop()
                
                if time.time() > deadline:
                    return None
            
            return None

        return _dfs_ext(len(current_seq), last_val, prefix_sums, forbidden_mask, current_seq)

    # Phase 2: Target-Length Extension with multiple strategies
    dfs_deadline = start_time + time_limit - 3.0
    
    current_target = best_len + 1
    consecutive_failures = 0
    
    while current_target <= 256 and time.time() < dfs_deadline:
        found = False
        
        # Try multiple strategies to extend
        # Strategy 1: Extend directly from best_seq
        for k in range(min(len(best_seq), 5)):
            if time.time() > dfs_deadline:
                break
            
            base_seq = best_seq[:len(best_seq) - k]
            if len(base_seq) >= current_target:
                continue
            
            # Try with different random orders
            for attempt in range(3):
                if time.time() > dfs_deadline:
                    break
                result = extend_sequence(base_seq, current_target, dfs_deadline, 
                                        random_order=(attempt % 2 == 0))
                if result is not None:
                    best_seq = result
                    best_len = current_target
                    found = True
                    break
            
            if found:
                break
        
        if found:
            consecutive_failures = 0
            current_target += 1
            continue
        
        # If direct extension failed, try fresh greedy + extend
        if time.time() < dfs_deadline - 5:
            new_seed = random.randint(0, 100000)
            new_greedy = greedy_construct(seed=new_seed, prob_smallest=0.6)
            if len(new_greedy) > best_len:
                best_seq = new_greedy
                best_len = len(new_greedy)
                current_target = best_len + 1
                continue
            
            # Try extending the new greedy sequence
            if len(new_greedy) < current_target:
                result = extend_sequence(new_greedy, current_target, dfs_deadline, random_order=True)
                if result is not None:
                    best_seq = result
                    best_len = current_target
                    found = True
        
        if not found:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                # Give up on this target, try next
                current_target += 1
                consecutive_failures = 0
            else:
                # Try a different approach: truncate more aggressively
                continue
    
    return best_seq
