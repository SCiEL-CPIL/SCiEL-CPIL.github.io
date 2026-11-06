import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    def get_consecutive_sums(seq):
        """Get all sums of non-empty consecutive blocks."""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def is_valid(seq):
        """Check if sequence is strictly increasing and all consecutive block sums are distinct."""
        if not seq:
            return True
        # Check strictly increasing
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        # Check distinct consecutive block sums
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def greedy_build():
        """Greedy: pick smallest valid next element."""
        seq = []
        sums = set()
        for x in range(1, 257):
            # Compute new consecutive sums if we append x
            new_sums = [x]
            total = x
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            if not sums.intersection(new_sums):
                seq.append(x)
                sums.update(new_sums)
        return seq
    
    # Start with greedy baseline
    best_seq = greedy_build()
    best_len = len(best_seq)
    
    # Try to improve using simulated annealing / local search
    # We'll try to find sequences of length best_len + 1 or more
    
    def try_extend(seq, max_extra=2):
        """Try to extend sequence by adding elements at the end."""
        current = list(seq)
        current_sums = get_consecutive_sums(current)
        last_val = current[-1] if current else 0
        
        # Try to add 1 or 2 more elements
        for target_len in range(len(current) + 1, len(current) + max_extra + 1):
            if time.time() > deadline:
                break
            
            # Backtracking search for extension
            def search_extend(vals, sums_set, last_v):
                nonlocal best_seq, best_len, deadline
                
                if time.time() > deadline:
                    return False
                
                if len(vals) == target_len:
                    if len(vals) > best_len:
                        best_len = len(vals)
                        best_seq = list(vals)
                    return True
                
                # Pruning: if we need k more elements, min possible sum increase
                k = target_len - len(vals)
                min_next = last_v + 1
                # The smallest k elements would be min_next, min_next+1, ..., min_next+k-1
                # Their total contribution to the sum is at least k*min_next + k*(k-1)//2
                # But we also need to ensure all new consecutive sums don't collide
                
                for x in range(last_v + 1, 257):
                    # Check if x can be added
                    new_sums = [x]
                    total = x
                    for a in reversed(vals):
                        total += a
                        new_sums.append(total)
                    
                    if sums_set.intersection(new_sums):
                        continue
                    
                    new_vals = vals + [x]
                    new_sums_set = sums_set.union(new_sums)
                    
                    if search_extend(new_vals, new_sums_set, x):
                        return True
                return False
            
            search_extend(current, current_sums, last_val)
            
            if len(best_seq) >= target_len:
                return best_seq
        
        return best_seq
    
    # Try extending the greedy sequence
    best_seq = try_extend(best_seq, max_extra=3)
    
    # Now try a more systematic search: find sequences of length L where L > best_len
    # Use constraint programming with pruning
    
    def systematic_search(target_len):
        """Search for a valid sequence of exactly target_len elements."""
        if time.time() > deadline:
            return None
        
        def backtrack(vals, sums_set, last_v, prefix_sum):
            nonlocal best_seq, best_len, deadline
            
            if time.time() > deadline:
                return None
            
            if len(vals) == target_len:
                return list(vals)
            
            k = target_len - len(vals)
            
            # Lower bound on next value
            min_x = last_v + 1
            
            # Upper bound: we need room for k elements, each at least 1 more than previous
            # So max_x such that we can still fit k-1 more elements after it
            # max_x + (max_x+1) + ... + (max_x + k - 1) <= 256
            # k*max_x + k*(k-1)//2 <= 256
            # max_x <= (256 - k*(k-1)//2) / k
            if k > 0:
                max_x_bound = (256 - k * (k - 1) // 2) // k
                max_x = min(256, max_x_bound)
            else:
                max_x = 256
            
            if min_x > max_x:
                return None
            
            for x in range(min_x, max_x + 1):
                # Quick check: new consecutive sums
                new_sums = [x]
                total = x
                for a in reversed(vals):
                    total += a
                    new_sums.append(total)
                
                if sums_set.intersection(new_sums):
                    continue
                
                result = backtrack(vals + [x], sums_set.union(new_sums), x, prefix_sum + x)
                if result is not None:
                    return result
            
            return None
        
        return backtrack([], set(), 0, 0)
    
    # Try to find longer sequences
    for target_len in range(best_len + 1, 16):
        if time.time() > deadline - 5:
            break
        
        result = systematic_search(target_len)
        if result is not None:
            best_seq = result
            best_len = target_len
        else:
            # If we can't find a sequence of this length, try one more and break
            # Actually, let's try one more to be safe
            if target_len > best_len + 2:
                break
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_build()
    
    return best_seq
