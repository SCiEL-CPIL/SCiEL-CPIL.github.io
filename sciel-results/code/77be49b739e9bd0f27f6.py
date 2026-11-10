import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Greedy Baseline
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy()
    best_len = len(best_seq)
    
    # 2. Local Search Engine
    def get_penalty(seq):
        # Returns number of duplicate sums
        # len(seq) is L
        L = len(seq)
        counts = {}
        for i in range(L):
            s = 0
            for j in range(i, L):
                s += seq[j]
                counts[s] = counts.get(s, 0) + 1
        penalty = 0
        for c in counts.values():
            if c > 1:
                penalty += c - 1
        return penalty

    def random_increasing_seq(L):
        # Pick L distinct numbers from 1 to 256
        # Sort them
        return sorted(random.sample(range(1, 257), L))

    def sa_optimize(seq, time_limit):
        # seq is a list
        L = len(seq)
        # Precompute prefix sums? No, dynamic.
        
        # Initialize state
        current_seq = list(seq)
        current_penalty = get_penalty(current_seq)
        if current_penalty == 0:
            return current_seq, 0
        
        best_seq_so_far = list(current_seq)
        best_penalty = current_penalty
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999 # Tune this
        
        # Precompute intervals for each index i
        # intervals[i] = list of (j, k) such that j <= i <= k
        # Actually, we just need to iterate j from 0 to i, and k from i to L-1
        # Sum = prefix[k+1] - prefix[j]
        # If we change a[i], we need to update sums where i is in [j, k].
        
        # To speed up, we can maintain a list of current sums?
        # No, just recalculate the affected ones.
        
        start = time.time()
        while time.time() - start < time_limit:
            # Pick move
            i = random.randint(0, L - 1)
            
            # Determine valid range for a[i]
            low = current_seq[i - 1] + 1 if i > 0 else 1
            high = current_seq[i + 1] - 1 if i < L - 1 else 256
            
            if low > high:
                continue
                
            # Pick new value
            # Bias towards current value? Or uniform?
            # Uniform is safer.
            v = random.randint(low, high)
            if v == current_seq[i]:
                continue
                
            # Calculate delta penalty
            # We need to update sums involving i.
            # Old sums:
            old_val = current_seq[i]
            diff = v - old_val
            
            # We will simulate the change
            # We need the current counts of sums.
            # To avoid O(L^2) recalculation of ALL sums, we need a global count dict.
            # But maintaining a global count dict with O(L^2) updates per step is O(L^2).
            # This is what we planned.
            
            # Let's maintain a global `sum_counts` dict outside the loop?
            # No, it's easier to just recalculate penalty if we don't have incremental updates.
            # Recalculating penalty is O(L^2).
            # L=40, L^2=1600.
            # 10,000 steps * 1600 = 16M ops.
            # In Python, 16M ops might take 1-2 seconds.
            # This is acceptable.
            # So, simple approach:
            # 1. Change a[i] to v
            # 2. Calculate new penalty
            # 3. Accept or reject
            # 4. If reject, revert
            
            current_seq[i] = v
            new_penalty = get_penalty(current_seq)
            
            delta = new_penalty - current_penalty
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_seq_so_far = list(current_seq)
            else:
                current_seq[i] = old_val
                
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_seq_so_far, best_penalty

    # 3. Main Search Loop
    # Try to extend length
    # Start from best_len + 1
    
    # We know 40 is possible.
    # Let's try to find 41.
    
    # We will allocate time slices to each length attempt.
    # If we find a valid sequence of length L, we can try L+1.
    
    # To be safe, we first ensure we have a good sequence of length best_len.
    # Then try best_len + 1.
    
    # Since greedy gives 39, and we know 40 exists, let's target 40 first.
    # If we find 40, target 41.
    
    target_len = best_len
    # If best_len is 39, try 40.
    
    # Let's try to improve the current best_seq first?
    # No, greedy is already valid (penalty 0).
    
    # So we just need to find a valid sequence of length > best_len.
    
    # We'll try lengths in increasing order.
    # Allocate 10 seconds per length attempt?
    # Total 55s.
    
    # If best_len is 39, try 40.
    # If 40 works, try 41.
    
    # But what if 40 is hard to find from random start?
    # We can seed the random start with the greedy sequence extended?
    # Or just random.
    
    # Let's try random start for target_len.
    
    # We'll run multiple restarts if time permits.
    
    # Strategy:
    # 1. Try to find valid seq of length L_target.
    # 2. If found, update best.
    # 3. Increment L_target.
    
    # To speed up finding the first valid seq, we can use the greedy sequence as a "hint".
    # But greedy is length 39.
    
    # Let's just use random starts.
    
    # Time management:
    # We have 55s.
    # Let's give 15s to try length 40.
    # If found, 15s to try length 41.
    # If found, 15s to try length 42.
    
    # But if 40 is not found in 15s, we might still be at 39.
    
    # Let's make it dynamic.
    
    current_target = best_len + 1
    if current_target > 256:
        return best_seq
        
    while time.time() < deadline - 5.0:
        # Try to find a valid sequence of length current_target
        found = False
        attempts = 0
        # Give 10 seconds per attempt? Or 5 seconds?
        # Let's do 5 seconds per SA run.
        # If not found, try again with new random seed.
        
        # Actually, just run SA for a chunk of time.
        chunk_time = 5.0
        if time.time() + chunk_time > deadline - 2.0:
            chunk_time = deadline - 2.0 - time.time()
            if chunk_time < 1.0: break
            
        # Generate random start
        # To help convergence, we can sort a random sample.
        start_seq = random_increasing_seq(current_target)
        
        # Run SA
        res_seq, res_pen = sa_optimize(start_seq, chunk_time)
        
        if res_pen == 0:
            # Found valid sequence
            if len(res_seq) > len(best_seq):
                best_seq = res_seq
                best_len = len(best_seq)
            # Try to go higher
            current_target += 1
            if current_target > 256:
                break
        else:
            # Not found yet.
            # If we have tried this length many times and failed, maybe move on?
            # Or keep trying?
            # For now, keep trying if time permits.
            # But if we are stuck at 39 and can't find 40, we should return 39.
            # We know 40 exists.
            
            # Heuristic: If we've spent > 20s on a length and failed, increment target?
            # No, if 40 exists, we should find it.
            # Maybe random start is bad.
            
            # Let's try a different start strategy:
            # Take the best_seq (length 39) and insert a random element?
            # This preserves the "good" structure of the greedy sequence.
            
            if attempts > 0 and attempts % 5 == 0:
                 # Try hybrid start
                 # Insert random element into best_seq
                 # But best_seq is length 39. Target is 40.
                 # So just insert one element.
                 pass # Implement if needed

    return best_seq

# Helper for random increasing sequence
def random_increasing_seq(L):
    return sorted(random.sample(range(1, 257), L))
