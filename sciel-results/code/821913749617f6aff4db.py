import time
import random
import sys

# EVOLVE-BLOCK-START

def _greedy_construct():
    """
    Standard greedy: pick the smallest valid integer.
    """
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

def _randomized_greedy(trial_seed, deadline, current_best_len):
    """
    Randomized greedy: pick a random valid integer from the available candidates.
    Stops early if it exceeds current best length.
    """
    rand = random.Random(trial_seed)
    values = []
    sums = set()
    current_max = 0
    
    # We iterate up to 256 times, but we can break if no valid x or time runs out
    for _ in range(256):
        if time.time() > deadline:
            return values
            
        valid_xs = []
        # Find all valid x > current_max
        for x in range(current_max + 1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                valid_xs.append(x)
                # Optimization: If we find many valid ones, we might not need all, 
                # but for L~15, checking all is fast. 
                # However, to allow "jumping", we collect them all.
                
        if not valid_xs:
            break
            
        # Heuristic: Bias towards smaller values? 
        # Pure random might pick large values and exhaust the range.
        # Let's try a mix: 50% smallest, 50% random from valid.
        # Or just uniform random as per the "inspiration" that worked.
        # The cell finding said "Random restarts... Measured: 1 -> 1.282".
        # Let's stick to uniform random choice from valid_xs.
        
        x = rand.choice(valid_xs)
        
        # Recompute tails for the chosen x to update sums
        # We already computed them in the loop above, but we didn't store them per x.
        # Re-computing is O(L), which is fine.
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
            
        values.append(x)
        sums.update(tails)
        current_max = x
        
        if len(values) > current_best_len:
            # Found a better sequence, return immediately to save time?
            # Or continue to see if we can get even longer?
            # Returning immediately is safer for time budget, 
            # but we might miss a much longer one.
            # Given the 90s limit, and that we run many trials, 
            # returning early on improvement is a good heuristic to spread time.
            return values
            
    return values

def _dfs_construct(target_len, deadline, initial_seq=None):
    """
    DFS to find a sequence of exactly target_len.
    Uses pruning based on used sums.
    """
    # State: (values, used_sums, last_val, suffix_sums)
    # suffix_sums[i] = sum of last i+1 elements
    # We want to find ANY sequence of length target_len.
    
    best_seq = None
    
    def dfs(values, used_sums, last_val, suffix_sums):
        nonlocal best_seq
        
        if time.time() > deadline:
            return False
            
        if len(values) == target_len:
            best_seq = values[:]
            return True
            
        if last_val >= 256:
            return False
            
        # Heuristic: Try smaller x first to keep values low?
        # Or try x in random order?
        # Smaller x is generally better for maximizing length (greedy insight).
        # Let's try ascending order.
        
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            new_suffixes = [x]
            total = x
            valid = True
            
            # Check if x itself is in used_sums
            if x in used_sums:
                continue
                
            for s in suffix_sums:
                total += s
                new_suffixes.append(total)
                if total in used_sums:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # If we get here, x is valid
            new_used_sums = used_sums.union(new_suffixes)
            
            # Recurse
            if dfs(values + [x], new_used_sums, x, new_suffixes):
                return True
                
        return False
        
    # Start DFS
    # If initial_seq is provided, we could start from there, 
    # but finding a *different* structure of the same length is harder.
    # Let's start from scratch to ensure we find a valid one if it exists.
    # However, starting from scratch for L=15 is a huge space.
    # To make this effective, we should use the greedy prefix as a hint?
    # No, let's just try to find *any* sequence of target_len.
    # If target_len is small (e.g., 10), it's fast. If large (e.g., 15), it's slow.
    
    # Optimization: We can use the greedy sequence's first few elements to constrain the search?
    # No, that biases it.
    
    # Let's run the DFS.
    dfs([], set(), 0, [])
    
    if best_seq:
        return best_seq
    return None

def construct():
    start_time = time.time()
    # 90s limit. Keep 5s buffer.
    deadline = start_time + 85.0
    
    # 1. Baseline Greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    # 2. Randomized Greedy Phase
    # Run many trials. 
    # Each trial is fast (O(256^2 * L) approx).
    # Let's allocate 60% of the time to this phase.
    randomized_deadline = start_time + (85.0 * 0.6)
    
    trial = 0
    while time.time() < randomized_deadline:
        # Use different seeds
        seed = trial * 101 + 7
        seq = _randomized_greedy(seed, randomized_deadline, best_len)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            # Optional: If we find a very long one, we might want to try to extend it further?
            # The function returns early if it beats best_len.
            # We can try to continue from this best_seq in subsequent randomized runs?
            # For now, fresh starts are better for diversity.
        trial += 1
        
        # Safety: If we've found a very long sequence (e.g., > 15), 
        # randomized search might be plateauing. 
        # But let's stick to the time budget.
        
    # 3. DFS Phase
    # Try to find a sequence of length best_len + 1, then +2, etc.
    # DFS is expensive. Only attempt if we have time left.
    # If best_len is already high (e.g., > 12), finding +1 is hard.
    # Let's try target_len = best_len + 1.
    
    if time.time() < deadline - 2.0:
        target = best_len + 1
        # Cap the target to avoid trying impossible lengths if best_len is already max?
        # Max possible is bounded by sum constraints. 
        # Sum of 1..256 is ~32896. Number of blocks is L(L+1)/2.
        # If L=15, blocks=120. Max sum of blocks is roughly L * MaxVal.
        # It's safe to try up to best_len + 3.
        
        max_target_attempts = 3
        for i in range(max_target_attempts):
            current_target = best_len + 1 + i
            if time.time() > deadline - 2.0:
                break
                
            # Run DFS for this target
            # We pass a deadline that is slightly before the global deadline
            # to ensure we return something.
            dfs_deadline = deadline - 2.0
            
            # To speed up DFS, we can use the current best_seq as a "hint" 
            # or just run it. 
            # Running from scratch is safer for correctness.
            
            found_seq = _dfs_construct(current_target, dfs_deadline)
            
            if found_seq:
                best_len = current_target
                best_seq = found_seq
            else:
                # If we couldn't find length L+1, it's unlikely we can find L+2 
                # in the remaining time with this simple DFS.
                # Break out of the loop.
                break

    return best_seq

# EVOLVE-BLOCK-END
