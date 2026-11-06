import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    
    # Pre-compute potential candidates
    all_candidates = list(range(1, max_val + 1))
    
    def get_sums(s):
        """Return a set of all pairwise sums a+b where a<=b from sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                b = s[j]
                sums.add(a + b)
        return sums

    def can_add(s, sums, x):
        """Check if x can be added to s without creating duplicate sums."""
        # Check sums involving x: x + y for y in s, and 2x
        # If any of these are already in 'sums', it's a conflict.
        # Note: sums contains all a+b for a,b in s.
        # If we add x, new sums are {x+y | y in s} U {2x}.
        # We must ensure none of these are in the existing 'sums'.
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_fill(s, sums, candidates, bias=None):
        """
        Greedily add elements from candidates to s.
        Returns the updated sorted list s and the updated sums set.
        bias: 'random', 'low', 'high'
        """
        s = sorted(s)
        sums = get_sums(s)
        
        # Filter candidates that are not in s
        s_set = set(s)
        avail = [c for c in candidates if c not in s_set]
        
        if bias == 'random':
            random.shuffle(avail)
        elif bias == 'low':
            avail.sort()
        elif bias == 'high':
            avail.sort(reverse=True)
        else:
            random.shuffle(avail)
            
        for x in avail:
            if time.time() > deadline - 2:
                break
            if can_add(s, sums, x):
                # Add x
                s.append(x)
                s.sort()
                # Update sums
                # New sums are x + y for y in old_s, and 2x
                # Since we maintain sorted s, we can just recompute or update incrementally.
                # Incremental update:
                for y in s[:-1]: # s[:-1] is the set before adding x? No, s was appended then sorted.
                    # Actually, let's just recompute sums for simplicity if n is small, 
                    # or update carefully.
                    # n is max 20. n^2 is 400. Recomputing is cheap.
                    pass
                
                # Recompute sums for correctness and simplicity
                sums = get_sums(s)
                
        return s, sums

    def kick_and_refill(current_set, current_sums, kick_size):
        """
        Remove 'kick_size' non-base elements from current_set, then greedy refill.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
            
        # Identify non-base elements
        non_base_elements = [x for x in current_set if x not in base_set]
        
        if len(non_base_elements) < kick_size:
            # Can't kick enough, just refill
            new_set, new_sums = greedy_fill(current_set, current_sums, all_candidates, bias='random')
            return new_set, new_sums
            
        # Randomly select elements to remove
        to_remove = random.sample(non_base_elements, kick_size)
        remove_set = set(to_remove)
        
        # Create new set without removed elements
        new_set = [x for x in current_set if x not in remove_set]
        new_sums = get_sums(new_set)
        
        # Refill
        new_set, new_sums = greedy_fill(new_set, new_sums, all_candidates, bias='random')
        
        return new_set, new_sums

    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    # Try several initial greedy fills with different biases to find a good starting point
    biases = ['random', 'low', 'high', 'random', 'random']
    for bias in biases:
        if time.time() > deadline - 20:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)
            
    # Local Search Loop
    # We use a strategy of kicking 1, 2, or 3 elements and refilling.
    # We track the current state and try to improve upon the best.
    
    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 50 # Restart or change strategy if stuck
    
    while time.time() < deadline - 10:
        
        # Determine kick size
        # Mostly kick 1 or 2, occasionally 3
        r = random.random()
        if r < 0.5:
            kick_size = 1
        elif r < 0.9:
            kick_size = 2
        else:
            kick_size = 3
            
        new_set, new_sums = kick_and_refill(current_set, current_sums, kick_size)
        
        if len(new_set) > best_size:
            best_set = new_set
            best_sums = new_sums
            best_size = len(new_set)
            current_set = new_set
            current_sums = new_sums
            current_size = best_size
            no_improve_count = 0
        elif len(new_set) == best_size:
            # Accept equal size moves to explore the landscape
            # Only accept if it's different or randomly to avoid cycles
            if new_set != current_set:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
        else:
            # Worse size, reject (or accept with low prob? No, we want max size)
            # We stick to the current best or current state.
            # Since we always refill greedily, we often stay at the same size or drop.
            # If we drop, we lost elements. We should probably revert to current_set if it was better.
            # But kick_and_refill returns the result of the kick.
            # If the result is smaller, it means the refill couldn't recover the size.
            # We should probably not update current_set if it got smaller, unless we are exploring.
            # Let's update current_set to the new state to allow further kicks from a "broken" state?
            # Usually, it's better to keep the best valid state as the anchor.
            # So we only update current_set if it's at least as good as the previous current_set?
            # Actually, if we kick 2 elements and refill adds only 1, size drops by 1.
            # If we keep this as current, we are in a smaller set.
            # It's often better to revert to the last "good" state if we fail to improve.
            # However, sometimes a "dip" is needed.
            # Let's accept the new state as current if it's not drastically worse, or just always accept 
            # the result of the kick to allow random walk.
            current_set = new_set
            current_sums = new_sums
            current_size = len(new_set)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Reset current to best to restart exploration from the best known point
            current_set = best_set[:]
            current_sums = best_sums
            current_size = best_size
            no_improve_count = 0
            # Maybe try a different bias for the next kick?
            # Or just let random kick handle it.

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
