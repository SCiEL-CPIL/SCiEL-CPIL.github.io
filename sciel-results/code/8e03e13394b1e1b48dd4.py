import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a maximum-size Sidon set (Golomb ruler) in [1, 284]
    containing {1, 2, 5, 11}.
    """
    deadline = time.time() + 140  # Safety margin for 150s limit
    max_val = 284
    fixed_elements = {1, 2, 5, 11}
    
    best_size = 0
    best_set = list(fixed_elements)
    
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(s, sums, x):
        # Check if adding x creates any duplicate sums
        # New sums are x + s[i] for all i, and 2*x
        # We need to ensure none of these are already in `sums`
        # And that the new sums among themselves are distinct (which is true for distinct x and s[i] if s is Sidon, 
        # but we must check against existing sums)
        # Actually, if s is Sidon, the new sums {x+s[i]} are distinct.
        # We just need to check if any new sum is in `sums`.
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_expand(s, sums, candidates, rng):
        """
        Greedily add elements from candidates to s.
        Returns the updated s and sums.
        """
        s = list(s)
        sums = set(sums)
        cands = list(candidates)
        # Shuffle candidates to randomize the order of addition
        rng.shuffle(cands)
        
        for x in cands:
            if x in s:
                continue
            if is_valid_addition(s, sums, x):
                s.append(x)
                s.sort()
                # Update sums
                for y in s[:-1]: # s[:-1] is the old s
                    sums.add(x + y)
                sums.add(2 * x)
                
        return s, sums

    def local_search(s, sums, rng, time_limit):
        """
        Perform local search by kicking (removing) and re-filling.
        """
        current_s = list(s)
        current_sums = set(sums)
        best_local_s = list(current_s)
        best_local_sums = set(current_sums)
        
        # Pre-compute all possible candidates
        all_candidates = list(range(1, max_val + 1))
        
        while time.time() < time_limit:
            # Try to add any valid element
            improved = False
            # Shuffle candidates for randomization
            cands = [x for x in all_candidates if x not in current_s]
            rng.shuffle(cands)
            
            for x in cands:
                if is_valid_addition(current_s, current_sums, x):
                    # Add x
                    current_s.append(x)
                    current_s.sort()
                    for y in current_s[:-1]:
                        current_sums.add(x + y)
                    current_sums.add(2 * x)
                    improved = True
                    break
            
            if not improved:
                # Stuck. Perform a kick.
                # Remove a random non-fixed element
                removable = [x for x in current_s if x not in fixed_elements]
                if not removable:
                    break
                
                # Sometimes remove 2 elements if stuck deeply? Let's stick to 1 for now.
                # To escape local optima, we might want to remove an element that is "crowding" others.
                # Simple random kick:
                x_to_remove = rng.choice(removable)
                
                # Remove x_to_remove
                current_s.remove(x_to_remove)
                # Recompute sums (expensive but safe)
                current_sums = get_sums(current_s)
                
                # After removing, try to add again
                cands = [x for x in all_candidates if x not in current_s]
                rng.shuffle(cands)
                for x in cands:
                    if is_valid_addition(current_s, current_sums, x):
                        current_s.append(x)
                        current_s.sort()
                        for y in current_s[:-1]:
                            current_sums.add(x + y)
                        current_sums.add(2 * x)
                        break
                
                # Update best if improved
                if len(current_s) > len(best_local_s):
                    best_local_s = list(current_s)
                    best_local_sums = set(current_sums)
        
        return best_local_s, best_local_sums

    # Main search loop with multiple restarts
    rng = random.Random()
    
    while time.time() < deadline:
        # Start with fixed elements
        current_s = list(fixed_elements)
        current_sums = get_sums(current_s)
        
        # Greedy phase
        candidates = [x for x in range(1, max_val + 1) if x not in fixed_elements]
        current_s, current_sums = greedy_expand(current_s, current_sums, candidates, rng)
        
        # Local search phase
        time_limit_for_ls = min(deadline - 5, time.time() + 5) # Give local search some time
        if time.time() < deadline - 2:
             current_s, current_sums = local_search(current_s, current_sums, rng, deadline - 2)
        
        # Update global best
        if len(current_s) > best_size:
            best_size = len(current_s)
            best_set = list(current_s)
            
            # If we reached the theoretical max, we can stop
            if best_size >= 20:
                break

    return best_set
# EVOLVE-BLOCK-END
