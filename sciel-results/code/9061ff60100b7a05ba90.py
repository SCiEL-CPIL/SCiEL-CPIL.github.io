import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Mandatory elements
    seeds = [1, 2, 5, 11]
    
    # We want to find the maximum size.
    # Max possible is 20.
    # We will try to find a valid set of size K, starting from K=20 down to 4.
    
    def get_conflicts(s):
        """
        Returns the number of conflicting pairs.
        A conflict exists if two distinct pairs (a,b) and (c,d) with a<=b, c<=d, 
        (a,b) != (c,d) have a+b == c+d.
        We count the number of "extra" pairs for each sum.
        If a sum appears k times, it contributes k-1 to the conflict count 
        (or k choose 2 depending on definition, but we just need 0).
        Let's just count the number of pairs that are part of a collision.
        Simpler: Count total pairs vs distinct sums.
        Total pairs = K*(K+1)/2.
        Distinct sums = len(set(sums)).
        Conflicts = Total pairs - Distinct sums.
        If Conflicts == 0, it is a Sidon set.
        """
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        total_pairs = len(s) * (len(s) + 1) // 2
        return total_pairs - len(sums)

    def random_valid_initialization(k):
        """
        Create a random set of size k containing seeds.
        We add random numbers and check if they are distinct.
        We don't check Sidon property here, just distinctness.
        """
        s = set(seeds)
        while len(s) < k:
            s.add(random.randint(1, 284))
        return list(s)

    def local_search(k, time_limit=5.0):
        """
        Try to find a valid Sidon set of size k.
        Uses hill climbing / simulated annealing on conflict count.
        """
        if k < 4:
            return seeds[:]
        
        # Initialize
        current = random_valid_initialization(k)
        current_conflicts = get_conflicts(current)
        
        best = current[:]
        best_conflicts = current_conflicts
        
        if best_conflicts == 0:
            return best
        
        start = time.time()
        temp = 1.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Move: Swap a random element in current with a random element from [1, 284]
            idx = random.randint(0, k - 1)
            old_val = current[idx]
            new_val = random.randint(1, 284)
            
            # If new_val is already in the set (and not at idx), skip to maintain distinctness
            # Actually, if we swap, we might introduce duplicates.
            # We should ensure the set remains a set of distinct integers.
            # Let's pick new_val from the complement of the set.
            # This is expensive to check every time.
            # Instead, just try the swap. If it creates a duplicate, the conflict count will go up significantly (or invalid).
            # But distinctness is a hard constraint.
            # Better move: Pick an element to remove, and an element to add from [1,284]\current.
            
            # To keep it simple and fast, let's just swap values.
            # If new_val is in current (at j != idx), we have a duplicate.
            # We can just check if new_val is in current.
            # O(K) check. K is ~20. Fast enough.
            if new_val in current and new_val != old_val:
                continue
                
            current[idx] = new_val
            new_conflicts = get_conflicts(current)
            
            delta = new_conflicts - current_conflicts
            
            # Accept if better, or with probability exp(-delta/temp)
            if delta < 0 or (delta > 0 and random.random() < pow(2.718, -delta / temp)):
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best = current[:]
                    if best_conflicts == 0:
                        return best
            else:
                # Revert
                current[idx] = old_val
                
            temp *= cooling
            
        return best if best_conflicts == 0 else None

    # Try sizes from 20 down to 4
    # We allocate more time to higher sizes as they are more valuable.
    # But if 20 is impossible, we need to drop quickly.
    # Let's give 10s to 20, 10s to 19, etc.
    # Total budget ~140s.
    
    # Since the optimal is 20, we should try hard for 20.
    # But if it's infeasible with seeds, we need to find the max feasible.
    
    # Let's try 20 first with a generous time budget.
    # If it fails, try 19, etc.
    
    # However, random restarts within local_search help.
    # Let's run multiple restarts for each size.
    
    for k in range(20, 3, -1):
        # Time budget for this size
        # We have ~140s total.
        # If we spend 20s on 20 and fail, we have 120s left.
        # Let's try 3 restarts of 5s each for each size?
        # Or just one long run with restarts inside?
        
        found = None
        restarts = 0
        max_restarts = 5
        time_per_restart = 2.0
        
        while restarts < max_restarts and time.time() < deadline - 5:
            res = local_search(k, time_limit=time_per_restart)
            if res is not None:
                found = res
                break
            restarts += 1
        
        if found is not None:
            return sorted(found)
            
    # Fallback: Return the seeds if nothing else works (should not happen)
    return seeds

# EVOLVE-BLOCK-END
