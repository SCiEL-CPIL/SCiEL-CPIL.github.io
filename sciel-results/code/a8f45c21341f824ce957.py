import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def can_add(z, A_set, A_list):
        # Check if A U {z} is sum-free
        # 1. z + a = b for some a, b in A
        # 2. a + b = z for some a, b in A
        # 3. z + z = b for some b in A (covered by 1 if we allow a=z? No, a in A)
        #    Actually, condition is: no a,b,c in A' such that a+b=c.
        #    If we add z, new triples involve z.
        #    Cases:
        #    a + b = z (a,b in A)
        #    z + a = b (a,b in A) -> z = b - a
        #    a + z = b (a,b in A) -> z = b - a
        #    z + z = b (b in A) -> 2z = b
        
        # Check 1: a + b = z
        # Iterate a in A, check if z - a in A
        for a in A_list:
            if z - a in A_set:
                # If a == z-a, i.e., 2a=z, we still have a pair (a,a).
                # The condition "a,b in A" allows a=b.
                return False
        
        # Check 2: z + a = b => b - a = z
        # Iterate a in A, check if a + z in A
        for a in A_list:
            if a + z in A_set:
                return False
                
        # Check 3: z + z = b => 2z = b
        if 2 * z in A_set:
            return False
            
        return True

    def get_best_solution(S):
        S_set = set(S)
        n = len(S)
        
        # Initial candidates
        candidates = []
        
        # 1. All odds
        A1 = [x for x in S if x % 2 != 0]
        candidates.append(A1)
        
        # 2. Greedy Ascending
        A2 = []
        A2_set = set()
        for x in S:
            if can_add(x, A2_set, A2):
                A2.append(x)
                A2_set.add(x)
        candidates.append(A2)
        
        # 3. Greedy Descending
        A3 = []
        A3_set = set()
        for x in reversed(S):
            if can_add(x, A3_set, A3):
                A3.append(x)
                A3_set.add(x)
        candidates.append(A3)
        
        # 4. Upper half
        max_val = max(S)
        A4 = [x for x in S if x > max_val / 2]
        candidates.append(A4)
        
        # 5. Lower half
        min_val = min(S)
        A5 = [x for x in S if x < min_val / 2]
        candidates.append(A5)
        
        # 6. Randomized Greedy (a few times)
        for _ in range(5):
            A6 = []
            A6_set = set()
            shuffled = S[:]
            random.shuffle(shuffled)
            for x in shuffled:
                if can_add(x, A6_set, A6):
                    A6.append(x)
                    A6_set.add(x)
            candidates.append(A6)

        best_A = max(candidates, key=len)
        best_A_set = set(best_A)
        
        # Local Search / Hill Climbing
        start_time = time.time()
        deadline = start_time + 10.0 # 10s per instance to be safe
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any valid element
            added_any = False
            # We need to check all elements not in A
            # To make this faster, we can just iterate S
            for z in S:
                if z not in best_A_set:
                    if can_add(z, best_A_set, best_A):
                        best_A.append(z)
                        best_A_set.add(z)
                        added_any = True
                        improved = True
                        # Don't break, try to add more in this pass
            if added_any:
                continue # Restart loop to check for more additions
            
            # Phase 2: If no additions, try to remove one to enable more
            best_gain = -1
            best_remove = None
            
            # Calculate potential gains for removing each x
            # gain = (# of z in S\A that can be added to A\{x}) - 1
            # This is O(n^3) roughly if done naively.
            # n=240. n^3 = 13M. can_add is O(n). So 13M * 240 = 3B ops. Too slow.
            
            # Optimization:
            # We only need to check removals if we are stuck.
            # Let's limit the search space for removals or use a heuristic.
            # Heuristic: Remove element with highest "conflict degree".
            # Conflict degree of x: number of pairs (a,b) in A such that a+b=x OR x+a=b OR x+b=a.
            # Actually, if we remove x, which z's become addable?
            # z becomes addable if the only conflict preventing it was related to x.
            
            # Simpler heuristic: Try removing each x, and just check if we can add *any* new element.
            # If we can add at least 2 elements, it's a net gain.
            # But checking "how many can be added" is expensive.
            # Let's just try to find *any* removal that allows an addition.
            
            # Iterate through A
            for x in best_A:
                # Temporarily remove x
                best_A_set.remove(x)
                
                # Check if any z can be added
                added_z = None
                for z in S:
                    if z not in best_A_set:
                        if can_add(z, best_A_set, best_A):
                            added_z = z
                            break # Just find one
                
                if added_z is not None:
                    # We found a removal x that allows adding z.
                    # Is it a net gain? We removed 1, added 1. Net 0.
                    # But maybe adding z allows adding more later?
                    # Or maybe we can add MORE than 1?
                    # For now, let's just take this step if we are stuck.
                    # To ensure progress, we should only do this if we expect improvement.
                    # If we remove x and add z, size is same.
                    # This is a plateau move.
                    # Let's count how many can be added.
                    count_addable = 0
                    for z in S:
                        if z not in best_A_set:
                            if can_add(z, best_A_set, best_A):
                                count_addable += 1
                    # If count_addable > 1, we have a net gain.
                    if count_addable > 1:
                        # Perform the move
                        best_A.remove(x)
                        # Add all addable ones?
                        # Let's add them one by one to maintain validity
                        # Actually, if they are all valid against A\{x}, are they valid against each other?
                        # Not necessarily.
                        # So we should add them greedily.
                        for z in S:
                            if z not in best_A_set:
                                if can_add(z, best_A_set, best_A):
                                    best_A.append(z)
                                    best_A_set.add(z)
                        improved = True
                        break # Restart loop
                    else:
                        # Restore x
                        best_A_set.add(x)
                else:
                    # Restore x
                    best_A_set.add(x)
                    
            # If no beneficial removal found, try random restart or accept current
            if not improved:
                # Try a random perturbation?
                # Remove a random element, then try to fill up
                if time.time() < deadline - 1.0:
                    if best_A:
                        idx = random.randint(0, len(best_A)-1)
                        x = best_A.pop(idx)
                        best_A_set.remove(x)
                        # Now try to add
                        for z in S:
                            if z not in best_A_set:
                                if can_add(z, best_A_set, best_A):
                                    best_A.append(z)
                                    best_A_set.add(z)
                        # Check if size increased
                        # If not, we might have degraded. 
                        # To keep it simple, we only accept if size >= previous best?
                        # We don't track previous best easily here without extra state.
                        # Let's just rely on the hill climbing.
                        # If this perturbation leads to a dead end, we stop.
                        # To be safe, let's just stop if no strict improvement.
                        break
                        
        return best_A

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(get_best_solution(S))
        
    return results

# EVOLVE-BLOCK-END
