import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    def count_conflicts(A):
        """
        Returns the number of pairs of triples that have the same sum.
        If A is a B_3 set, this is 0.
        """
        sums = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                for k in range(j, len(A)):
                    s = A[i] + A[j] + A[k]
                    if s in sums:
                        # We just need to count if there's a conflict, 
                        # but for annealing, counting total conflicts is better for gradient.
                        # Let's return the number of duplicate sums.
                        pass
                    else:
                        sums[s] = 1
        
        # A more efficient way to count conflicts for SA:
        # Count frequency of each sum. Sum over all sums of (freq choose 2).
        freqs = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                for k in range(j, len(A)):
                    s = A[i] + A[j] + A[k]
                    freqs[s] = freqs.get(s, 0) + 1
        
        conflicts = 0
        for f in freqs.values():
            if f > 1:
                conflicts += f * (f - 1) // 2
        return conflicts

    def is_valid(A):
        return count_conflicts(A) == 0

    def get_conflict_elements(A):
        """
        Identify which elements are involved in conflicts to target them for swapping.
        """
        sums = {}
        triples = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                for k in range(j, len(A)):
                    s = A[i] + A[j] + A[k]
                    if s in sums:
                        # Conflict found
                        # Mark i, j, k and the previous triple's indices
                        pass
                    else:
                        sums[s] = (i, j, k)
        
        # This is getting complex to track exactly which elements cause which conflicts efficiently.
        # Simpler: Just swap random elements or elements that appear in many sums.
        # For now, let's just do random swaps in SA.
        return []

    best_set = []
    best_size = 0
    best_conflicts = float('inf')

    # We know max size is around 41. Let's try to find a set of size 40, 41, 42.
    # Start with a random set of size 30 and grow?
    # Or start with size 40 and try to fix?
    
    # Let's try a "Grow and Repair" strategy.
    # Start with empty set.
    # Add elements one by one.
    # If stuck, use SA to rearrange the current set to make room?
    
    # Actually, a pure SA on the set content is robust.
    # Let's try to find a valid set of size K.
    # We will iterate K from 40 down to 1? No, up.
    # Let's try K=40 first.
    
    for target_size in [42, 41, 40, 39, 38]:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random distinct integers
        current_set = random.sample(range(1, 4097), target_size)
        current_conflicts = count_conflicts(current_set)
        
        if current_conflicts == 0:
            best_set = current_set
            best_size = target_size
            break
        
        # Simulated Annealing
        T = 100.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
                
            # Generate a new state
            new_set = current_set.copy()
            
            # Move: Swap one element in set with one outside
            idx_in = random.randint(0, len(new_set) - 1)
            # Pick a random candidate from 1..4096 not in set
            # To be efficient, we can just pick random and check
            candidates = []
            for _ in range(50): # Sample a few
                c = random.randint(1, 4096)
                if c not in new_set:
                    candidates.append(c)
                    break
            if not candidates:
                continue
            
            new_val = candidates[0]
            old_val = new_set[idx_in]
            new_set[idx_in] = new_val
            
            new_conflicts = count_conflicts(new_set)
            
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_set = new_set
                current_conflicts = new_conflicts
                
                if current_conflicts == 0:
                    best_set = current_set
                    best_size = target_size
                    break
            
            T *= cooling_rate
            
        if best_size == target_size:
            break

    # If we didn't find a valid set of the target sizes, fall back to the greedy baseline
    if not best_set:
        # Re-run the simple greedy from the original code as a fallback
        def run_greedy(order):
            selected = []
            current_sums = set()
            for x in order:
                new_sums = set()
                s = 3 * x
                if s in current_sums:
                    continue
                new_sums.add(s)
                
                conflict = False
                for a in selected:
                    s = 2 * x + a
                    if s in current_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.add(s)
                if conflict:
                    continue
                
                for a, b in combinations_with_replacement(selected, 2):
                    s = x + a + b
                    if s in current_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.add(s)
                
                if not conflict:
                    selected.append(x)
                    current_sums.update(new_sums)
            return selected

        best_set = run_greedy(range(1, 4097))
        best_size = len(best_set)

    return best_set
