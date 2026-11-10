import time
import random
import sys

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums 
    of different cardinalities are disjoint.
    
    Strategy: Global conflict-driven removal.
    1. Start with a random large subset of candidates.
    2. Compute all subset sums indexed by cardinality using DP (bitset representation for speed).
    3. Identify cross-cardinality collisions (sums appearing in S_k and S_j for k != j).
    4. Remove the element that is involved in the most collisions (or a heuristic proxy).
    5. Repeat until no collisions remain or time limit is reached.
    6. Try to add back elements if possible? No, removal is safer and monotonic.
       Actually, we can try to start with a slightly smaller set and add, but removal from a large set is a known good strategy for this specific problem type (packing).
       
    Wait, the "inspiration" mentioned "Global conflict-driven removal with bitset DP".
    Let's implement that.
    
    Bitset DP for subset sums:
    T[k] is a bitset where bit s is set if sum s is achievable with k elements.
    Max sum = 60 * 256 = 15360. Bitset size ~ 2000 bytes. Very fast.
    
    Collision counting:
    For each pair (k, j) with k < j, find intersection of T[k] and T[j].
    If non-empty, we have a conflict.
    Which element to remove?
    We can approximate the "blame" for a collision sum s in T[k] ∩ T[j] by looking at which elements are likely to form that sum.
    However, a simpler heuristic:
    If we have a conflict, remove the element that maximizes the "spread" or something?
    Or, simpler: Just remove a random element involved in the largest sum range?
    
    Actually, a very effective heuristic for this specific problem (cardinality-separated subset sums) is that the sums S_k are roughly distributed in [k*min, k*max].
    If the sets are "tight", they don't overlap.
    Overlap happens when the ranges [min_sum_k, max_sum_k] and [min_sum_j, max_sum_j] overlap AND there are actual sums in the intersection.
    
    Let's stick to the collision count.
    To find which element to remove:
    We can compute for each element x, the number of collisions that would be resolved if x were removed? That's expensive (O(N) recomputations).
    
    Alternative:
    Start with a random set of size ~50.
    While conflicts exist:
        Find a specific conflict (s, k, j).
        We know s is in S_k and S_j.
        This means there is a subset A of size k summing to s, and B of size j summing to s.
        If we remove an element from A, the sum s might disappear from S_k (if it was unique or if all subsets summing to s used that element).
        If we remove an element from B, same for S_j.
        
        A good heuristic: Remove the element that appears in the most "critical" subsets.
        But we don't track subsets, just sums.
        
        Let's try a different structural approach based on the "inspiration" of linear inequalities or global optimization.
        
        Actually, looking at the scores:
        Current: 1.25
        Inspiration: 3.875 (which is likely size 15.5? No, score is value/seed. If seed is 12.5? No, seed program score is 1.0 usually.
        Wait, "search score is value/seed".
        If current score is 1.25, and seed is 1.0, then value = 1.25 * seed_value?
        Usually "value" is the raw number of integers.
        If seed program returns 12, score 1.0.
        Current returns 15, score 1.25.
        Inspiration returns 46? 46/12 = 3.83. Close to 3.875.
        So the target is to get as close to 46-50 as possible.
        
        The bitset DP removal strategy is promising.
        
        Algorithm:
        1. Initialize with a random set of size N_start (e.g., 50) from 1..256.
        2. Compute T[k] bitsets.
        3. Check for conflicts.
        4. If no conflicts, try to add more elements (randomly or smallest available) and re-check.
        5. If conflicts, identify the "most responsible" element.
           Heuristic for responsibility:
           For each conflict sum s in S_k ∩ S_j:
           We don't know which elements formed it.
           However, elements that are "large" tend to push sums to higher cardinalities?
           
           Let's try a simpler global heuristic:
           Sort elements.
           If we have a conflict between S_k and S_j (k<j), it implies the sums are "too close".
           Usually, this happens if the elements are too small or too dense.
           Removing the LARGEST element might separate the upper bounds?
           Removing the SMALLEST element might separate the lower bounds?
           
           Let's try removing the element that is part of the most "tight" constraints.
           
           Actually, a known good method for this specific problem (distinct subset sums by cardinality) is to use a "Sidon-like" or "Bose-Chowla" type construction, but here we just need cardinality separation.
           
           Let's implement the removal loop with a heuristic:
           Compute the total number of colliding sums.
           For each element x in the set:
               Temporarily remove x.
               Compute new conflicts? Too slow if done for every x every step.
               
           Better heuristic:
           If S_k and S_j collide, it's often because the range of sums is wide.
           Widening the gap between elements helps.
           So, maybe we should start with a set that is already well-separated?
           E.g., powers of 2? 1, 2, 4, 8... 2^7=128, 2^8=256. Only 8 elements.
           We need ~46.
           
           How about arithmetic progression? 1, 2, ..., 46.
           S_k range: [k(k+1)/2, k(2*46-k+1)/2].
           S_1: [1, 46]
           S_2: [3, 89]
           Overlap! 3..46.
           So simple AP fails.
           
           We need to "stretch" the set.
           
           Let's go back to the removal strategy.
           Start with a random set of size 50.
           While conflicts:
               Pick the pair (k, j) with the most collisions.
               Among the elements, pick the one that minimizes the "variance" of the sums?
               
           Let's try a specific heuristic:
           If S_k and S_j collide, the sums are "crowded".
           Removing an element that is frequently used in forming sums in the intersection range would help.
           Since we don't track usage, let's approximate:
           Elements closer to the "average" sum contribution are more central.
           
           Alternative:
           Use a local search on the set of elements.
           Start with a random set.
           While time remains:
               If valid, try to add a random element. If invalid, try to swap an element with a random other element to reduce conflicts.
               
           Let's implement a Swap-based local search.
           1. Start with a random set of size 40.
           2. Compute conflicts.
           3. If 0 conflicts, try to add.
           4. If > 0 conflicts:
              - Pick a random element x in the set.
              - Pick a random element y not in the set.
              - Swap x and y.
              - If new conflicts < old conflicts (or equal and size same), keep swap.
              - If new conflicts == 0, try to add more.
              
           To speed up conflict computation, use bitsets.
           
           Let's refine the conflict count.
           Total conflicts = sum over k<j of |S_k ∩ S_j|.
           
           This is a structural change from "backtracking addition" to "local search with swaps".
    """
    
    deadline = time.time() + 55.0
    
    # Bitset utility
    # Max sum = 60 * 256 = 15360.
    # We need bitsets up to that size.
    # Python ints are arbitrary precision, so we can use them as bitsets.
    
    def compute_sums(elements):
        """
        Returns a list of bitsets T where T[k] has bit s set if sum s is achievable with k elements.
        """
        n = len(elements)
        T = [0] * (n + 1)
        T[0] = 1  # Bit 0 is set
        
        for x in elements:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                if T[k-1]:
                    T[k] |= (T[k-1] << x)
        
        return T
    
    def count_conflicts(T):
        """
        Count total number of colliding sums between different cardinalities.
        """
        n = len(T) - 1
        conflicts = 0
        for k in range(n):
            for j in range(k+1, n+1):
                if T[k] and T[j]:
                    # Count bits in intersection
                    inter = T[k] & T[j]
                    # popcount
                    conflicts += bin(inter).count('1')
        return conflicts
    
    def is_valid(T):
        return count_conflicts(T) == 0
    
    # Initial set
    # Start with a random set of size 45
    candidates = list(range(1, 257))
    current_set = random.sample(candidates, 45)
    
    best_set = current_set[:]
    best_conflicts = count_conflicts(compute_sums(current_set))
    
    # If we start valid, great. If not, we need to fix it.
    # Usually random 45 will have conflicts.
    
    # Local Search
    # We will perform swaps to reduce conflicts.
    
    T = compute_sums(current_set)
    conflicts = count_conflicts(T)
    
    if conflicts == 0:
        best_set = current_set[:]
        best_conflicts = 0
    
    # We want to minimize conflicts.
    # If conflicts become 0, we try to add elements.
    
    used = set(current_set)
    
    # Main loop
    while time.time() < deadline:
        if conflicts == 0:
            # Try to add elements
            added = False
            # Try adding smallest available? Or random?
            # Smallest tends to pack tighter, might cause conflicts.
            # Largest tends to spread out.
            # Let's try adding random elements.
            
            available = [x for x in candidates if x not in used]
            random.shuffle(available)
            
            for x in available:
                if time.time() > deadline:
                    break
                if len(current_set) >= 60:
                    break
                
                # Test addition
                test_set = current_set + [x]
                test_T = compute_sums(test_set)
                test_conflicts = count_conflicts(test_T)
                
                if test_conflicts == 0:
                    # Accept
                    current_set.append(x)
                    used.add(x)
                    T = test_T
                    conflicts = 0
                    added = True
                    
                    # Update best
                    if len(current_set) > len(best_set):
                        best_set = current_set[:]
                        best_conflicts = 0
                    # Continue trying to add more?
                    # Yes, loop again.
                    break
            if added:
                continue
            else:
                # Couldn't add any, try to swap to create space?
                # Or just stop?
                # Let's try swaps even if valid, maybe we can swap to a "better" configuration that allows more additions later?
                # Unlikely. If valid, it's valid.
                # But maybe a different valid set allows more additions.
                # Let's do a few random swaps among valid sets to see if we can find one that allows more additions.
                # This is complex. Let's just try swaps if we are stuck (can't add).
                pass
        
        # Perform a swap
        # Pick random x in set, y not in set.
        if len(current_set) == 0:
            break
            
        x_idx = random.randint(0, len(current_set) - 1)
        x = current_set[x_idx]
        
        available = [y for y in candidates if y not in used]
        if not available:
            break
            
        y = random.choice(available)
        
        # Swap
        test_set = current_set[:]
        test_set[x_idx] = y
        
        test_T = compute_sums(test_set)
        test_conflicts = count_conflicts(test_T)
        
        # Accept if better, or if equal and random?
        # Hill climbing: accept if test_conflicts < conflicts
        # Or accept if test_conflicts == conflicts and random() < 0.1 (plateau search)
        
        if test_conflicts < conflicts or (test_conflicts == conflicts and random.random() < 0.1):
            # Apply swap
            current_set[x_idx] = y
            used.remove(x)
            used.add(y)
            T = test_T
            conflicts = test_conflicts
            
            if conflicts == 0 and len(current_set) > len(best_set):
                best_set = current_set[:]
                best_conflicts = 0
            
            # If we just became valid, we might be able to add more in the next iteration
            # If we reduced conflicts, good.
            
            # Additional: If we are valid, try to add immediately?
            # The loop structure above handles adding if conflicts==0 at the start of the loop.
            # So if we just became valid, the next iteration will try to add.
        else:
            # Rejection
            pass
    
    # Final check: ensure best_set is valid
    # It should be, but just in case.
    # If best_set is empty, return a known valid small set?
    # The problem allows empty set? "The empty subset is included."
    # "Return a list of ints."
    # If we fail, we should return the best valid set found.
    
    # Verify best_set
    if best_set:
        best_T = compute_sums(best_set)
        if count_conflicts(best_T) > 0:
            # Fallback: find a valid subset of best_set?
            # Or just return empty?
            # Better to return a small known valid set.
            # 1 is always valid.
            # 1, 3?
            # S0={0}, S1={1,3}, S2={4}. Disjoint.
            # Let's construct a small valid set as fallback.
            fallback = [1, 3, 7, 15, 31, 63, 127]
            # Powers of 2 minus 1?
            # 1, 3, 7...
            # S1: 1,3,7...
            # S2: 1+3=4, 1+7=8...
            # 4 is not in S1.
            # This set is valid.
            best_set = fallback
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
