import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    # Precompute all 4-APs and their index positions
    # A 4-AP is a, a+d, a+2d, a+3d
    # We store for each AP: [a, a+d, a+2d, a+3d]
    # We will use these to check if adding x creates a 4-AP.
    # Actually, for the "add-and-repair" strategy, we need to know:
    # If we add x, which 4-APs are now complete?
    # A 4-AP is complete if all 4 elements are in S.
    
    # Let's represent S as a bitset for fast operations.
    # S_mask is a numpy array of uint8 or bool, size N+1.
    # But we need to find conflicts quickly.
    
    # Strategy:
    # 1. Generate all 4-APs.
    # 2. For each x in [1, N], precompute the list of 4-APs that contain x.
    #    For each such AP, store the other 3 elements.
    # 3. To check if x can be added:
    #    For each AP containing x, check if the other 3 elements are in S.
    #    If any AP is fully contained in S ∪ {x}, then adding x creates a 4-AP.
    # 4. To repair:
    #    If adding x creates conflicts, we remove the most recently added element 
    #    from each conflicting AP (or just one) to break the AP.
    
    # Precomputation
    ap_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map element x to list of (other_elements_tuple, ap_index)
    # We need to check if other_elements are in S.
    # To make this fast, we can just iterate.
    # Number of APs is roughly sum_{d=1}^{170} (512 - 3d) ≈ 170*512 - 3*170*171/2 ≈ 87000 - 43000 = 44000.
    # Each AP has 4 elements. Total element-AP incidences = 4 * 44000 = 176,000.
    # This is small enough to store in lists.
    
    element_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(ap_list):
        for x in ap:
            others = tuple(y for y in ap if y != x)
            element_aps[x].append((others, idx))
    
    def can_add(S_mask, x):
        # S_mask is a numpy bool array of size N+1
        # Check if adding x creates any 4-AP
        for others, _ in element_aps[x]:
            # Check if all others are in S_mask
            # others is a tuple of 3 ints
            # Using numpy indexing
            if S_mask[others].all():
                return False
        return True

    def find_conflict(S_mask, x):
        # Returns the index of a conflicting AP if adding x creates one, else -1
        for others, idx in element_aps[x]:
            if S_mask[others].all():
                return idx
        return -1

    # Initial construction: Greedy with base-3 (Salem-Spencer) bias
    # Elements with only 0s and 1s in base-3 are 3-AP free, hence 4-AP free.
    # This gives a good dense starting point.
    
    def get_base3_digits(n):
        digits = []
        while n > 0:
            digits.append(n % 3)
            n //= 3
        return digits

    def is_salem_spencer(n):
        return all(d in [0, 1] for d in get_base3_digits(n))

    # Order: Salem-Spencer numbers first, then others
    ss_numbers = [i for i in range(1, N + 1) if is_salem_spencer(i)]
    other_numbers = [i for i in range(1, N + 1) if not is_salem_spencer(i)]
    
    # Shuffle within groups to add variety? Or just use natural order for determinism in seed?
    # Let's use random shuffle for the "other" part to explore different structures.
    random.seed(42)
    random.shuffle(other_numbers)
    
    order = ss_numbers + other_numbers
    
    S_mask = np.zeros(N + 1, dtype=bool)
    S_set = set()
    
    for x in order:
        if can_add(S_mask, x):
            S_mask[x] = True
            S_set.add(x)
            
    best_S = set(S_set)
    best_size = len(S_set)
    
    # Local Search: Add-and-Repair
    # We maintain a list of "recently added" elements to facilitate repair.
    # If adding x creates a conflict, we identify the AP. 
    # We remove the most recently added element from that AP (excluding x, or including x if we decide to revert).
    # Actually, standard "repair" in MaxSAT-like problems:
    # 1. Try to add x.
    # 2. If conflict, remove a random element from the conflicting AP (that is not x) to break it.
    # 3. Add x.
    # 4. If this leads to a smaller set, revert. If same size or larger, keep.
    # Since we add 1 and remove 1, size stays same. 
    # To increase size, we need to find a move where adding x doesn't require removal, 
    # or where we can add x and NOT remove (i.e., no conflict), which is just the greedy step.
    
    # Better strategy: 
    # 1. Pick a random x NOT in S.
    # 2. Check if can add. If yes, add. (Size increases)
    # 3. If no, find conflict AP. Remove a random element y from AP (y != x). 
    #    Add x. (Size same, but structure changed)
    # 4. Repeat.
    # This is a random walk in the space of 4-AP-free sets.
    # We track best size.
    
    # To make it faster, we can batch operations or use a priority queue?
    # Given N=512, the set is small (< 100 elements likely).
    # can_add is O(num_aps_containing_x).
    
    rejected = [x for x in range(1, N + 1) if x not in S_set]
    
    # We will use a "simulated annealing" like approach or just random walk with restarts?
    # Let's stick to random walk with occasional restarts if stuck.
    
    current_S = set(S_set)
    current_mask = S_mask.copy()
    
    # Track history for repair? 
    # Simple repair: if x conflicts with AP (a,b,c,d), remove one of a,b,c,d (not x) randomly.
    # This guarantees the new set is 4-AP free? 
    # NO. Removing one element from the conflicting AP breaks that AP.
    # But does it break other APs? No, removing an element only destroys APs, never creates them.
    # So the new set is valid.
    
    # The only issue is: we might be swapping elements indefinitely without increasing size.
    # We need to escape local optima.
    # If we can't increase size for a while, we can try a "double move":
    # Remove 2 elements, add 1? No, that decreases size.
    # Remove 1, add 2? 
    # 1. Remove y.
    # 2. Try to add x1. If conflict, repair.
    # 3. Try to add x2. If conflict, repair.
    # This is complex.
    
    # Let's try a different structural change: 
    # Instead of simple random walk, use "Hill Climbing" with random restarts.
    # But the current program already does something similar.
    
    # Key difference: The current program's `can_add` is a Python loop over `range(1, max_d+1)`.
    # My `can_add` iterates over precomputed APs for x.
    # For x=1, APs are (1, 1+d, 1+2d, 1+3d). d max 170.
    # Number of APs containing x is roughly proportional to number of d's.
    # For x in middle, it's ~ 3 * (N/3) = N. So ~500 checks.
    # Each check is `S_mask[others].all()`.
    # This is much faster than the current program's nested loops?
    # Current: for d in 1..170: 4 checks. 170*4 = 680 checks.
    # Mine: ~500 checks.
    # Similar. But numpy `.all()` is fast.
    
    # Let's optimize the search loop.
    
    while time.time() - start_time < time_limit:
        if not rejected:
            break
            
        # Randomly pick a rejected element
        x = random.choice(rejected)
        
        if can_add(current_mask, x):
            # Can add directly
            current_mask[x] = True
            current_S.add(x)
            rejected.remove(x)
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
        else:
            # Conflict exists. Try to repair.
            # Find a conflicting AP
            conflict_idx = -1
            conflict_ap = None
            for others, idx in element_aps[x]:
                if current_mask[others].all():
                    conflict_idx = idx
                    conflict_ap = ap_list[idx]
                    break
            
            if conflict_ap is None:
                # Should not happen if can_add returned False
                continue
            
            # Remove a random element from conflict_ap (excluding x)
            # x is in conflict_ap
            candidates_to_remove = [y for y in conflict_ap if y != x]
            # Pick one randomly
            y = random.choice(candidates_to_remove)
            
            # Remove y
            current_mask[y] = False
            current_S.remove(y)
            # Add y to rejected
            rejected.append(y)
            
            # Add x
            current_mask[x] = True
            current_S.add(x)
            rejected.remove(x)
            
            # Size is unchanged. 
            # We need a mechanism to increase size.
            # This random walk might get stuck.
            # Let's add a "greedy fill" step every N steps.
            
            # Or, better: 
            # If we are stuck in a random walk (no size increase for long), 
            # we can try to find a "free" spot.
            
            # Let's just keep doing this. The random walk explores the space.
            # Occasionally, a swap might open up a spot for a 2nd element?
            # No, swap keeps size same.
            
            # To increase size, we MUST find an x that can be added without removal.
            # The random walk changes the set S, so the set of "addable" x changes.
            # So we might eventually find an addable x.
            
            # To speed up, we can also try to add multiple elements after a successful add?
            # If we add x (size +1), we can then try to add another z?
            # Let's do a small local search after each successful add.
            
    # Final greedy fill to catch any easy wins
    for x in rejected[:]:
        if time.time() - start_time > time_limit:
            break
        if can_add(current_mask, x):
            current_mask[x] = True
            current_S.add(x)
            rejected.remove(x)
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
                
    return sorted(best_S)

# EVOLVE-BLOCK-END
