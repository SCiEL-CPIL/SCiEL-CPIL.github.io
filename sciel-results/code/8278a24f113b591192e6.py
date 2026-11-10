import time
import random
from itertools import product

# Precompute all 4-APs in [1, 512]
N = 512
APs = []
for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        APs.append((a, a + d, a + 2 * d, a + 3 * d))

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    # 1. Initial Greedy Construction
    selected = set()
    for x in range(1, N + 1):
        # Check if x forms a 4-AP with 3 existing elements
        # Since we add in increasing order, x is the largest.
        # We need to check if there exist a, b, c in selected such that a, b, c, x is a 4-AP.
        # This is equivalent to checking if there is a 4-AP ending at x.
        valid = True
        # Max d such that x - 3d >= 1
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if all(x - j * d in selected for j in (1, 2, 3)):
                valid = False
                break
        if valid:
            selected.add(x)
    
    initial_size = len(selected)
    
    # 2. Local Search / Hill Climbing
    # Precompute for each x, the list of 4-APs it participates in
    # This speeds up conflict detection
    x_to_aps = {x: [] for x in range(1, N + 1)}
    for ap in APs:
        for x in ap:
            x_to_aps[x].append(ap)
            
    best_set = set(selected)
    best_size = initial_size
    
    # Shuffle rejected numbers for randomization
    rejected = [x for x in range(1, N + 1) if x not in selected]
    random.shuffle(rejected)
    
    # We will iterate through rejected numbers and try to add them
    # If we can't add directly, we try swaps.
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Re-shuffle rejected list occasionally? Or just iterate.
        # Let's iterate through all non-selected numbers.
        # To be efficient, we maintain a list of candidates.
        candidates = [x for x in range(1, N + 1) if x not in best_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if (time.time() - start_time) > timeout:
                break
                
            # Check if x can be added directly
            # Find all APs containing x
            conflicts = []
            for ap in x_to_aps[x]:
                # Check if the other 3 elements are in best_set
                others = [y for y in ap if y != x]
                if all(y in best_set for y in others):
                    conflicts.append(others)
            
            if not conflicts:
                # Can add x
                best_set.add(x)
                best_size += 1
                improved = True
                # Remove x from candidates? Not necessary, loop will end or x is now in set.
                # Actually, we should probably restart or just continue.
                # Continuing is fine, but x is now in best_set, so future checks for x will fail.
                # But we are iterating a static list 'candidates'.
                # If we add x, we should probably break and restart the candidate loop?
                # Or just mark it.
                # Let's just break and restart the while loop to pick up new candidates.
                # This might be slow.
                # Alternative: just continue. The added x might enable other swaps?
                # No, adding x only creates more conflicts, it doesn't remove them.
                # So if we add x, we can continue checking other candidates.
                # However, the set 'candidates' is static.
                # If we add x, we should probably remove it from consideration for swaps in this pass?
                # It's already added, so it won't be swapped out in this simple logic.
                pass
            else:
                # Try to swap: remove one element from a conflict AP and add x
                # We want to find a swap that increases size or maintains size.
                # Since we are removing 1 and adding 1, size stays same.
                # We only accept if it leads to a better state later?
                # For now, let's only accept if it increases size.
                # But we can't increase size by swapping 1-for-1.
                # So this local search only grows the set.
                # This is effectively a greedy repair.
                # To improve, we need to allow 2-for-3 swaps or similar?
                # Or just accept equal size swaps with probability to escape local minima.
                
                # Let's try to find a swap that allows adding x.
                # If we remove y from best_set, does x become addable?
                # Yes, if y was part of all conflicts.
                # But removing y might break other APs? No, removing elements never creates APs.
                # So if we remove y, the set is still valid.
                # Then we add x.
                # New size = old size.
                # We should only do this if it helps future additions.
                # This is complex to evaluate.
                
                # Simpler: Just try to add x by removing ONE element from a conflict.
                # If we do this, we don't gain size.
                # So this specific loop only adds elements that don't conflict.
                # This is just a more thorough greedy.
                
                # To actually improve, we need to consider removing 2 elements to add 3?
                # Or just use Simulated Annealing on the set of selected elements.
                
                # Let's implement a simple SA step:
                # Pick a random element in best_set, remove it.
                # Pick a random element not in best_set, add it.
                # If size increases, accept.
                # If size same, accept with prob exp(-1/T).
                # If size decreases, reject (or accept with very low prob).
                
                # This is getting complex to implement correctly within the time limit.
                # Let's stick to the "Add if possible" logic but with random restarts of the greedy.
                
                pass
                
        # If no improvement in this pass, break?
        # Or try random restarts?
        
    # If time remains, try a few random restarts of the greedy with different orders?
    # The greedy order 1..N is standard.
    # What if we use a randomized greedy?
    
    # Let's add a simple random restart mechanism if time permits.
    if (time.time() - start_time) < timeout - 5:
        for _ in range(10):
            if (time.time() - start_time) > timeout - 5:
                break
            # Randomized Greedy
            rand_set = set()
            # Shuffle numbers? No, order matters for APs.
            # Instead, let's try the "Reverse Greedy"
            for x in range(N, 0, -1):
                valid = True
                # Check if x is the smallest in a 4-AP with 3 larger elements?
                # No, if we build backwards, x is the smallest.
                # We need to check if there are y, z, w in rand_set such that x, y, z, w is a 4-AP.
                # This is equivalent to checking if there is a 4-AP starting at x.
                max_d = (N - x) // 3
                for d in range(1, max_d + 1):
                    if all(x + j * d in rand_set for j in (1, 2, 3)):
                        valid = False
                        break
                if valid:
                    rand_set.add(x)
                    
            if len(rand_set) > best_size:
                best_set = rand_set
                best_size = len(rand_set)
                
    return sorted(best_set)
