import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses bitmask-accelerated local search with multiple random starts.
    """
    m = 205
    deadline = time.time() + 580.0  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitmasks
    # adj_mask[i] is a bitmask where bit j is 1 if (i-j) is in forbidden
    # Actually, it's easier to define conflict_mask[i] as bitmask of nodes conflicting with i
    # If i and j conflict, then j is in conflict_mask[i] and i is in conflict_mask[j]
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_valid_set(selected_mask):
        # Check if any two elements in selected_mask conflict
        # For each i in selected, check if (selected & conflict_mask[i]) has any bits other than i
        # But since conflict_mask[i] doesn't include i, we just check if selected & conflict_mask[i] is non-zero
        # However, we need to ensure we don't count self. conflict_mask[i] does not have bit i set.
        # So if (selected & conflict_mask[i]) != 0, it means there is a conflict.
        # We must be careful: if we iterate through set bits, we might double count.
        # A valid set S is independent if for all i in S, (S & conflict_mask[i]) == 0 is NOT quite right because 
        # if j in S and j conflicts with i, then j is in conflict_mask[i].
        # So we check: for all i in S, (S & conflict_mask[i]) must be 0? 
        # No, if i and j conflict, j is in conflict_mask[i]. So S & conflict_mask[i] will contain j.
        # Thus, for an independent set, for every i in S, (S & conflict_mask[i]) must be 0? 
        # Wait, if S={i, j} and they conflict, then j is in conflict_mask[i]. 
        # So S & conflict_mask[i] has bit j set. 
        # So yes, if S is independent, for all i in S, (S & conflict_mask[i]) == 0.
        
        # Optimization: We can just check if any pair conflicts.
        # But checking all pairs is O(N^2).
        # The bitmask check is O(N * (bit operations)).
        # Let's stick to the pairwise check for validity verification at the end.
        # During search, we maintain validity.
        pass

    def get_conflicts(selected_mask, candidate):
        """
        Returns True if candidate conflicts with any node in selected_mask.
        """
        # candidate is an integer index 0..204
        # conflict_mask[candidate] has bits set for all nodes conflicting with candidate.
        # If any of those bits are also set in selected_mask, there is a conflict.
        return (selected_mask & conflict_mask[candidate]) != 0

    def greedy_random(seed):
        """
        Generate a random greedy solution.
        """
        random.seed(seed)
        selected_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for c in candidates:
            if not get_conflicts(selected_mask, c):
                selected_mask |= (1 << c)
                
        return selected_mask

    def local_search(start_mask):
        """
        Improve a solution using 1-add and 1-swap moves.
        Returns the best mask found.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        no_improvement_count = 0
        max_iterations = 10000 # Prevent infinite loops in case of oscillation
        
        while no_improvement_count < 50 and max_iterations > 0 and time.time() < deadline:
            max_iterations -= 1
            improved = False
            
            # 1. Try 1-Add: Add a node that doesn't conflict
            # Find all candidates not in current_mask that don't conflict
            # We can iterate through 0..204
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if not get_conflicts(current_mask, c):
                        # Add c
                        new_mask = current_mask | (1 << c)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break # Restart search after improvement
            
            if improved:
                no_improvement_count = 0
                continue
            
            # 2. Try 1-Swap: Remove one node, add one or two
            # To save time, we only do 1-swap if 1-add failed.
            # A swap (remove i, add j) is valid if j doesn't conflict with (current_mask \ {i}).
            # This allows us to trade a "blocking" node for a "better" one, potentially enabling more adds later.
            # However, pure 1-swap (size same) might not increase size immediately.
            # A 2-swap (remove 1, add 2) increases size by 1.
            
            # Let's try to find a pair (i, j) such that removing i and adding j results in a valid set.
            # This is O(N^2) per iteration. N=205, so ~40k checks.
            # We can optimize by precomputing compatible sets.
            
            # Actually, let's try a simpler heuristic:
            # Identify "bad" nodes that block many additions.
            # For each i in current_mask, count how many candidates are blocked by i.
            # If we remove i, we might free up candidates.
            
            # Let's just do a random restart of local search if stuck?
            # Or try specific swaps.
            
            # Efficient Swap:
            # For each i in current_mask:
            #   For each j not in current_mask:
            #     Check if j conflicts with (current_mask without i).
            #     If not, perform swap.
            
            # To make it faster, we can precompute for each i, the mask of nodes that ONLY conflict with i 
            # (among those in current_mask). No, that's complex.
            
            # Let's just do the O(N^2) check but break early on first improvement.
            current_indices = [i for i in range(m) if current_mask & (1 << i)]
            
            swapped = False
            for i in current_indices:
                mask_without_i = current_mask & ~(1 << i)
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    # Check if j conflicts with mask_without_i
                    if not (mask_without_i & conflict_mask[j]):
                        # Valid swap
                        new_mask = mask_without_i | (1 << j)
                        current_mask = new_mask
                        # Size remains same, but structure changed
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                no_improvement_count = 0
                # Don't update best_size/size as it's same, but structure changed
                # We continue to try to add more
            else:
                no_improvement_count += 1
                
        return best_mask, best_size

    # Main Search Loop
    best_global_mask = 0
    best_global_size = 0
    
    # Start with a few deterministic/random seeds
    num_starts = 50
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Use different seeds
        seed = i * 12345 + 67890
        init_mask = greedy_random(seed)
        
        # Local search
        improved_mask, improved_size = local_search(init_mask)
        
        if improved_size > best_global_size:
            best_global_size = improved_size
            best_global_mask = improved_mask
            
        # If we found a very large set, we can stop early? 
        # Max possible size is roughly m / (avg_degree+1). 
        # Squares mod 205: 205 = 5 * 41.
        # Number of squares is roughly m/2. Degree is roughly 100.
        # Max independent set is roughly 205 / 101 ~ 2. That's too low.
        # Wait, the graph is not regular in a simple way.
        # Let's check the size of forbidden set.
        # Squares mod 5: {1, 4} (2 non-zero)
        # Squares mod 41: 20 non-zero.
        # By CRT, number of squares mod 205 is 2 * 20 = 40? 
        # No. x^2 mod 5 in {1,4}. x^2 mod 41 in 20 residues.
        # Total non-zero squares = 40.
        # So degree is 40.
        # Max independent set size >= 205 / (40+1) ~ 5.
        # Actually, the seed score 1.0 corresponds to some size.
        # The knowledge base said 1.333 was achieved.
        # If seed is 1.0, and 1.333 is better, the sizes are likely 9 and 12? 
        # Or 10 and 13?
        # Let's assume we are looking for size > 10.

    # Convert best mask to array
    result = []
    for i in range(m):
        if best_global_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
