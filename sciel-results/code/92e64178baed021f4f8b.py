import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute all 3-term APs and 4-term APs as numpy arrays
    # 3-AP: (a, a+d, a+2d) where 1 <= a, a+2d <= n
    ap3_list = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            c = a + 2 * d
            if c <= n:
                ap3_list.append((a - 1, a + d - 1, c - 1))  # 0-indexed
    
    ap3 = np.array(ap3_list, dtype=np.int32) if ap3_list else np.empty((0, 3), dtype=np.int32)
    num_3aps = len(ap3)
    
    # 4-AP: (a, a+d, a+2d, a+3d) where 1 <= a, a+3d <= n
    ap4_list = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            e = a + 3 * d
            if e <= n:
                ap4_list.append((a - 1, a + d - 1, a + 2 * d - 1, e - 1))  # 0-indexed
    
    ap4 = np.array(ap4_list, dtype=np.int32) if ap4_list else np.empty((0, 4), dtype=np.int32)
    num_4aps = len(ap4)
    
    # Precompute for each element x, the list of 3-AP indices that contain x
    # And for fast 4-AP checking
    # element_ap3[x] = list of AP indices in ap3 that include element x
    element_ap3 = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(ap3):
        element_ap3[a].append(i)
        element_ap3[b].append(i)
        element_ap3[c].append(i)
    
    # element_ap4[x] = list of 4-AP indices in ap4 that include element x
    element_ap4 = [[] for _ in range(n)]
    for i, (a, b, c, e) in enumerate(ap4):
        element_ap4[a].append(i)
        element_ap4[b].append(i)
        element_ap4[c].append(i)
        element_ap4[e].append(i)
    
    def count_3aps_fast(mask):
        """Count 3-term APs in the set represented by mask using vectorized numpy."""
        if num_3aps == 0:
            return 0
        # mask is a numpy bool array of length n
        # For each AP (a, b, c), check if all three are in mask
        a_idx, b_idx, c_idx = ap3[:, 0], ap3[:, 1], ap3[:, 2]
        # Vectorized check: mask[a] & mask[b] & mask[c]
        valid = mask[a_idx] & mask[b_idx] & mask[c_idx]
        return int(np.sum(valid))
    
    def has_4ap_fast(mask):
        """Check if the set represented by mask contains a 4-term AP using vectorized numpy."""
        if num_4aps == 0:
            return False
        a_idx, b_idx, c_idx, e_idx = ap4[:, 0], ap4[:, 1], ap4[:, 2], ap4[:, 3]
        # Vectorized check: any 4-AP fully contained in mask
        valid = mask[a_idx] & mask[b_idx] & mask[c_idx] & mask[e_idx]
        return bool(np.any(valid))
    
    def get_potential_3aps(x, mask):
        """Get the number of new 3-APs that would be formed if we add element x."""
        if x in mask:
            return 0
        count = 0
        for ap_idx in element_ap3[x]:
            a, b, c = ap3[ap_idx]
            # Check if the other two elements are already in mask
            other = [a, b, c]
            other.remove(x)
            if mask[other[0]] and mask[other[1]]:
                count += 1
        return count
    
    def get_4ap_conflicts(x, mask):
        """Check if adding x would create any 4-AP."""
        for ap_idx in element_ap4[x]:
            a, b, c, e = ap4[ap_idx]
            others = [a, b, c, e]
            others.remove(x)
            # Check if all other 3 are in mask
            if all(mask[y] for y in others):
                return True
        return False
    
    # Greedy construction with density bias
    # Start with empty set
    mask = np.zeros(n, dtype=bool)
    
    # Order elements by their total potential 3-AP participation (static estimate)
    # Element x participates in len(element_ap3[x]) 3-APs
    potentials = np.array([len(element_ap3[x]) for x in range(n)])
    order = np.argsort(-potentials)  # Descending order of potential
    
    for x in order:
        if mask[x]:
            continue
        # Check if adding x creates a 4-AP
        if get_4ap_conflicts(x, mask):
            continue
        # Add x
        mask[x] = True
    
    best_mask = mask.copy()
    best_score = count_3aps_fast(best_mask)
    
    # Local search: steepest ascent with multiple restarts
    def local_search(mask, max_iters=10000):
        """Perform steepest ascent local search."""
        current_mask = mask.copy()
        current_score = count_3aps_fast(current_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            best_gain = 0
            best_move = None
            
            # Try adding each element not in set
            for x in range(n):
                if current_mask[x]:
                    continue
                if get_4ap_conflicts(x, current_mask):
                    continue
                gain = get_potential_3aps(x, current_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', x)
            
            # Try removing each element in set (only if it frees up capacity for more 3-APs)
            # Actually, removal never increases 3-AP count directly, but might enable swaps
            # For now, just try adds
            
            if best_move is None:
                break
            
            move_type, x = best_move
            if move_type == 'add':
                current_mask[x] = True
                current_score += best_gain
        
        return current_mask, current_score
    
    # Multiple restarts with perturbations
    best_overall_mask = best_mask.copy()
    best_overall_score = best_score
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the best solution: remove some random elements
        perturbed = best_overall_mask.copy()
        elements_in_set = np.where(perturbed)[0]
        if len(elements_in_set) > 0:
            # Remove 10-20% of elements randomly
            num_remove = max(1, len(elements_in_set) // 10)
            to_remove = np.random.choice(elements_in_set, size=min(num_remove, len(elements_in_set)), replace=False)
            perturbed[to_remove] = False
        
        # Run local search from perturbed state
        local_mask, local_score = local_search(perturbed, max_iters=5000)
        
        if local_score > best_overall_score:
            best_overall_mask = local_mask
            best_overall_score = local_score
    
    # Final polish: try all possible single-element additions
    if time.time() < deadline - 5:
        for x in range(n):
            if best_overall_mask[x]:
                continue
            if get_4ap_conflicts(x, best_overall_mask):
                continue
            new_mask = best_overall_mask.copy()
            new_mask[x] = True
            new_score = count_3aps_fast(new_mask)
            if new_score > best_overall_score:
                best_overall_mask = new_mask
                best_overall_score = new_score
    
    # Convert mask to sorted list of 1-indexed elements
    result = sorted([int(x + 1) for x in np.where(best_overall_mask)[0]])
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
