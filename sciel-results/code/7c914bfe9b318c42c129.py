"""Bitmask-based local search for 4-AP-free subsets of [1,256] maximizing 3-APs."""
import time
import random
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Precompute 4-AP constraints: for each element i (0-indexed), 
    # which triples (j, k, l) with j<i<k<l and i as first, second, third, or fourth term
    # Actually, we need: for any 4-term AP a, a+d, a+2d, a+3d, none of the 4 can all be in set.
    # For local search, we need to check if adding/removing an element creates/destroys 4-APs.
    
    # Represent set as a 256-bit integer (using Python's arbitrary precision ints)
    # Or use a boolean numpy array. Let's use a Python int as a bitmask for speed.
    
    # Precompute all 4-APs in [1,256]
    # A 4-AP is (a, a+d, a+2d, a+3d) where a + 3d <= 256
    # Store for each position, which 4-APs it participates in
    
    # For efficient checking: when we consider adding element x, we need to verify
    # no 4-AP containing x has all other 3 elements already in the set.
    # When removing element x, we just remove it (no new 4-APs created by removal).
    
    # Precompute: for each x, list of (other three elements) such that {x} ∪ others forms a 4-AP
    four_ap_partners = [[] for _ in range(N)]
    for a in range(N):  # 0-indexed, a = position-1
        for d in range(1, (N - a) // 3 + 1):
            if a + 3*d >= N:
                break
            b = a + d
            c = a + 2*d
            e = a + 3*d
            # The 4-AP is (a, b, c, e)
            four_ap_partners[a].append((b, c, e))
            four_ap_partners[b].append((a, c, e))
            four_ap_partners[c].append((a, b, e))
            four_ap_partners[e].append((a, b, c))
    
    # Precompute 3-APs for scoring
    # A 3-AP is (a, a+d, a+2d) with positive d
    # Count = number of pairs (a, a+2d) in set such that a+d is also in set
    # For scoring: score = sum over all 3-APs of indicator(all three in set)
    
    # For incremental scoring when adding/removing element x:
    # Adding x: gain = number of 3-APs containing x where the other two are in set
    # Removing x: loss = number of 3-APs containing x where the other two are in set
    
    # Precompute: for each x, list of (y, z) such that {x, y, z} forms a 3-AP
    three_ap_partners = [[] for _ in range(N)]
    for a in range(N):
        for d in range(1, (N - a) // 2 + 1):
            if a + 2*d >= N:
                break
            b = a + d
            c = a + 2*d
            three_ap_partners[a].append((b, c))
            three_ap_partners[b].append((a, c))
            three_ap_partners[c].append((a, b))
    
    def is_valid_add(mask, x):
        """Check if adding element x to set (represented by mask) creates no 4-AP."""
        for (y, z, w) in four_ap_partners[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True
    
    def count_3aps(mask):
        """Count number of 3-APs in the set."""
        count = 0
        for x in range(N):
            if (mask >> x) & 1:
                for (y, z) in three_ap_partners[x]:
                    if (mask >> y) & 1 and (mask >> z) & 1:
                        # Avoid double counting: only count if x is the smallest
                        if x < y and x < z:
                            count += 1
        return count
    
    def three_ap_gain(mask, x):
        """Number of 3-APs gained by adding x (where other two are in set)."""
        gain = 0
        for (y, z) in three_ap_partners[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                # Only count if this 3-AP is newly formed, i.e., x wasn't in it before
                # Since x is being added, all such 3-APs are new
                gain += 1
        return gain
    
    def three_ap_loss(mask, x):
        """Number of 3-APs lost by removing x (where other two are in set)."""
        loss = 0
        for (y, z) in three_ap_partners[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                loss += 1
        return loss
    
    # Greedy initialization
    def greedy_init():
        mask = 0
        # Try multiple orderings
        order = list(range(N))
        random.shuffle(order)
        for x in order:
            if is_valid_add(mask, x):
                mask |= (1 << x)
        return mask
    
    # Local search with add/remove/swap moves
    def local_search(mask, max_iters=50000):
        current_score = count_3aps(mask)
        best_mask = mask
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding an element
            for x in range(N):
                if not ((mask >> x) & 1):
                    if is_valid_add(mask, x):
                        new_mask = mask | (1 << x)
                        gain = three_ap_gain(mask, x)
                        new_score = current_score + gain
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask
                            mask = new_mask
                            current_score = new_score
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try removing an element (if it helps enable additions later)
            # Actually, for maximizing 3-APs with 4-AP constraint, removal is rarely helpful
            # unless it enables more additions. Let's try swaps.
            
            # Try swap: remove x, add y
            # This is expensive. Let's just try removals that might enable better additions.
            # For now, let's just do a simple hill climbing with adds only, 
            # and occasionally try a random restart.
            
            if not improved:
                # Local optimum reached with adds only
                # Try a random perturbation: remove a random element and re-add greedily
                if current_score > 0:
                    # Pick a random element to remove
                    in_set = [x for x in range(N) if (mask >> x) & 1]
                    if in_set:
                        x_remove = random.choice(in_set)
                        new_mask = mask & ~(1 << x_remove)
                        # Try to add elements back greedily
                        order = list(range(N))
                        random.shuffle(order)
                        for x in order:
                            if not ((new_mask >> x) & 1):
                                if is_valid_add(new_mask, x):
                                    new_mask |= (1 << x)
                        new_score = count_3aps(new_mask)
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask
                            mask = new_mask
                            current_score = new_score
                            improved = True
            
            if not improved and it > 1000:
                # Stuck, break
                break
        
        return best_mask, best_score
    
    # Run multiple restarts
    best_mask_overall = 0
    best_score_overall = 0
    
    # First, a deterministic greedy from low to high
    mask = 0
    for x in range(N):
        if is_valid_add(mask, x):
            mask |= (1 << x)
    score = count_3aps(mask)
    if score > best_score_overall:
        best_score_overall = score
        best_mask_overall = mask
    
    # Then random restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        mask = greedy_init()
        mask, score = local_search(mask, max_iters=20000)
        
        if score > best_score_overall:
            best_score_overall = score
            best_mask_overall = mask
    
    # Convert mask to sorted list
    result = [x + 1 for x in range(N) if (best_mask_overall >> x) & 1]
    
    if not result:
        result = [1]
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result[:20]}...")
