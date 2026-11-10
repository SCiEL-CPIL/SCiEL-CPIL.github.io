import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    
    # 1. Precompute 3-APs and 4-APs
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    three_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            three_aps_by_v[a].append(mask)
            three_aps_by_v[b].append(mask)
            three_aps_by_v[c].append(mask)
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    # Map element v to list of 4-AP masks containing v
    four_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[a].append(mask)
            four_aps_by_v[b].append(mask)
            four_aps_by_v[c].append(mask)
            four_aps_by_v[e].append(mask)
            
    # 2. Initial Greedy Set
    # Greedy by index (like seed)
    current_mask = 0
    for x in range(1, N+1):
        bit = 1 << (x-1)
        new_mask = current_mask | bit
        # Check if new_mask has 4-AP
        # Only need to check 4-APs containing x
        valid = True
        for ap_mask in four_aps_by_v[x]:
            if (new_mask & ap_mask) == ap_mask:
                valid = False
                break
        if valid:
            current_mask = new_mask
            
    # Calculate initial score
    def get_score(mask):
        return sum(1 for ap_mask in three_aps if (mask & ap_mask) == ap_mask)
        
    current_score = get_score(current_mask)
    
    # Helper to get set elements
    def get_elements(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    # Incremental score update for swap (u -> v)
    def compute_swap_score(mask, u, v, current_score):
        u_bit = 1 << (u-1)
        v_bit = 1 << (v-1)
        new_mask = (mask & ~u_bit) | v_bit
        
        # Count 3-APs that are broken by removing u
        # A 3-AP is broken if it contains u and the other two elements are in mask
        lost = 0
        for ap_mask in three_aps_by_v[u]:
            if (ap_mask & u_bit) and (ap_mask & ~u_bit & mask) == (ap_mask & ~u_bit):
                lost += 1
        
        # Count 3-APs that are gained by adding v
        # A 3-AP is gained if it contains v and the other two elements are in new_mask
        gained = 0
        for ap_mask in three_aps_by_v[v]:
            if (ap_mask & v_bit) and (ap_mask & ~v_bit & new_mask) == (ap_mask & ~v_bit):
                gained += 1
        
        return current_score - lost + gained, new_mask
    
    # 3. Local Search
    # We will perform swaps
    # To avoid getting stuck, we can randomize the order of u and v
    
    while time.time() - start_time < DEADLINE:
        improved = False
        elements = get_elements(current_mask)
        non_elements = [i for i in range(1, N+1) if not (current_mask >> (i-1)) & 1]
        
        # Shuffle to avoid bias
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        for u in elements:
            if time.time() - start_time > DEADLINE:
                break
            u_bit = 1 << (u-1)
            for v in non_elements:
                if time.time() - start_time > DEADLINE:
                    break
                v_bit = 1 << (v-1)
                new_mask = (current_mask & ~u_bit) | v_bit
                
                # Check validity
                # Only 4-APs containing v can be new violations
                valid = True
                for ap_mask in four_aps_by_v[v]:
                    if (new_mask & ap_mask) == ap_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                    
                # Calculate score incrementally
                new_score, _ = compute_swap_score(current_mask, u, v, current_score)
                
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    # Break inner loop to restart with new mask
                    break
            if improved:
                break
                
        if not improved:
            # No improvement found in this sweep
            # Try a random kick to escape local optima
            # Pick random u in S, random v not in S
            if elements and non_elements:
                u = random.choice(elements)
                v = random.choice(non_elements)
                u_bit = 1 << (u-1)
                v_bit = 1 << (v-1)
                new_mask = (current_mask & ~u_bit) | v_bit
                # Only accept if valid, even if score drops? 
                # Or just accept if valid to explore.
                valid = True
                for ap_mask in four_aps_by_v[v]:
                    if (new_mask & ap_mask) == ap_mask:
                        valid = False
                        break
                if valid:
                    current_mask = new_mask
                    current_score = get_score(current_mask)
            else:
                break # No moves possible

    # Return sorted list
    return get_elements(current_mask)
