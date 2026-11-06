import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # Represent sets as bitmasks for fast intersection checks
    # Element i (1-indexed) corresponds to bit (i-1)
    
    aps3 = []
    aps4 = []
    
    # Map element to list of AP indices
    elem_to_aps3 = [[] for _ in range(n+1)]
    elem_to_aps4 = [[] for _ in range(n+1)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                # Store as tuple of (a, b, c) and bitmask
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                aps3.append((a, b, c, mask))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
                aps4.append((a, b, c, e, mask))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    def count_3aps(A_mask):
        count = 0
        for _, _, _, mask in aps3:
            if (A_mask & mask) == mask:
                count += 1
        return count
        
    def count_4aps(A_mask):
        count = 0
        for _, _, _, _, mask in aps4:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def is_valid_add(A_mask, x):
        # Check if adding x creates a 4-AP
        # x is 1-indexed
        bit_x = 1 << (x-1)
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d, mask = aps4[ap_idx]
            # Check if other 3 elements are in A
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                return False
        return True

    def run_greedy(seed):
        rng = random.Random(seed)
        A_mask = 0
        A_set = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        # We need to be careful: greedy addition might not be optimal for 3-APs
        # but it ensures 4-AP freedom.
        # To maximize 3-APs, we should prioritize elements that form many 3-APs with existing set.
        # However, simple random greedy is a good baseline.
        # Let's do a "weighted" greedy: at each step, pick the valid element that maximizes immediate 3-AP gain.
        # This is O(n^2) per step, but n=256, so 256*256 = 65k ops per step, 256 steps -> 16M ops.
        # This might be slow in Python if done naively.
        # Let's stick to random greedy for initialization speed, then refine with local search.
        
        for x in candidates:
            if is_valid_add(A_mask, x):
                A_mask |= (1 << (x-1))
                A_set.add(x)
                
        return A_mask, A_set

    def local_search(A_mask, A_set, time_limit):
        # Steepest ascent local search
        # Moves: Swap (remove u, add v), Add (add v if valid)
        # We will scan all possible moves and pick the best one.
        
        best_score = count_3aps(A_mask)
        no_improvement_count = 0
        
        while time.time() < time_limit:
            best_delta = 0
            best_move = None
            best_u = None
            best_v = None
            best_type = None # 'swap' or 'add'
            
            # Check Add moves
            for v in range(1, n + 1):
                if v in A_set:
                    continue
                if not is_valid_add(A_mask, v):
                    continue
                
                # Calculate delta 3-APs
                bit_v = 1 << (v-1)
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    a, b, c, mask = aps3[ap_idx]
                    others_mask = mask ^ bit_v
                    if (A_mask & others_mask) == others_mask:
                        gained += 1
                
                if gained > best_delta:
                    best_delta = gained
                    best_move = 'add'
                    best_v = v
                    best_u = None
                    best_type = 'add'
            
            # Check Swap moves
            # To speed this up, we can limit the search or sample.
            # But n=256, so 256*256 = 65k swaps.
            # For each swap, we check 4-AP validity and 3-AP delta.
            # 4-AP check: O(k4) where k4 is # 4-APs touching u or v.
            # 3-AP delta: O(k3) where k3 is # 3-APs touching u or v.
            # k3, k4 are small (~10-20).
            # So 65k * 20 = 1.3M ops per full scan.
            # This is fast enough to do many scans in 60s.
            
            for u in A_set:
                bit_u = 1 << (u-1)
                for v in range(1, n + 1):
                    if v == u or v in A_set:
                        continue
                    
                    # Check 4-AP validity for swap
                    # Remove u, add v.
                    # New set A' = (A \ {u}) U {v}
                    # Check if A' has any 4-AP.
                    # This is equivalent to checking if any 4-AP is fully contained in A'.
                    # We can check:
                    # 1. Does adding v create a 4-AP with elements in A\{u}?
                    # 2. Does removing u break a 4-AP? (No, removing elements doesn't create 4-APs)
                    # So we only need to check if v completes a 4-AP with other 3 elements in A\{u}.
                    
                    valid = True
                    bit_v = 1 << (v-1)
                    for ap_idx in elem_to_aps4[v]:
                        a, b, c, d, mask = aps4[ap_idx]
                        others_mask = mask ^ bit_v
                        # The other 3 elements must be in A\{u}
                        # i.e., in A, and not u
                        # (A_mask & others_mask) == others_mask AND (bit_u & others_mask) == 0
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                valid = False
                                break
                            else:
                                # One of the others is u, so it's removed.
                                # This 4-AP is not formed.
                                pass
                    
                    if not valid:
                        continue
                    
                    # Calculate 3-AP delta
                    lost = 0
                    for ap_idx in elem_to_aps3[u]:
                        a, b, c, mask = aps3[ap_idx]
                        others_mask = mask ^ bit_u
                        # Others must be in A\{u} (which they are, by definition of mask)
                        # But we need to ensure they are NOT v? No, v is not in A yet.
                        # So if others are in A, the AP is lost.
                        if (A_mask & others_mask) == others_mask:
                            lost += 1
                            
                    gained = 0
                    for ap_idx in elem_to_aps3[v]:
                        a, b, c, mask = aps3[ap_idx]
                        others_mask = mask ^ bit_v
                        # Others must be in A\{u}
                        # i.e., in A, and not u
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                gained += 1
                            else:
                                # One other is u, so it's removed. AP not formed.
                                pass
                                
                    delta = gained - lost
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_move = 'swap'
                        best_u = u
                        best_v = v
                        best_type = 'swap'
                        
            if best_move is None:
                break
            
            # Apply best move
            if best_type == 'add':
                A_mask |= (1 << (best_v-1))
                A_set.add(best_v)
            else:
                A_mask &= ~(1 << (best_u-1))
                A_set.remove(best_u)
                A_mask |= (1 << (best_v-1))
                A_set.add(best_v)
                
            no_improvement_count = 0
            
        return A_mask, A_set, count_3aps(A_mask)

    # Main logic
    best_mask = 0
    best_set = set()
    best_score = 0
    
    # Try several random initializations
    num_initializations = 20
    time_per_init = (deadline - time.time()) / num_initializations
    
    for i in range(num_initializations):
        if time.time() > deadline - 5:
            break
            
        A_mask, A_set = run_greedy(i)
        
        # Refine with local search
        # Give each local search a time budget
        ls_time_limit = min(time.time() + time_per_init * 0.8, deadline - 2)
        
        A_mask, A_set, score = local_search(A_mask, A_set, ls_time_limit)
        
        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = A_set
            
    # Final verification
    if count_4aps(best_mask) != 0:
        # Fallback
        A_mask, A_set = run_greedy(0)
        best_mask = A_mask
        best_set = A_set
        best_score = count_3aps(best_mask)

    return sorted(best_set)

# EVOLVE-BLOCK-END
