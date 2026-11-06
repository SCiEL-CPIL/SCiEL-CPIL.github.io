import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    aps3 = []
    aps4 = []
    
    elem_to_aps3 = [[] for _ in range(n+1)]
    elem_to_aps4 = [[] for _ in range(n+1)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
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
        bit_x = 1 << (x-1)
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d, mask = aps4[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                return False
        return True

    def run_sa(seed, target_size):
        rng = random.Random(seed)
        
        # Initialize with a random valid set of size target_size
        # We use a greedy random fill to ensure validity
        A_set = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        A_mask = 0
        
        for x in candidates:
            if len(A_set) >= target_size:
                break
            if is_valid_add(A_mask, x):
                A_set.add(x)
                A_mask |= (1 << (x-1))
        
        # If we couldn't fill up to target_size, we work with what we have
        # But usually we can fill up to ~100+
        
        current_score = count_3aps(A_mask)
        best_score = current_score
        best_mask = A_mask
        best_set = set(A_set)
        
        # Simulated Annealing Parameters
        T_start = 10.0
        T_end = 0.1
        steps = 20000
        # Time check per iteration is too slow, so we check every 1000 steps
        # Or we just run a fixed number of steps and rely on the outer loop time check
        
        # We estimate time per step.
        # Swap validity check: O(k4) ~ 20 ops.
        # Score delta: O(k3) ~ 20 ops.
        # Total ~ 40 ops. 20k steps * 40 = 800k ops. Very fast in Python? 
        # Actually, Python overhead is high. Let's say 1 microsecond per op? 
        # 800k ops * 1us = 0.8s. So 20k steps is ~1s.
        # We can do more steps if time permits.
        
        for i in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random u in A, v not in A
            if not A_set:
                break
                
            u = rng.choice(list(A_set))
            # Pick v from complement
            # Efficient way: pick random number until not in A_set?
            # Or maintain a list of non-elements? 
            # For n=256, checking membership in set is O(1).
            # Picking v:
            while True:
                v = rng.randint(1, n)
                if v not in A_set:
                    break
            
            # Check if swap (u -> v) is valid
            # Valid if A' = (A \ {u}) U {v} has no 4-APs.
            # This is true iff v does not complete a 4-AP with 3 elements from A\{u}.
            valid = True
            bit_v = 1 << (v-1)
            bit_u = 1 << (u-1)
            for ap_idx in elem_to_aps4[v]:
                a, b, c, d, mask = aps4[ap_idx]
                others_mask = mask ^ bit_v
                # Others must be in A_mask
                if (A_mask & others_mask) == others_mask:
                    # If any of the others is u, then it's removed, so AP is not formed
                    # If none of the others is u, then AP is formed -> Invalid
                    if (bit_u & others_mask) == 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Calculate score delta
            lost = 0
            for ap_idx in elem_to_aps3[u]:
                a, b, c, mask = aps3[ap_idx]
                others_mask = mask ^ bit_u
                if (A_mask & others_mask) == others_mask:
                    lost += 1
                    
            gained = 0
            for ap_idx in elem_to_aps3[v]:
                a, b, c, mask = aps3[ap_idx]
                others_mask = mask ^ bit_v
                if (A_mask & others_mask) == others_mask:
                    # Ensure u is not in others (it isn't, as u is removed)
                    # But wait, if u is in others_mask, then the AP was formed by u, v, w.
                    # If we remove u, is it still formed? No, because u is gone.
                    # So we only count it if the other two are in A\{u}.
                    # others_mask contains v and two others.
                    # We need to check if the two others are in A_mask AND not u.
                    # Since v is not in A_mask yet, (A_mask & others_mask) checks if the other two are in A.
                    # If u is one of the other two, then (A_mask & others_mask) will include u's bit.
                    # So we must ensure u is NOT in others_mask.
                    if (bit_u & others_mask) == 0:
                        gained += 1
                    # If u IS in others_mask, then the AP was (u, x, v). 
                    # Removing u breaks it. So it is not gained.
                    
            delta = gained - lost
            
            # Acceptance criterion
            if delta > 0:
                accept = True
            else:
                T = T_start * (T_end / T_start) ** (i / steps)
                if rng.random() < math.exp(delta / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                # Apply swap
                A_set.remove(u)
                A_set.add(v)
                A_mask ^= bit_u
                A_mask |= bit_v
                
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                    best_set = set(A_set)
                    
        return best_mask, best_set, best_score

    best_mask = 0
    best_set = set()
    best_score = 0
    
    # Try different target sizes and seeds
    # Heuristic: optimal size is around 1/3 to 1/2 of n? 
    # For n=256, maybe 60-100.
    target_sizes = [60, 70, 80, 90, 100]
    seeds = [1, 2, 3, 4, 5]
    
    # Shuffle order of (size, seed) pairs to distribute time
    trials = [(s, sz) for sz in target_sizes for s in seeds]
    random.shuffle(trials)
    
    for seed, sz in trials:
        if time.time() > deadline - 10:
            break
            
        A_mask, A_set, score = run_sa(seed, sz)
        
        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = A_set
            
    # Final verification
    if count_4aps(best_mask) != 0:
        # Fallback to a simple valid set
        # Just pick random valid elements
        rng = random.Random(0)
        A_mask = 0
        A_set = set()
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        for x in candidates:
            if is_valid_add(A_mask, x):
                A_set.add(x)
                A_mask |= (1 << (x-1))
        best_mask = A_mask
        best_set = A_set
        best_score = count_3aps(best_mask)

    return sorted(best_set)
