import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    aps3_list = []
    aps4_list = []
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
                aps3_list.append(mask)
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
                aps4_list.append(mask)
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    def count_3aps(A_mask):
        count = 0
        for mask in aps3_list:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def check_4ap_validity(A_mask, x, remove_u=False, u=None):
        bit_x = 1 << (x-1)
        for ap_idx in elem_to_aps4[x]:
            mask = aps4_list[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                if remove_u:
                    bit_u = 1 << (u-1)
                    if (bit_u & others_mask) != 0:
                        continue
                return False
        return True

    def run_sa(seed, time_budget):
        rng = random.Random(seed)
        
        # Initialize with a random valid set
        A_mask = 0
        A_elements = []
        A_set = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if check_4ap_validity(A_mask, x):
                A_mask |= (1 << (x-1))
                A_set.add(x)
                A_elements.append(x)
                
        current_score = count_3aps(A_mask)
        best_score = current_score
        best_set = A_set.copy()
        
        T = 10.0
        T_min = 0.01
        alpha = 0.999
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            # Generate a move: swap u in A with v not in A
            if len(A_elements) > 0:
                idx_u = rng.randint(0, len(A_elements) - 1)
                u = A_elements[idx_u]
            else:
                u = None
            
            v = rng.randint(1, n)
            
            if u == v or (u is not None and v in A_set):
                continue
                
            # Check validity of swap
            if u is None:
                # Add move
                if not check_4ap_validity(A_mask, v):
                    continue
                new_mask = A_mask | (1 << (v-1))
                bit_v = 1 << (v-1)
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    mask = aps3_list[ap_idx]
                    others = mask ^ bit_v
                    if (A_mask & others) == others:
                        gained += 1
                delta = gained
            else:
                # Swap move
                if not check_4ap_validity(A_mask, v, remove_u=True, u=u):
                    continue
                
                bit_u = 1 << (u-1)
                bit_v = 1 << (v-1)
                
                # Lost 3-APs involving u
                lost = 0
                for ap_idx in elem_to_aps3[u]:
                    mask = aps3_list[ap_idx]
                    others = mask ^ bit_u
                    if (A_mask & others) == others:
                        if (bit_v & mask) == 0:
                            lost += 1
                            
                # Gained 3-APs involving v
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    mask = aps3_list[ap_idx]
                    others = mask ^ bit_v
                    if (A_mask & others) == others:
                        if (bit_u & others) == 0:
                            gained += 1
                            
                delta = gained - lost
                new_mask = (A_mask & ~bit_u) | bit_v

            # Acceptance
            if delta > 0:
                accept = True
            else:
                if T > T_min:
                    prob = math.exp(delta / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                if u is None:
                    A_mask = new_mask
                    A_set.add(v)
                    A_elements.append(v)
                else:
                    A_mask = new_mask
                    A_set.remove(u)
                    A_set.add(v)
                    A_elements[idx_u] = v
                    
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = A_set.copy()
                    
            T *= alpha
            
        return best_score, best_set

    # Main logic
    best_score = 0
    best_set = set()
    
    # Run multiple SA instances with different seeds
    num_runs = 3
    time_per_run = (deadline - time.time()) / num_runs - 2.0
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        score, s = run_sa(i, time_per_run)
        if score > best_score:
            best_score = score
            best_set = s

    return sorted(best_set)
