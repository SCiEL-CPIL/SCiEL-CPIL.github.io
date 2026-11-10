import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 590.0
    
    forbidden = set((x*x) % m for x in range(1, m))
    forbidden.discard(0)
    
    conflict_masks = [0] * m
    for i in range(m):
        mask_val = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask_val |= (1 << j)
        conflict_masks[i] = mask_val
        
    def count_bits(mask):
        return bin(mask).count('1')
        
    def mask_to_list(mask):
        return [i for i in range(m) if mask & (1 << i)]
        
    def greedy_seed():
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        order = list(range(m))
        random.shuffle(order)
        order.sort(key=lambda x: degrees[x])
        mask = 0
        for node in order:
            if (mask & conflict_masks[node]) == 0:
                mask |= (1 << node)
        return mask
        
    # Strong seed from 41
    def solve_41():
        m41 = 41
        forb41 = set((x*x) % m41 for x in range(1, m41))
        forb41.discard(0)
        cm41 = [0] * m41
        for i in range(m41):
            mv = 0
            for j in range(m41):
                if i != j and (i-j) % m41 in forb41:
                    mv |= (1 << j)
            cm41[i] = mv
            
        degs = [bin(cm41[i]).count('1') for i in range(m41)]
        order = list(range(m41))
        order.sort(key=lambda x: degs[x])
        best_mask_41 = 0
        for node in order:
            if (best_mask_41 & cm41[node]) == 0:
                best_mask_41 |= (1 << node)
                
        # Fast SA for 41
        T = 1.0
        cur = best_mask_41
        end_t = time.time() + 1.0
        all_41 = list(range(m41))
        while time.time() < end_t:
            node = random.choice(all_41)
            if cur & (1 << node):
                new_mask = cur & ~(1 << node)
                delta = -1
            else:
                if (cur & cm41[node]) == 0:
                    new_mask = cur | (1 << node)
                    delta = 1
                else:
                    continue
            if delta > 0 or random.random() < math.exp(delta/T):
                cur = new_mask
                if bin(cur).count('1') > bin(best_mask_41).count('1'):
                    best_mask_41 = cur
            T *= 0.999
            if T < 0.01: T = 0.01
        return best_mask_41

    seed_41 = solve_41()
    seed_205 = 0
    for i in range(41):
        if seed_41 & (1 << i):
            seed_205 |= (1 << (25 * i))
            
    greedy_205 = greedy_seed()
    best_mask = seed_205 if bin(seed_205).count('1') > bin(greedy_205).count('1') else greedy_205
    best_size = bin(best_mask).count('1')
    
    all_nodes = list(range(m))
    
    while time.time() < deadline:
        if random.random() < 0.1:
            start_mask = greedy_seed()
        else:
            start_mask = best_mask
            nodes_in = mask_to_list(start_mask)
            if len(nodes_in) > 5:
                to_remove = random.sample(nodes_in, min(5, len(nodes_in)))
                for n in to_remove:
                    start_mask &= ~(1 << n)
            
        cur_mask = start_mask
        cur_size = bin(cur_mask).count('1')
        T = 1.0
        end_slice = time.time() + 10.0
        
        while time.time() < end_slice:
            move = random.random()
            new_mask = cur_mask
            new_size = cur_size
            valid = False
            
            if move < 0.4:
                node = random.choice(all_nodes)
                if not (cur_mask & (1 << node)):
                    if (cur_mask & conflict_masks[node]) == 0:
                        new_mask |= (1 << node)
                        new_size += 1
                        valid = True
            elif move < 0.6:
                node = random.choice(all_nodes)
                if cur_mask & (1 << node):
                    new_mask &= ~(1 << node)
                    new_size -= 1
                    valid = True
            elif move < 0.9:
                rem_node = random.choice(all_nodes)
                if cur_mask & (1 << rem_node):
                    add_node = random.choice(all_nodes)
                    if not (cur_mask & (1 << add_node)):
                        temp = cur_mask & ~(1 << rem_node)
                        if (temp & conflict_masks[add_node]) == 0:
                            new_mask = temp | (1 << add_node)
                            new_size = cur_size
                            valid = True
            else:
                # 2-Swap
                if cur_size >= 2:
                    rem_nodes = set()
                    while len(rem_nodes) < 2:
                        n = random.choice(all_nodes)
                        if cur_mask & (1 << n):
                            rem_nodes.add(n)
                    if len(rem_nodes) == 2:
                        r1, r2 = list(rem_nodes)
                        add_nodes = set()
                        while len(add_nodes) < 2:
                            n = random.choice(all_nodes)
                            if not (cur_mask & (1 << n)) and n not in rem_nodes:
                                add_nodes.add(n)
                        if len(add_nodes) == 2:
                            a1, a2 = list(add_nodes)
                            temp = cur_mask & ~(1 << r1) & ~(1 << r2)
                            if (temp & conflict_masks[a1]) == 0 and (temp & conflict_masks[a2]) == 0 and (conflict_masks[a1] & (1 << a2)) == 0:
                                new_mask = temp | (1 << a1) | (1 << a2)
                                new_size = cur_size
                                valid = True
                                
            if not valid:
                T *= 0.999
                if T < 0.01: T = 0.01
                continue
                
            delta = new_size - cur_size
            if delta > 0 or (T > 0.01 and random.random() < math.exp(delta/T)):
                cur_mask = new_mask
                cur_size = new_size
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
            T *= 0.999
            if T < 0.01: T = 0.01
            
    final_set = mask_to_list(best_mask)
    
    F = forbidden
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            if (final_set[i] - final_set[j]) % m in F or (final_set[j] - final_set[i]) % m in F:
                return np.array([0], dtype=np.int64)
                
    return np.array(final_set, dtype=np.int64)
