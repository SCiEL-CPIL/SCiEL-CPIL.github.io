import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580.0
    
    forbidden = set((x*x) % m for x in range(1, m))
    forbidden.discard(0)
    
    conflict_masks = [0] * m
    for i in range(m):
        mask_val = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask_val |= (1 << j)
        conflict_masks[i] = mask_val
    
    # Precompute: for each node, which nodes can be added if this node is removed
    # and which nodes conflict with this node
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
    
    def try_add_node(cur_mask, node):
        return (cur_mask & conflict_masks[node]) == 0
        
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
                
        T = 1.0
        cur = best_mask_41
        cur_size = bin(cur).count('1')
        best_size_41 = cur_size
        end_t = time.time() + 1.5
        all_41 = list(range(m41))
        while time.time() < end_t:
            node = random.choice(all_41)
            if cur & (1 << node):
                new_mask = cur & ~(1 << node)
                new_size = cur_size - 1
                delta = -1
            else:
                if (cur & cm41[node]) == 0:
                    new_mask = cur | (1 << node)
                    new_size = cur_size + 1
                    delta = 1
                else:
                    continue
            if delta > 0 or random.random() < math.exp(delta/T):
                cur = new_mask
                cur_size = new_size
                if cur_size > best_size_41:
                    best_size_41 = cur_size
                    best_mask_41 = cur
            T *= 0.9995
            if T < 0.01: T = 0.01
        return best_mask_41

    # Try multiple lift strategies
    def lift_41(mask_41, offset=0, scale=25):
        mask_205 = 0
        for i in range(41):
            if mask_41 & (1 << i):
                pos = (offset + scale * i) % 205
                mask_205 |= (1 << pos)
        return mask_205
    
    seed_41 = solve_41()
    
    # Try different lifts and pick the best
    best_seed = 0
    best_seed_size = 0
    for offset in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        candidate = lift_41(seed_41, offset=offset, scale=25)
        size = bin(candidate).count('1')
        # Verify no conflicts
        valid = True
        for i in range(m):
            if candidate & (1 << i):
                for j in range(m):
                    if i != j and (candidate & (1 << j)):
                        if (i - j) % m in forbidden:
                            valid = False
                            break
            if not valid:
                break
        if valid and size > best_seed_size:
            best_seed_size = size
            best_seed = candidate
    
    greedy_205 = greedy_seed()
    seed_count = bin(best_seed).count('1')
    greedy_count = bin(greedy_205).count('1')
    if seed_count > greedy_count:
        best_mask = best_seed
        best_size = seed_count
    else:
        best_mask = greedy_205
        best_size = greedy_count
    
    all_nodes = list(range(m))
    not_in_mask = [0] * m  # bitmasks of nodes not in current mask
    
    def update_not_in(cur_mask):
        for i in range(m):
            if cur_mask & (1 << i):
                not_in_mask[i] = 0
            else:
                not_in_mask[i] = (1 << i)
    
    # Main SA loop with improved move generation
    while time.time() < deadline:
        if random.random() < 0.05:
            start_mask = greedy_seed()
        else:
            start_mask = best_mask
            nodes_in = mask_to_list(start_mask)
            if len(nodes_in) > 3:
                k = random.randint(2, min(5, len(nodes_in)))
                to_remove = random.sample(nodes_in, k)
                for n in to_remove:
                    start_mask &= ~(1 << n)
            
        cur_mask = start_mask
        cur_size = bin(cur_mask).count('1')
        T = 2.0
        end_slice = time.time() + 8.0
        
        # Precompute addable nodes for initial state
        addable = []
        for n in all_nodes:
            if not (cur_mask & (1 << n)):
                if (cur_mask & conflict_masks[n]) == 0:
                    addable.append(n)
        
        while time.time() < end_slice:
            move = random.random()
            new_mask = cur_mask
            new_size = cur_size
            valid = False
            
            if move < 0.35:
                # Add a node
                if addable:
                    node = random.choice(addable)
                    new_mask |= (1 << node)
                    new_size += 1
                    valid = True
                    # Update addable: remove node, remove nodes that conflict with it
                    new_addable = [n for n in addable if n != node and not (conflict_masks[node] & (1 << n))]
                    # Add nodes that can now be added
                    for n in all_nodes:
                        if not (cur_mask & (1 << n)) and n != node:
                            if (new_mask & conflict_masks[n]) == 0 and n not in new_addable:
                                new_addable.append(n)
                    addable = new_addable
            elif move < 0.55:
                # Remove a node
                nodes_in = mask_to_list(cur_mask)
                if nodes_in:
                    node = random.choice(nodes_in)
                    new_mask &= ~(1 << node)
                    new_size -= 1
                    valid = True
                    # Update addable: add node back, add nodes that conflict with it
                    new_addable = list(addable)
                    if node not in new_addable:
                        new_addable.append(node)
                    for n in all_nodes:
                        if not (cur_mask & (1 << n)) and n != node:
                            if (conflict_masks[node] & (1 << n)) and n not in new_addable:
                                new_addable.append(n)
                    addable = new_addable
            elif move < 0.85:
                # Swap: remove one, add one
                nodes_in = mask_to_list(cur_mask)
                if nodes_in and addable:
                    rem_node = random.choice(nodes_in)
                    temp = cur_mask & ~(1 << rem_node)
                    # Find addable nodes after removal
                    candidates = []
                    for n in all_nodes:
                        if not (cur_mask & (1 << n)):
                            if (temp & conflict_masks[n]) == 0:
                                candidates.append(n)
                    if candidates:
                        add_node = random.choice(candidates)
                        new_mask = temp | (1 << add_node)
                        new_size = cur_size
                        valid = True
                        # Update addable
                        new_addable = [n for n in candidates if n != add_node]
                        for n in all_nodes:
                            if not (temp & (1 << n)) and n != add_node:
                                if (new_mask & conflict_masks[n]) == 0 and n not in new_addable:
                                    new_addable.append(n)
                        addable = new_addable
            else:
                # 2-Swap
                if cur_size >= 2:
                    nodes_in = mask_to_list(cur_mask)
                    if len(nodes_in) >= 2:
                        r1, r2 = random.sample(nodes_in, 2)
                        temp = cur_mask & ~(1 << r1) & ~(1 << r2)
                        candidates = []
                        for n in all_nodes:
                            if not (cur_mask & (1 << n)):
                                if (temp & conflict_masks[n]) == 0:
                                    candidates.append(n)
                        if len(candidates) >= 2:
                            a1, a2 = random.sample(candidates, 2)
                            if (conflict_masks[a1] & (1 << a2)) == 0:
                                new_mask = temp | (1 << a1) | (1 << a2)
                                new_size = cur_size
                                valid = True
                                # Update addable
                                new_addable = [n for n in candidates if n != a1 and n != a2]
                                for n in all_nodes:
                                    if not (temp & (1 << n)) and n != a1 and n != a2:
                                        if (new_mask & conflict_masks[n]) == 0 and n not in new_addable:
                                            new_addable.append(n)
                                addable = new_addable
                                
            if not valid:
                T *= 0.999
                if T < 0.02: T = 0.02
                continue
                
            delta = new_size - cur_size
            if delta > 0 or (T > 0.02 and random.random() < math.exp(delta/T)):
                cur_mask = new_mask
                cur_size = new_size
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
            T *= 0.999
            if T < 0.02: T = 0.02
            
    # Final polish: greedy add
    final_mask = best_mask
    changed = True
    while changed:
        changed = False
        for n in all_nodes:
            if not (final_mask & (1 << n)):
                if (final_mask & conflict_masks[n]) == 0:
                    final_mask |= (1 << n)
                    changed = True
    
    final_set = mask_to_list(final_mask)
    
    F = forbidden
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            if (final_set[i] - final_set[j]) % m in F or (final_set[j] - final_set[i]) % m in F:
                return np.array([0], dtype=np.int64)
                
    return np.array(final_set, dtype=np.int64)
