import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 110
    
    v = 16
    all_triples = list(itertools.combinations(range(v), 3))
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    pairs = list(zip(all_triples, triple_masks))
    random.shuffle(pairs)
    
    count = [0] * (1 << v)
    
    def add_triple(idx):
        m = triple_masks[idx]
        # Iterate all masks
        for s in range(1 << v):
            if (s & m) == m:
                count[s] += 1
    def remove_triple(idx):
        m = triple_masks[idx]
        for s in range(1 << v):
            if (s & m) == m:
                count[s] -= 1
    def get_alpha():
        max_size = 0
        for s in range(1 << v):
            if count[s] == 0:
                pop = s.bit_count()
                if pop > max_size:
                    max_size = pop
        return max_size

    current_indices = []
    # Greedy init
    for i in range(len(pairs)):
        m = triple_masks[i]
        linear = True
        for j in current_indices:
            if (m & triple_masks[j]).bit_count() > 1:
                linear = False
                break
        if linear:
            current_indices.append(i)
            add_triple(i)
            
    current_alpha = get_alpha()
    best_alpha = current_alpha
    best_indices = current_indices[:]
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            idx = random.randint(0, len(pairs)-1)
            if idx not in current_indices:
                m = triple_masks[idx]
                linear = True
                for j in current_indices:
                    if (m & triple_masks[j]).bit_count() > 1:
                        linear = False
                        break
                if linear:
                    current_indices.append(idx)
                    add_triple(idx)
                    new_alpha = get_alpha()
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_indices = current_indices[:]
                    elif new_alpha == current_alpha:
                        if random.random() < 0.5:
                            current_alpha = new_alpha
                        else:
                            current_indices.pop()
                            remove_triple(idx)
                    else:
                        current_indices.pop()
                        remove_triple(idx)
        elif move_type == 'remove':
            if current_indices:
                idx_rm = random.choice(current_indices)
                current_indices.remove(idx_rm)
                remove_triple(idx_rm)
                new_alpha = get_alpha()
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_indices = current_indices[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.5:
                        current_alpha = new_alpha
                    else:
                        current_indices.append(idx_rm)
                        add_triple(idx_rm)
                else:
                    current_indices.append(idx_rm)
                    add_triple(idx_rm)
        elif move_type == 'swap':
            if current_indices:
                idx_rm = random.choice(current_indices)
                idx_add = random.randint(0, len(pairs)-1)
                if idx_add != idx_rm and idx_add not in current_indices:
                    m_add = triple_masks[idx_add]
                    linear = True
                    for j in current_indices:
                        if j == idx_rm:
                            continue
                        if (m_add & triple_masks[j]).bit_count() > 1:
                            linear = False
                            break
                    if linear:
                        current_indices.remove(idx_rm)
                        remove_triple(idx_rm)
                        current_indices.append(idx_add)
                        add_triple(idx_add)
                        
                        new_alpha = get_alpha()
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_indices = current_indices[:]
                        elif new_alpha == current_alpha:
                            if random.random() < 0.5:
                                current_alpha = new_alpha
                            else:
                                current_indices.remove(idx_add)
                                remove_triple(idx_add)
                                current_indices.append(idx_rm)
                                add_triple(idx_rm)
                        else:
                            current_indices.remove(idx_add)
                            remove_triple(idx_add)
                            current_indices.append(idx_rm)
                            add_triple(idx_rm)

    result = []
    for i in best_indices:
        result.append(list(pairs[i][0]))
    return result
