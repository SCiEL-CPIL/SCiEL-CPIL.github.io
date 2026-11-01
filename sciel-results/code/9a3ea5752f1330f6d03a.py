def _local_search(blocks, deadline):
    active_set = set(blocks)
    triple_counts = [0] * num_triples
    for b_idx in active_set:
        for t_idx in block_triples[b_idx]:
            triple_counts[t_idx] += 1
            
    def prune():
        nonlocal active_set, triple_counts
        changed = True
        while changed:
            changed = False
            to_remove = []
            for b_idx in active_set:
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if triple_counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b_idx)
            if to_remove:
                for b_idx in to_remove:
                    active_set.remove(b_idx)
                    for t_idx in block_triples[b_idx]:
                        triple_counts[t_idx] -= 1
                changed = True
        return len(active_set)

    prune()
    current_size = len(active_set)
    all_indices = list(range(num_blocks))
    
    while time.time() < deadline:
        # Snapshot
        old_active = set(active_set)
        old_counts = triple_counts[:]
        old_size = current_size
        
        # Randomly choose between Add and Swap
        if random.random() < 0.5 and len(active_set) > 0:
            # Swap
            b_old = random.choice(list(active_set))
            b_new = random.choice(all_indices)
            if b_new in active_set:
                continue
                
            # Apply Swap
            active_set.remove(b_old)
            for t_idx in block_triples[b_old]:
                triple_counts[t_idx] -= 1
                
            active_set.add(b_new)
            for t_idx in block_triples[b_new]:
                triple_counts[t_idx] += 1
                
            prune()
            
            new_size = len(active_set)
            if new_size <= old_size:
                current_size = new_size
            else:
                active_set = old_active
                triple_counts = old_counts
        else:
            # Add
            b_new = random.choice(all_indices)
            if b_new in active_set:
                continue
                
            active_set.add(b_new)
            for t_idx in block_triples[b_new]:
                triple_counts[t_idx] += 1
                
            prune()
            
            new_size = len(active_set)
            if new_size <= old_size:
                current_size = new_size
            else:
                active_set = old_active
                triple_counts = old_counts
                
    return list(active_set)
