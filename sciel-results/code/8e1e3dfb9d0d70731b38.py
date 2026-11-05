import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute t-subsets and mapping
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(t_subsets)}
    id_to_subset = {i: s for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    uncovered = set(range(num_t_subsets))
    
    # Precompute block -> set of t-subset IDs
    block_to_subset_ids = {}
    for block in itertools.combinations(range(v), k):
        subs = [subset_to_id[sub] for sub in itertools.combinations(block, t)]
        block_to_subset_ids[block] = set(subs)
        
    # Helper to insert x into sorted tuple rest and return new sorted tuple
    def insert_sorted(rest, x):
        # rest is a sorted tuple
        # Find position of x
        # Since t is small, simple loop is fine
        new_list = list(rest)
        for i in range(len(new_list)):
            if x < new_list[i]:
                new_list.insert(i, x)
                return tuple(new_list)
        new_list.append(x)
        return tuple(new_list)

    def simulate_completion(partial_block):
        current = list(partial_block)
        current_set = set(current)
        
        while len(current) < k:
            best_x = -1
            best_gain = -1
            
            # Precompute (t-1) subsets of current
            # Only if len(current) >= t-1
            if len(current) >= t - 1:
                rest_combos = list(itertools.combinations(current, t - 1))
            else:
                rest_combos = []
                
            for x in range(v):
                if x in current_set:
                    continue
                
                gain = 0
                for rest in rest_combos:
                    # Check if rest + x forms a t-subset
                    # rest is sorted tuple
                    # x is int
                    # If x is in rest, skip (but x not in current_set, so x not in rest)
                    sub_tuple = insert_sorted(rest, x)
                    sid = subset_to_id.get(sub_tuple, -1)
                    if sid != -1 and sid in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Fallback
                for x in range(v):
                    if x not in current_set:
                        best_x = x
                        break
            
            current.append(best_x)
            current_set.add(best_x)
            
        return tuple(sorted(current))

    blocks_found = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        first_uncovered_id = min(uncovered)
        start_subset = id_to_subset[first_uncovered_id]
        
        partial = list(start_subset)
        
        while len(partial) < k:
            best_x = -1
            best_potential = -1
            partial_set = set(partial)
            
            for x in range(v):
                if x in partial_set:
                    continue
                
                sim_block = simulate_completion(partial + [x])
                sim_subsets = block_to_subset_ids[sim_block]
                gain = len(sim_subsets & uncovered)
                
                if gain > best_potential:
                    best_potential = gain
                    best_x = x
            
            if best_x == -1:
                for x in range(v):
                    if x not in partial_set:
                        best_x = x
                        break
            
            partial.append(best_x)
        
        final_block = tuple(sorted(partial))
        
        covered_by_final = block_to_subset_ids[final_block]
        uncovered.difference_update(covered_by_final)
        
        blocks_found.append(list(final_block))

    return blocks_found
