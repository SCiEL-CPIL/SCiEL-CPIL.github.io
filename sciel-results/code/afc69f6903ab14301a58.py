import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85

    # Precompute comparability lists
    # comp_list[i] contains all j such that i and j are comparable
    comp_list = [[] for _ in range(total_masks)]
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp_list[i].append(j)
                comp_list[j].append(i)

    def is_comp(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0

    def greedy_init():
        selected = []
        # Start with a chain
        chain = [0]
        for i in range(1, n):
            chain.append(chain[-1] | (1 << i))
        selected.extend(chain)
        selected_set = set(selected)
        
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            for mask in range(total_masks):
                if mask in selected_set:
                    continue
                score = 0
                for s in selected:
                    if is_comp(mask, s):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            selected.append(best_mask)
            selected_set.add(best_mask)
        return selected

    best_config = greedy_init()
    
    # Calculate initial score
    best_score = 0
    for i in range(m):
        for j in range(i+1, m):
            if is_comp(best_config[i], best_config[j]):
                best_score += 1

    current_config = best_config[:]
    current_set = set(current_config)
    
    # Build compat array
    compat = [0] * total_masks
    for s in current_config:
        for v in comp_list[s]:
            compat[v] += 1
    # Note: compat[s] for s in current_config includes comparisons with other selected sets.
    # It does NOT include self (since s is not in comp_list[s]).
    # So compat[s] is exactly the degree of s in the induced subgraph.
    
    current_score = best_score

    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Find min compat in selected
        min_u_idx = -1
        min_u_val = float('inf')
        for i, s in enumerate(current_config):
            if compat[s] < min_u_val:
                min_u_val = compat[s]
                min_u_idx = i
                
        # Find max compat in unselected
        max_v = -1
        max_v_val = -1
        for mask in range(total_masks):
            if mask not in current_set:
                if compat[mask] > max_v_val:
                    max_v_val = compat[mask]
                    max_v = mask
                    
        if max_v_val > min_u_val:
            # Perform swap
            u = current_config[min_u_idx]
            v = max_v
            
            current_set.remove(u)
            current_set.add(v)
            current_config[min_u_idx] = v
            
            current_score += max_v_val - min_u_val
            
            # Update compat
            for w in comp_list[u]:
                compat[w] -= 1
            for w in comp_list[v]:
                compat[w] += 1
                
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_config = current_config[:]
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation
                if random.random() < 0.05:
                    i1 = random.randint(0, m-1)
                    i2 = random.randint(0, m-1)
                    if i1 == i2:
                        no_improve = 0
                        continue
                        
                    u1 = current_config[i1]
                    u2 = current_config[i2]
                    
                    # Find 2 random unselected
                    v1 = -1
                    v2 = -1
                    for _ in range(100):
                        v1 = random.randint(0, total_masks-1)
                        v2 = random.randint(0, total_masks-1)
                        if v1 in current_set or v2 in current_set or v1 == v2:
                            continue
                        break
                    if v1 == -1:
                        no_improve = 0
                        continue

                    # Check gain
                    # Gain = (compat[v1] + compat[v2]) - (compat[u1] + compat[u2])
                    # This is an approximation. It ignores the edge (u1, u2) and (v1, v2) and cross edges.
                    # If u1 and u2 are comparable, we lose that edge.
                    # If v1 and v2 are comparable, we gain that edge.
                    # If u1 and v2 are comparable, we lose that edge (u1 removed).
                    # If u2 and v1 are comparable, we lose that edge (u2 removed).
                    # If u1 and v1 are comparable, we gain that edge (v1 added).
                    # If u2 and v2 are comparable, we gain that edge (v2 added).
                    
                    # Let's just compute exact delta
                    old_score = current_score
                    
                    current_set.remove(u1)
                    current_set.remove(u2)
                    current_set.add(v1)
                    current_set.add(v2)
                    current_config[i1] = v1
                    current_config[i2] = v2
                    
                    # Recalculate score (expensive)
                    new_score = 0
                    for i in range(m):
                        for j in range(i+1, m):
                            if is_comp(current_config[i], current_config[j]):
                                new_score += 1
                                
                    if new_score > old_score:
                        current_score = new_score
                        # Rebuild compat
                        for k in range(total_masks):
                            compat[k] = 0
                        for s in current_config:
                            for w in comp_list[s]:
                                compat[w] += 1
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_config = current_config[:]
                    else:
                        # Revert
                        current_set.remove(v1)
                        current_set.remove(v2)
                        current_set.add(u1)
                        current_set.add(u2)
                        current_config[i1] = u1
                        current_config[i2] = u2
                        current_score = old_score
                        no_improve = 0 # reset to try again
                else:
                    no_improve = 0 # reset to avoid infinite loop if stuck

    result = []
    for mask in best_config:
        result.append([i for i in range(n) if (mask >> i) & 1])
    return result
