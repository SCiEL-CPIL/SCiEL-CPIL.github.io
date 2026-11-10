import time
import random
from math import gcd

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    N = 256
    nodes = list(range(2, N + 1))
    
    # Precompute incompatibility sets
    incompat = {x: set() for x in nodes}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            g = gcd(a, b)
            if (a * b) // g <= 256:
                incompat[a].add(b)
                incompat[b].add(a)
                
    def score(S):
        return sum(1.0 / x for x in S)
        
    def is_compatible(x, S):
        return len(incompat[x] & S) == 0
        
    def greedy_construct(order):
        S = set()
        for x in order:
            if is_compatible(x, S):
                S.add(x)
        return S
        
    def local_search(S):
        S = set(S)
        best_score = score(S)
        improved = True
        max_iterations = 200
        iteration = 0
        
        while improved and iteration < max_iterations:
            iteration += 1
            improved = False
            
            # Phase 1: Try to add any compatible node (smallest x first)
            candidates = sorted([x for x in nodes if x not in S])
            
            for v in candidates:
                if is_compatible(v, S):
                    S.add(v)
                    best_score += 1.0 / v
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Phase 2: Try 1-swap
            best_gain = 0
            best_swap = None
            S_list = list(S)
            S_set = S
            candidates_not_in_S = sorted([x for x in nodes if x not in S_set])
            
            for u in S_list:
                S_without_u = S_set - {u}
                for v in candidates_not_in_S:
                    if is_compatible(v, S_without_u):
                        gain = (1.0 / v) - (1.0 / u)
                        if gain > best_gain + 1e-12:
                            best_gain = gain
                            best_swap = (u, v)
                        break
            
            if best_swap is not None:
                u, v = best_swap
                S.remove(u)
                S.add(v)
                best_score += best_gain
                improved = True
                continue
                
            # Phase 3: Try 1-remove + greedy-add (limited to 3 additions)
            best_move = None
            best_move_score = best_score
            candidates_not_in_S = sorted([x for x in nodes if x not in S])
            
            for u in S_list:
                S_temp = S - {u}
                current_score = best_score - (1.0 / u)
                
                added_count = 0
                for v in candidates_not_in_S:
                    if added_count >= 3:
                        break
                    if is_compatible(v, S_temp):
                        S_temp.add(v)
                        current_score += 1.0 / v
                        added_count += 1
                        
                if current_score > best_move_score + 1e-12:
                    best_move_score = current_score
                    best_move = S_temp.copy()
                    
            if best_move is not None:
                S = best_move
                best_score = best_move_score
                improved = True
                continue
                
            # Phase 4: Try 2-swap for small sets
            best_gain = 0
            best_2swap = None
            S_list = list(S)
            candidates_not_in_S = sorted([x for x in nodes if x not in S])
            
            if len(S_list) <= 15:
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u1, u2 = S_list[i], S_list[j]
                        S_without = S - {u1, u2}
                        current_loss = 1.0 / u1 + 1.0 / u2
                        
                        best_pair_gain = 0
                        best_pair = None
                        for vi in range(len(candidates_not_in_S)):
                            v1 = candidates_not_in_S[vi]
                            if not is_compatible(v1, S_without):
                                continue
                            for vj in range(vi + 1, len(candidates_not_in_S)):
                                v2 = candidates_not_in_S[vj]
                                if not is_compatible(v2, S_without):
                                    continue
                                if not is_compatible(v1, {v2} | S_without):
                                    continue
                                gain = (1.0 / v1 + 1.0 / v2) - current_loss
                                if gain > best_pair_gain:
                                    best_pair_gain = gain
                                    best_pair = (v1, v2)
                        
                        if best_pair is not None and best_pair_gain > best_gain:
                            best_gain = best_pair_gain
                            best_2swap = (u1, u2, best_pair)
            
            if best_2swap is not None and best_gain > 1e-12:
                u1, u2, (v1, v2) = best_2swap
                S.discard(u1)
                S.discard(u2)
                S.add(v1)
                S.add(v2)
                best_score += best_gain
                improved = True
                continue
                
        return S

    # Main Search
    best_S = set()
    best_val = 0.0
    
    # Initial Solution 1: Greedy by smallest x
    S1 = greedy_construct(nodes)
    S1 = local_search(S1)
    val1 = score(S1)
    if val1 > best_val:
        best_val = val1
        best_S = set(S1)
        
    # Initial Solution 2: Greedy by largest x
    S2 = greedy_construct(sorted(nodes, reverse=True))
    S2 = local_search(S2)
    val2 = score(S2)
    if val2 > best_val:
        best_val = val2
        best_S = set(S2)
        
    # Random Restarts
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        if i % 3 == 0:
            random_order = nodes[:]
            random.shuffle(random_order)
            S_rand = greedy_construct(random_order)
        elif i % 3 == 1:
            if len(best_S) > 2:
                S_rand = set(best_S)
                num_remove = random.randint(1, min(3, len(S_rand) - 1))
                for _ in range(num_remove):
                    node_to_remove = random.choice(list(S_rand))
                    S_rand.remove(node_to_remove)
            else:
                random_order = nodes[:]
                random.shuffle(random_order)
                S_rand = greedy_construct(random_order)
        else:
            random_order = nodes[:]
            weights = [1.0 / x for x in random_order]
            total_weight = sum(weights)
            sampled = set()
            remaining = random_order[:]
            rem_weights = weights[:]
            for _ in range(len(random_order)):
                if not remaining:
                    break
                r = random.random() * total_weight
                cumsum = 0
                idx = 0
                for k, w in enumerate(rem_weights):
                    cumsum += w
                    if r <= cumsum:
                        idx = k
                        break
                sampled.add(remaining[idx])
                total_weight -= rem_weights[idx]
                rem_weights.pop(idx)
                remaining.pop(idx)
            S_rand = greedy_construct(list(sampled))
            
        S_rand = local_search(S_rand)
        val_rand = score(S_rand)
        
        if val_rand > best_val:
            best_val = val_rand
            best_S = set(S_rand)
            
    # Final Polish
    if len(best_S) > 1:
        for _ in range(10):
            if time.time() - start_time > timeout:
                break
            S_perturb = set(best_S)
            node_to_remove = random.choice(list(S_perturb))
            S_perturb.remove(node_to_remove)
            S_perturb = local_search(S_perturb)
            val_perturb = score(S_perturb)
            if val_perturb > best_val:
                best_val = val_perturb
                best_S = set(S_perturb)

    return list(best_S)
