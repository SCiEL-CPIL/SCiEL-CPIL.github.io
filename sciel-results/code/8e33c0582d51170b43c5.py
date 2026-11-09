import random
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115
    N = 284
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    def get_initial_state(k):
        S = list(prescribed)
        S_set = set(S)
        candidates = [x for x in range(1, N + 1) if x not in S_set]
        random.shuffle(candidates)
        for x in candidates:
            if len(S) >= k:
                break
            S.append(x)
            S_set.add(x)
        return S, S_set

    def compute_collisions(S):
        sum_counts = {}
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                sum_counts[s] = sum_counts.get(s, 0) + 1
        collisions = 0
        for count in sum_counts.values():
            if count > 1:
                collisions += count * (count - 1) // 2
        return collisions, sum_counts

    def sa_search(k, max_steps=100000, start_temp=10.0, end_temp=0.01):
        S, S_set = get_initial_state(k)
        collisions, sum_counts = compute_collisions(S)
        
        if collisions == 0:
            return S
        
        best_S = list(S)
        best_collisions = collisions
        
        T = start_temp
        alpha = (end_temp / start_temp) ** (1.0 / max_steps)
        
        non_prescribed_indices = [i for i, x in enumerate(S) if x not in prescribed_set]
        
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            if not non_prescribed_indices:
                break
                
            idx = random.choice(non_prescribed_indices)
            old_val = S[idx]
            
            new_val = random.randint(1, N)
            if new_val in S_set:
                for _ in range(10):
                    new_val = random.randint(1, N)
                    if new_val not in S_set:
                        break
                else:
                    continue
            
            sums_to_touch = set()
            for y in S:
                if y != old_val:
                    sums_to_touch.add(old_val + y)
                    sums_to_touch.add(new_val + y)
            sums_to_touch.add(2 * old_val)
            sums_to_touch.add(2 * new_val)
            
            snapshot = {s: sum_counts.get(s, 0) for s in sums_to_touch}
            
            old_local_coll = 0
            for s in sums_to_touch:
                c = snapshot[s]
                old_local_coll += c * (c - 1) // 2
            
            for y in S:
                if y != old_val:
                    s = old_val + y
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
            s_2old = 2 * old_val
            sum_counts[s_2old] -= 1
            if sum_counts[s_2old] == 0:
                del sum_counts[s_2old]
            
            for y in S:
                if y != old_val:
                    s = new_val + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
            s_2new = 2 * new_val
            sum_counts[s_2new] = sum_counts.get(s_2new, 0) + 1
            
            new_local_coll = 0
            for s in sums_to_touch:
                c = sum_counts.get(s, 0)
                new_local_coll += c * (c - 1) // 2
            
            delta_c = new_local_coll - old_local_coll
            
            accept = False
            if delta_c < 0:
                accept = True
            else:
                if T > 0:
                    if random.random() < math.exp(-delta_c / T):
                        accept = True
            
            if accept:
                S[idx] = new_val
                S_set.remove(old_val)
                S_set.add(new_val)
                collisions += delta_c
                if collisions < best_collisions:
                    best_collisions = collisions
                    best_S = list(S)
                    if collisions == 0:
                        return S
            else:
                for s in sums_to_touch:
                    if s in snapshot:
                        sum_counts[s] = snapshot[s]
                    else:
                        sum_counts.pop(s, None)
            
            T *= alpha
            
        return best_S

    best_solution = None
    best_size = len(prescribed)
    
    for k in range(20, 11, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        found = False
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            S = sa_search(k, max_steps=100000)
            if len(S) == k:
                coll, _ = compute_collisions(S)
                if coll == 0:
                    if best_solution is None or k > best_size:
                        best_solution = S
                        best_size = k
                    found = True
                    break
        
        if found:
            break
            
    if best_solution is None:
        S = list(prescribed)
        S_set = set(S)
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a+b)
        
        for x in range(12, N+1):
            if x in S_set: continue
            valid = True
            new_sums = set()
            for y in S:
                v = x + y
                if v in sums or v in new_sums:
                    valid = False
                    break
                new_sums.add(v)
            if valid:
                v = 2*x
                if v in sums or v in new_sums:
                    valid = False
            if valid:
                S.append(x)
                S_set.add(x)
                sums.update(new_sums)
                sums.add(2*x)
        best_solution = S
        
    return sorted(best_solution)
