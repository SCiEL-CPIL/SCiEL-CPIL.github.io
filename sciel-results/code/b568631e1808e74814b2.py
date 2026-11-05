import random
import time

N = 40

def construct():
    start_time = time.time()
    timeout = 80.0
    
    target_diffs = set(range(1, 39))
    n_diffs = len(target_diffs)
    
    best_S = None
    best_size = 99
    
    def get_diffs(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs
    
    def coverage(S):
        return len(get_diffs(S))
    
    def greedy_init(k):
        S = [1, 39]
        for _ in range(k - 2):
            current_diffs = get_diffs(S)
            best_candidate = None
            best_gain = -1
            for c in range(2, 39):
                if c in S:
                    continue
                # Compute new differences added
                gain = 0
                for s in S:
                    d = abs(c - s)
                    if d in target_diffs and d not in current_diffs:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
            if best_candidate is not None:
                S.append(best_candidate)
            else:
                S.append(random.randint(2, 38))
        return S
    
    # Try sizes from 10 upward
    for size in [10, 11, 12]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        num_restarts = 300 if size == 10 else 100
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Mix of greedy and random initialization
            if restart < num_restarts // 2:
                S = greedy_init(size)
            else:
                S = [1, 39]
                candidates = list(range(2, 39))
                random.shuffle(candidates)
                S.extend(candidates[:size-2])
            
            current_cov = coverage(S)
            
            if current_cov == n_diffs:
                best_S = S
                best_size = size
                found = True
                break
            
            # Local search
            for it in range(50000):
                if time.time() - start_time > timeout:
                    break
                
                if current_cov == n_diffs:
                    best_S = S
                    best_size = size
                    found = True
                    break
                
                # Random choice: single replace or swap
                if random.random() < 0.5:
                    idx = random.randint(0, len(S)-1)
                    old_val = S[idx]
                    new_val = random.randint(1, 39)
                    if new_val in S and new_val != old_val:
                        continue
                    S[idx] = new_val
                    new_cov = coverage(S)
                    if new_cov >= current_cov:
                        current_cov = new_cov
                    else:
                        S[idx] = old_val
                else:
                    if len(S) < 2:
                        continue
                    idx1, idx2 = random.sample(range(len(S)), 2)
                    old1, old2 = S[idx1], S[idx2]
                    new1, new2 = random.sample(range(1, 40), 2)
                    S_no_old = [x for i, x in enumerate(S) if i not in (idx1, idx2)]
                    if new1 in S_no_old or new2 in S_no_old:
                        continue
                    S[idx1] = new1
                    S[idx2] = new2
                    new_cov = coverage(S)
                    if new_cov >= current_cov:
                        current_cov = new_cov
                    else:
                        S[idx1] = old1
                        S[idx2] = old2
        
        if found:
            break
    
    if best_S is None:
        S = list(range(1, 40))
        best_S = S
        best_size = 39
    
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found_cycle = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            raise Exception("Cycle not found for length " + str(L))
    
    return {'edges': edges, 'cycles': cycles}
