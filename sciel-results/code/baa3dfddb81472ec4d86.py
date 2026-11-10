import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_set = []
    best_size = 0
    
    def try_add(chosen, counts, x):
        # Returns True if valid, and updates counts if so
        # But we need to check first.
        pass

    def run_greedy(strategy):
        chosen = []
        counts = {}
        candidates = set(range(1, 1001))
        
        while candidates:
            if time.time() > deadline:
                break
            
            if strategy == 'smallest':
                # Find smallest valid
                found = None
                for x in sorted(candidates): # Sorted every time is slow. Keep list?
                    # Since we remove, maintaining sorted list is hard.
                    # Just iterate 1..1000 and check if in candidates?
                    pass
                
            # Better: Keep a list of candidates.
            # For 'smallest', iterate from 1.
            # For others, iterate all.
            
            valid_list = []
            
            if strategy == 'smallest':
                for x in range(1, 1001):
                    if x not in candidates: continue
                    # Check validity
                    ok = True
                    for y in chosen:
                        if y > x: continue
                        if counts.get(x+y, 0) == 2:
                            ok = False
                            break
                    if ok and counts.get(2*x, 0) == 2:
                        ok = False
                    if ok:
                        found = x
                        break
                if found:
                    # Commit
                    chosen.append(found)
                    candidates.discard(found)
                    for y in chosen[:-1]:
                        if y <= found:
                            counts[found+y] = counts.get(found+y, 0) + 1
                    counts[2*found] = counts.get(2*found, 0) + 1
                else:
                    break
            else:
                # 'largest', 'least_sat', 'rand'
                # Iterate all candidates
                for x in candidates:
                    ok = True
                    sat = 0
                    for y in chosen:
                        if y > x: continue
                        c = counts.get(x+y, 0)
                        if c == 2:
                            ok = False
                            break
                        if c == 1:
                            sat += 1
                    if ok:
                        c = counts.get(2*x, 0)
                        if c == 2:
                            ok = False
                        elif c == 1:
                            sat += 1
                    if ok:
                        valid_list.append((x, sat))
                
                if not valid_list:
                    break
                
                if strategy == 'largest':
                    x = max(valid_list, key=lambda t: t[0])[0]
                elif strategy == 'least_sat':
                    # Pick min sat, break ties by smallest x?
                    x = min(valid_list, key=lambda t: (t[1], t[0]))[0]
                elif strategy == 'rand':
                    x = random.choice(valid_list)[0]
                elif strategy == 'least_sat_rand':
                    # Pick from top 5 lowest sat
                    valid_list.sort(key=lambda t: t[1])
                    top = valid_list[:5]
                    x = random.choice(top)[0]
                else:
                    x = random.choice(valid_list)[0]
                
                # Commit
                chosen.append(x)
                candidates.discard(x)
                for y in chosen[:-1]:
                    if y <= x:
                        counts[x+y] = counts.get(x+y, 0) + 1
                counts[2*x] = counts.get(2*x, 0) + 1
        
        return chosen

    strategies = ['smallest', 'largest', 'least_sat', 'least_sat_rand', 'rand']
    
    # Run each strategy multiple times
    for s in strategies:
        for _ in range(20): # 20 restarts per strategy
            if time.time() > deadline:
                break
            res = run_greedy(s)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res[:]
        
        if time.time() > deadline:
            break
            
    return best_set
