"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    best_solution = [None]
    best_max_val = [float('inf')]

    def check_time():
        return time.time() > deadline

    def dfs(elements, used_diffs, missing_diffs):
        if check_time():
            return
        
        if not missing_diffs:
            current_max = elements[-1] if elements else 0
            if current_max < best_max_val[0]:
                best_max_val[0] = current_max
                best_solution[0] = list(elements)
            return

        if len(elements) >= 32:
            return
            
        if elements and elements[-1] >= best_max_val[0]:
            return

        last = elements[-1] if elements else 0
        
        candidates = set()
        for d in missing_diffs:
            for e in elements:
                x = e + d
                if x > last:
                    candidates.add(x)
        
        candidates = sorted(candidates)
        
        for x in candidates:
            if check_time():
                return
            
            if x >= best_max_val[0]:
                break

            new_diffs = []
            valid = True
            for e in elements:
                d = x - e
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            new_used_diffs = used_diffs.union(new_diffs)
            new_missing_diffs = missing_diffs.difference(new_diffs)
            
            dfs(elements + [x], new_used_diffs, new_missing_diffs)

    def greedy_init():
        a = [0]
        used = set()
        missing = set(range(1, 31))
        for _ in range(31):
            if not missing or check_time():
                break
            best_x = None
            best_score = -1
            candidates = set()
            for d in missing:
                for e in a:
                    x = e + d
                    if x > max(a):
                        candidates.add(x)
            
            for x in sorted(candidates):
                new_diffs = [x - e for e in a]
                if any(d in used for d in new_diffs):
                    continue
                covered = len(set(new_diffs).intersection(missing))
                if covered > best_score:
                    best_score = covered
                    best_x = x
            
            if best_x is None:
                for x in range(max(a) + 1, max(a) + 100):
                    new_diffs = [x - e for e in a]
                    if any(d in used for d in new_diffs):
                        continue
                    a.append(x)
                    used.update(new_diffs)
                    missing.difference_update(new_diffs)
                    break
            else:
                a.append(best_x)
                new_diffs = [best_x - e for e in a[:-1]]
                used.update(new_diffs)
                missing.difference_update(new_diffs)
                
        return a, max(a) if a else 0

    init_sol, init_max = greedy_init()
    if init_sol:
        diffs = {init_sol[j] - init_sol[i] for i in range(len(init_sol)) for j in range(i+1, len(init_sol))}
        if len(diffs) == len(init_sol)*(len(init_sol)-1)//2 and all(d in diffs for d in range(1, 31)):
            best_solution[0] = init_sol
            best_max_val[0] = init_max

    dfs([0], set(), set(range(1, 31)))
    
    if best_solution[0] is None:
        return [0, 1, 4, 10, 18, 29, 38, 49, 62, 77, 94, 113, 134, 157, 182, 209, 238, 269, 302, 337, 374, 413, 454, 497, 542, 589, 638, 689, 742, 797, 854, 913]
        
    return best_solution[0]
# EVOLVE-BLOCK-END
