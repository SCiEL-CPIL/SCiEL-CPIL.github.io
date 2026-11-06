import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Calculate initial sums
    initial_sums = set()
    for i in range(len(base)):
        for j in range(i, len(base)):
            initial_sums.add(base[i] + base[j])
            
    all_numbers = list(range(1, 285))
    
    # Greedy initialization
    def greedy_solve():
        curr_set = set(base)
        curr_sums = set(initial_sums)
        candidates = [x for x in all_numbers if x not in base_set]
        
        valid_candidates = []
        for x in candidates:
            if 2*x in curr_sums: continue
            conflict = False
            for y in curr_set:
                if x+y in curr_sums:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
                
        while valid_candidates:
            x = valid_candidates[0]
            
            new_sums = [2*x]
            for y in curr_set:
                new_sums.append(x+y)
                
            curr_set.add(x)
            for s in new_sums:
                curr_sums.add(s)
                
            new_valid = []
            for y in valid_candidates[1:]:
                if 2*y in curr_sums: continue
                conflict = False
                for z in curr_set:
                    if y+z in curr_sums:
                        conflict = True
                        break
                if not conflict:
                    new_valid.append(y)
            valid_candidates = new_valid
            
        return curr_set

    best_solution = greedy_solve()
    best_size = len(best_solution)
    
    def solve_bnb(current_set, current_sums, candidates, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
            
        if len(current_set) + len(candidates) <= best_size:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_solution = set(current_set)
            return
            
        x = candidates[0]
        
        # Branch 1: Include x
        if 2*x not in current_sums:
            new_sums = []
            valid = True
            for y in current_set:
                s = x + y
                if s in current_sums:
                    valid = False
                    break
                new_sums.append(s)
            new_sums.append(2*x)
            
            if valid:
                current_set.add(x)
                for s in new_sums:
                    current_sums.add(s)
                    
                new_candidates = []
                for y in candidates[1:]:
                    if 2*y in current_sums: continue
                    conf = False
                    for z in current_set:
                        if y+z in current_sums:
                            conf = True
                            break
                    if not conf:
                        new_candidates.append(y)
                        
                solve_bnb(current_set, current_sums, new_candidates, depth+1)
                
                current_set.remove(x)
                for s in new_sums:
                    current_sums.remove(s)
                    
        # Branch 2: Exclude x
        solve_bnb(current_set, current_sums, candidates[1:], depth+1)
        
    initial_candidates = [x for x in all_numbers if x not in base_set]
    filtered_candidates = []
    for x in initial_candidates:
        if 2*x in initial_sums: continue
        conf = False
        for y in base_set:
            if x+y in initial_sums:
                conf = True
                break
        if not conf:
            filtered_candidates.append(x)
            
    solve_bnb(base_set, set(initial_sums), filtered_candidates, 0)
    
    return sorted(best_solution)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
