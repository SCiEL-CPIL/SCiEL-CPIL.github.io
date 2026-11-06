import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_solution = None
    best_max_val = 100  # Initial guess, will be updated
    
    def get_diffs(elements):
        diffs = set()
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                diffs.add(elements[j] - elements[i])
        return diffs

    def is_valid_sidon(elements):
        n = len(elements)
        seen = set()
        for i in range(n):
            for j in range(i+1, n):
                d = elements[j] - elements[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def covers_all_diffs(elements):
        diffs = get_diffs(elements)
        return all(d in diffs for d in range(1, 31))

    def backtrack(elements, used_diffs, max_val_bound):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return False
        
        if all(d in used_diffs for d in range(1, 31)):
            current_max = elements[-1]
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = list(elements)
            return True
        
        if len(elements) > 32:
            return False
        
        last = elements[-1]
        if last >= best_max_val:
            return False
            
        remaining_needed = 30 - len(used_diffs & set(range(1, 31)))
        if remaining_needed > 0:
            max_possible_new_diffs = (32 - len(elements)) * len(elements)
            if remaining_needed > max_possible_new_diffs:
                return False
        
        candidates = []
        for candidate in range(last + 1, max_val_bound + 1):
            if time.time() > deadline:
                return False
            
            new_diffs = []
            valid = True
            for i in range(len(elements)):
                d = candidate - elements[i]
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            new_required_covered = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used_diffs)
            candidates.append((candidate, new_diffs, new_required_covered))
        
        candidates.sort(key=lambda x: (-x[2], x[0]))
        
        for candidate, new_diffs, _ in candidates:
            new_used_diffs = set(used_diffs)
            new_used_diffs.update(new_diffs)
            
            if backtrack(elements + (candidate,), new_used_diffs, max_val_bound):
                return True
        
        return False

    # Search for solutions with increasing bounds
    # Start with a reasonable bound
    for bound in range(40, 80, 2):
        if time.time() > deadline:
            break
        backtrack((0,), set(), bound)
        
    if best_solution is None:
        # Fallback
        sol = [0, 1, 3, 7, 12, 20, 30, 44, 65]
        if is_valid_sidon(sol) and covers_all_diffs(sol):
            best_solution = sol
        else:
            best_solution = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    return best_solution
