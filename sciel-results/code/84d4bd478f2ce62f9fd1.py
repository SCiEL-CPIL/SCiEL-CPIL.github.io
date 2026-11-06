import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    def greedy_asc():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals
    
    def greedy_desc():
        vals = []
        sums_set = set()
        for x in range(256, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals
    
    def greedy_skip_asc(skip_pattern):
        """Greedy with periodic skipping to avoid collisions"""
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals
    
    # Try multiple greedy strategies
    candidates = []
    candidates.append(greedy_asc())
    candidates.append(greedy_desc())
    
    # Greedy with different starting points
    for start in range(1, 20):
        vals = []
        sums_set = set()
        for x in range(start, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        candidates.append(vals)
    
    best_seq = max(candidates, key=len)
    best_len = len(best_seq)
    
    # Compute all consecutive block sums for a sequence
    def get_all_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    # Try to extend the best sequence by inserting new elements
    def try_extend(seq):
        """Try to insert additional elements into the sequence"""
        current = list(seq)
        current_sums = get_all_sums(current)
        
        # Try inserting each unused number at each position
        used = set(current)
        improved = True
        while improved:
            improved = False
            for pos in range(len(current) + 1):
                # Determine valid range for insertion at pos
                if pos == 0:
                    lo, hi = 1, current[0] - 1 if current else 256
                elif pos == len(current):
                    lo, hi = current[-1] + 1, 256
                else:
                    lo, hi = current[pos-1] + 1, current[pos] - 1
                
                for x in range(hi, lo - 1, -1):
                    if x in used:
                        continue
                    # Check if inserting x at pos keeps all sums distinct
                    new_seq = current[:pos] + [x] + current[pos:]
                    new_sums = get_all_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return current
    
    # Try to extend
    extended = try_extend(best_seq)
    if len(extended) > best_len:
        best_seq = extended
        best_len = len(best_seq)
    
    # Local search: try swapping elements to free up room for more insertions
    def local_search(seq, max_iters=200):
        """Try random swaps to find a configuration that allows more insertions"""
        current = list(seq)
        current_len = len(current)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            # Try removing one element and re-adding it elsewhere
            if len(current) < 2:
                break
            i = __import__('random').randint(0, len(current) - 1)
            val = current.pop(i)
            
            # Try to reinsert at a different position
            inserted = False
            for pos in range(len(current) + 1):
                if pos == i:
                    continue
                # Check bounds
                if pos == 0:
                    if current and current[0] <= val:
                        continue
                elif pos == len(current):
                    if current and current[-1] >= val:
                        continue
                else:
                    if current[pos-1] >= val or current[pos] <= val:
                        continue
                
                new_seq = current[:pos] + [val] + current[pos:]
                new_sums = get_all_sums(new_seq)
                if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                    current = new_seq
                    inserted = True
                    break
            
            if not inserted:
                current.insert(i, val)
            
            # Try to extend
            extended = try_extend(current)
            if len(extended) > current_len:
                return extended
        
        return current
    
    # Run local search
    searched = local_search(best_seq)
    if len(searched) > best_len:
        best_seq = searched
        best_len = len(best_seq)
    
    # DFS with better pruning - try to find even longer sequences
    def dfs(seq, sums_set, last_val):
        nonlocal best_seq, best_len, start_time, time_limit
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: max possible extension
        remaining = 256 - last_val
        if current_len + remaining <= best_len:
            return
        
        # Try candidates from high to low
        for x in range(256, last_val, -1):
            if (256 - x) % 8 == 0 and time.time() - start_time > time_limit:
                return
            
            # Compute new sums
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(seq):
                total += a
                new_sums.add(total)
            
            if sums_set.intersection(new_sums):
                continue
            
            dfs(seq + [x], sums_set | new_sums, x)
            
            if time.time() - start_time > time_limit:
                return
    
    # Start DFS from the best sequence found
    if best_seq:
        dfs(best_seq, get_all_sums(best_seq), best_seq[-1])
    
    return best_seq
