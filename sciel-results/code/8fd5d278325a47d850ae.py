import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def within_deadline():
        return elapsed() < DEADLINE

    best_seq = []
    best_len = 0

    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)

    def get_valid_candidates(seq, sums_set, suffix_sums, last_val):
        """
        Returns a list of valid x > last_val.
        """
        valid = []
        for x in range(last_val + 1, 257):
            if x in sums_set:
                continue
            
            conflict = False
            for s in suffix_sums:
                if (x + s) in sums_set:
                    conflict = True
                    break
            
            if not conflict:
                valid.append(x)
                
        return valid

    def dfs(seq, sums_set, suffix_sums, last_val, depth, heuristic_id):
        nonlocal best_seq, best_len
        
        if not within_deadline():
            return

        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)

        if last_val >= 256:
            return

        candidates = get_valid_candidates(seq, sums_set, suffix_sums, last_val)
        
        if not candidates:
            return

        # Apply heuristic to order candidates
        if heuristic_id == 0:
            candidates.sort()
        elif heuristic_id == 1:
            candidates.sort(reverse=True)
        elif heuristic_id == 2:
            random.shuffle(candidates)
        elif heuristic_id == 3:
            mid = (last_val + 256) / 2.0
            candidates.sort(key=lambda x: abs(x - mid))
        else:
            # Use a local random instance seeded by heuristic_id to be deterministic per run type?
            # Or just global random.
            random.shuffle(candidates)

        for x in candidates:
            if not within_deadline():
                break
            
            new_seq = seq + [x]
            
            new_sums_set = set(sums_set)
            new_sums_set.add(x)
            
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_sums_set.add(x + s)
                new_suffix_sums.append(x + s)
            
            dfs(new_seq, new_sums_set, new_suffix_sums, x, depth + 1, heuristic_id)

    # Initial greedy to get a baseline
    seq = []
    sums_set = set()
    suffix_sums = []
    last_val = 0
    while True:
        if not within_deadline():
            break
        candidates = get_valid_candidates(seq, sums_set, suffix_sums, last_val)
        if not candidates:
            break
        x = candidates[0] # Smallest
        seq.append(x)
        sums_set.add(x)
        new_suffix = [x]
        for s in suffix_sums:
            sums_set.add(x + s)
            new_suffix.append(x + s)
        suffix_sums = new_suffix
        last_val = x
    
    update_best(seq)
    
    # Run DFS with different heuristics
    # We interleave them to get diverse results
    heuristics = [0, 1, 2, 3, 4, 5, 6, 7]
    
    for h in heuristics:
        if not within_deadline():
            break
        dfs([], set(), [], 0, 0, h)

    return best_seq
