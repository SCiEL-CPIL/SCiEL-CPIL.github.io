import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit

    best_seq = []
    best_len = 0

    def dfs(seq, sums, remaining):
        nonlocal best_seq, best_len, deadline

        if time.time() > deadline:
            return

        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

        if not remaining:
            return

        # Find valid candidates
        candidates = list(remaining)
        random.shuffle(candidates)
        
        valid_candidates = []
        max_consider = 10
        
        for x in candidates:
            if len(valid_candidates) >= max_consider:
                break
            if time.time() > deadline:
                return

            # Check if x can be appended to seq
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            valid = True
            for s in new_sums:
                if s in sums:
                    valid = False
                    break
            
            if valid:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return

        # Try each valid candidate
        for x in valid_candidates:
            if time.time() > deadline:
                return

            # Recurse
            new_seq = seq + [x]
            
            new_sums_list = []
            total = x
            new_sums_list.append(total)
            for a in reversed(seq):
                total += a
                new_sums_list.append(total)
            
            new_sums_set = sums.union(new_sums_list)
            new_remaining = remaining - {x}
            
            dfs(new_seq, new_sums_set, new_remaining)

    initial_candidates = set(range(1, 257))
    dfs([], set(), initial_candidates)

    return best_seq
