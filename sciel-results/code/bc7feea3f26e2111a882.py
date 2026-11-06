import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (90s limit)
    
    best_seq = []
    best_len = 0
    
    def check_and_extend(seq, sums_set, next_candidates):
        """Try to extend the sequence greedily from given state."""
        nonlocal best_seq, best_len
        cur = list(seq)
        cur_sums = set(sums_set)
        candidates = list(next_candidates)
        
        while candidates:
            # Pick the smallest candidate that works
            found = False
            for i, x in enumerate(candidates):
                # Compute all new consecutive block sums ending at x
                tails = []
                total = x
                tails.append(total)
                for a in reversed(cur):
                    total += a
                    tails.append(total)
                
                if not cur_sums.intersection(tails):
                    cur.append(x)
                    cur_sums.update(tails)
                    candidates.pop(i)
                    found = True
                    
                    if len(cur) > best_len:
                        best_len = len(cur)
                        best_seq = cur[:]
                    break
            
            if not found:
                break
        
        return cur, cur_sums
    
    # Strategy 1: Original greedy from 1
    seq1, _ = check_and_extend([], set(), list(range(1, 257)))
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 2: Try different starting points
    for start in range(1, 30):
        if time.time() > deadline:
            break
        # Start with a few initial values from different positions
        for trial in range(20):
            if time.time() > deadline:
                break
            # Random subset of early values to seed
            seed_size = random.randint(1, 4)
            seed = random.sample(range(1, 257), seed_size)
            seed.sort()
            
            seq, sums = [], set()
            valid = True
            for x in seed:
                tails = []
                total = x
                tails.append(total)
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                if sums.intersection(tails):
                    valid = False
                    break
                seq.append(x)
                sums.update(tails)
            
            if not valid:
                continue
            
            # Now try to extend greedily
            candidates = [x for x in range(seq[-1] + 1, 257)]
            cur = list(seq)
            cur_sums = set(sums)
            
            while candidates:
                found = False
                for i, x in enumerate(candidates):
                    tails = []
                    total = x
                    tails.append(total)
                    for a in reversed(cur):
                        total += a
                        tails.append(total)
                    
                    if not cur_sums.intersection(tails):
                        cur.append(x)
                        cur_sums.update(tails)
                        candidates.pop(i)
                        found = True
                        
                        if len(cur) > best_len:
                            best_len = len(cur)
                            best_seq = cur[:]
                        break
                
                if not found:
                    break
            
            if time.time() > deadline:
                break
    
    # Strategy 3: Descending greedy - try starting from high values
    for high_start in range(256, 200, -1):
        if time.time() > deadline:
            break
        seq = [high_start]
        sums = {high_start}
        
        # Try to add values below high_start
        for x in range(high_start - 1, 0, -1):
            if time.time() > deadline:
                break
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                seq.insert(0, x)
                sums.update(tails)
                
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq[:]
        
        if time.time() > deadline:
            break
    
    # Strategy 4: More aggressive search with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random increasing sequence as seed
        seed_len = random.randint(2, 6)
        seed = sorted(random.sample(range(1, 257), seed_len))
        
        seq = []
        sums = set()
        valid = True
        for x in seed:
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                valid = False
                break
            seq.append(x)
            sums.update(tails)
        
        if not valid:
            continue
        
        # Greedy extension
        candidates = [x for x in range(seq[-1] + 1, 257)]
        cur = list(seq)
        cur_sums = set(sums)
        
        while candidates:
            found = False
            for i, x in enumerate(candidates):
                tails = []
                total = x
                tails.append(total)
                for a in reversed(cur):
                    total += a
                    tails.append(total)
                
                if not cur_sums.intersection(tails):
                    cur.append(x)
                    cur_sums.update(tails)
                    candidates.pop(i)
                    found = True
                    
                    if len(cur) > best_len:
                        best_len = len(cur)
                        best_seq = cur[:]
                    break
            
            if not found:
                break
    
    # Strategy 5: Systematic exploration with different gap patterns
    for max_gap in [1, 2, 3, 5, 8, 13]:
        if time.time() > deadline:
            break
        
        # Generate candidates with controlled gaps
        candidates = []
        x = 1
        while x <= 256:
            candidates.append(x)
            x += random.randint(1, max_gap)
        
        seq = []
        sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
                
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq[:]
    
    # Final: if best_seq is empty, use the original greedy
    if not best_seq:
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        best_seq = values
    
    return best_seq
