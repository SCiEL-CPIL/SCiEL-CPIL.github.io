"""Improved program using Prune-then-Grow local search."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    def get_block_sums(seq):
        sums = {}
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                if total not in sums:
                    sums[total] = []
                sums[total].append((i, j))
        return sums

    def is_valid(seq):
        sums = get_block_sums(seq)
        return all(len(v) == 1 for v in sums.values())

    def prune_to_valid(seq, max_iters=50):
        seq = list(seq)
        for _ in range(max_iters):
            if time.time() - start_time > 58:
                break
            sums = get_block_sums(seq)
            conflicts = [blocks for s, blocks in sums.items() if len(blocks) > 1]
            
            if not conflicts:
                return seq
            
            index_conflict_count = {}
            for blocks in conflicts:
                for (i, j) in blocks:
                    for k in range(i, j+1):
                        index_conflict_count[k] = index_conflict_count.get(k, 0) + 1
            
            if not index_conflict_count:
                break
                
            max_count = max(index_conflict_count.values())
            candidates = [k for k, v in index_conflict_count.items() if v == max_count]
            idx_to_remove = random.choice(candidates)
            seq.pop(idx_to_remove)
            
        return seq

    def try_grow(seq, unused):
        seq = list(seq)
        unused = [x for x in unused if x not in seq]
        
        random.shuffle(unused)
        
        # Limit the number of candidates we try to insert to keep it fast
        candidates = unused[:20] 
        
        for x in candidates:
            if time.time() - start_time > 59:
                break
            if x in seq:
                continue
                
            n = len(seq)
            # Try a few positions
            positions = [0, n, n//2, random.randint(0, n)]
            random.shuffle(positions)
            
            for pos in positions:
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    seq = new_seq
                    break
        return seq

    # Main Loop
    for _ in range(50): # 50 restarts
        if time.time() - start_time > 55:
            break
            
        # 1. Generate random initial set
        initial_size = random.randint(20, 30)
        initial_set = random.sample(range(1, 257), initial_size)
        initial_seq = sorted(initial_set)
        
        # 2. Prune to make valid
        valid_seq = prune_to_valid(initial_seq)
        
        # 3. Grow
        unused = [x for x in range(1, 257) if x not in valid_seq]
        grown_seq = try_grow(valid_seq, unused)
        
        update_best(grown_seq)
        
        # Also try a simple greedy start to ensure we have a baseline
        if _ == 0:
             # Simple forward greedy
             vals = []
             sums = set()
             for x in range(1, 257):
                 tails = []
                 total = x
                 tails.append(total)
                 for a in reversed(vals):
                     total += a
                     tails.append(total)
                 if not sums.intersection(tails):
                     vals.append(x)
                     sums.update(tails)
             update_best(vals)
             
    return best_seq
# EVOLVE-BLOCK-END
