import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    # Greedy initialization
    def greedy():
        values = []
        sums = set()
        for x in range(1, N + 1):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    init_seq, init_sums = greedy()
    best_seq = init_seq
    best_len = len(init_seq)
    
    # DFS with pruning
    def dfs(start_val, seq, suffix_sums, sums_set, depth):
        nonlocal best_seq, best_len
        
        # Pruning: if we can't beat the best length, return
        if len(seq) + (N - start_val + 1) <= best_len:
            return
        
        # Time check
        if time.time() > deadline:
            return
        
        # Iterate candidates
        for x in range(start_val, N + 1):
            # Calculate new suffix sums
            new_suffix_sums = [x]
            valid = True
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                if new_suffix_sums[-1] in sums_set:
                    valid = False
                    break
            
            if x in sums_set:
                valid = False
            
            if valid:
                new_sums_set = sums_set.union(new_suffix_sums)
                new_seq = seq + [x]
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
                
                # Recurse
                dfs(x + 1, new_seq, new_suffix_sums, new_sums_set, depth + 1)

    dfs(1, [], [], set(), 0)
    
    # If we have time, try random restarts with different greedy orderings
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            
            # Try a different greedy strategy: start from a random offset
            random.seed(int(time.time() * 1000) % 10000)
            values = []
            sums = set()
            # Try adding numbers in a slightly perturbed order
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            # But we need strictly increasing sequence, so we can't shuffle arbitrarily
            # Instead, we can try skipping some numbers and picking others
            # Actually, for strictly increasing, we must pick in order.
            # So random restarts don't help much for greedy.
            # Instead, let's try a local search: take the best_seq and try to swap/insert.
            # But that's complex. Let's just run DFS with a different starting point.
            # Since DFS already explores from 1, maybe we can start from a higher value?
            # No, that would likely give shorter sequences.
            
            # Let's try a different approach: use the greedy result as a seed
            # and try to extend it or modify it.
            # For now, just break out of the loop.
            break
    
    return best_seq
