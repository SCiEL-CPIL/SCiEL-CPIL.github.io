import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def is_valid_addition(seq, x, sums):
        total = x
        if total in sums:
            return False, None
        new_sums = [total]
        for a in reversed(seq):
            total += a
            if total in sums:
                return False, None
            new_sums.append(total)
        return True, set(new_sums)

    def greedy(strategy):
        seq = []
        sums = set()
        available = set(range(1, 257))
        
        while available:
            if strategy == 'smallest':
                candidates = sorted(available)
            elif strategy == 'largest':
                candidates = sorted(available, reverse=True)
            elif strategy == 'random':
                candidates = list(available)
                random.shuffle(candidates)
            elif strategy == 'top_random':
                # Pick from top 10 largest
                sorted_avail = sorted(available, reverse=True)
                k = min(10, len(sorted_avail))
                candidates = sorted_avail[:k]
                random.shuffle(candidates)
            elif strategy == 'bottom_random':
                # Pick from bottom 10 smallest
                sorted_avail = sorted(available)
                k = min(10, len(sorted_avail))
                candidates = sorted_avail[:k]
                random.shuffle(candidates)
            else:
                candidates = list(available)
                random.shuffle(candidates)
            
            added = False
            for x in candidates:
                valid, new_sums = is_valid_addition(seq, x, sums)
                if valid:
                    seq.append(x)
                    sums.update(new_sums)
                    available.remove(x)
                    added = True
                    break
            
            if not added:
                break
        
        return seq

    best_seq = []
    
    # Deterministic
    for strat in ['smallest', 'largest']:
        seq = greedy(strat)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # Stochastic
    strategies = ['random', 'top_random', 'bottom_random']
    
    while time.time() < deadline:
        strat = random.choice(strategies)
        seq = greedy(strat)
        if len(seq) > len(best_seq):
            best_seq = seq

    return best_seq
