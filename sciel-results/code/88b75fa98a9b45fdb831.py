import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Greedy initialization to get a good lower bound
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
    
    # Helper to check if we can append x to a given sequence and sums set
    # Returns True if valid
    def can_append(values, sums, x):
        total = x
        if total in sums:
            return False
        for a in reversed(values):
            total += a
            if total in sums:
                return False
        return True

    # Helper to update sums set after appending x
    def update_sums(sums, values, x):
        new_sums = set(sums)
        total = x
        new_sums.add(total)
        for a in reversed(values):
            total += a
            new_sums.add(total)
        return new_sums

    # 2. Randomized Greedy Search
    # We try to build a sequence by iterating through a random permutation of 1..N.
    # We only consider x if it is larger than the last element in the current sequence.
    # This ensures the sequence remains strictly increasing.
    
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        current_seq = []
        current_sums = set()
        
        for x in order:
            if current_seq and x <= current_seq[-1]:
                continue
                
            if can_append(current_seq, current_sums, x):
                current_seq.append(x)
                current_sums = update_sums(current_sums, current_seq[:-1], x)
                
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
    return best_seq
