import time
import random

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin before 60s
    
    # 1. Greedy Initialization
    def greedy_construct():
        values = []
        products = set()
        for x in range(2, 129):
            valid = True
            p = x
            tails = [p]
            for a in reversed(values):
                p *= a
                tails.append(p)
                if p in products:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                products.update(tails)
        return values

    # Helper to check validity of a sequence
    def is_valid(sequence):
        products = set()
        n = len(sequence)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                if p in products:
                    return False
                products.add(p)
        return True

    # Get initial sequence
    seq = greedy_construct()
    best_seq = seq
    best_len = len(seq)
    
    # Helper to get unused numbers
    def get_unused(seq):
        used = set(seq)
        return [x for x in range(2, 129) if x not in used]

    # Try to improve by 1-out/2-in
    def try_1out_2in(seq, unused, time_limit=0.5):
        n = len(seq)
        # Shuffle unused to randomize
        unused_copy = unused[:]
        random.shuffle(unused_copy)
        
        # Sort candidates by value? Or random? Random is good for exploration.
        # But we need to respect bounds.
        
        start = time.time()
        for i in range(n):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            candidates = [x for x in unused_copy if left_bound < x < right_bound]
            
            if len(candidates) < 2:
                continue
            
            # Limit number of pairs to try per position to avoid timeout
            # If candidates is large, sample
            max_pairs = 50
            if len(candidates) > 20:
                # Sample pairs
                pairs = []
                for _ in range(max_pairs):
                    x = random.choice(candidates)
                    y = random.choice(candidates)
                    if x != y:
                        pairs.append((x, y))
                # Remove duplicates
                pairs = list(set(pairs))
            else:
                pairs = [(x, y) for idx_x in range(len(candidates)) for idx_y in range(idx_x+1, len(candidates))]
            
            for x, y in pairs:
                new_seq = seq[:i] + [x, y] + seq[i+1:]
                if is_valid(new_seq):
                    return new_seq
            
            if time.time() - start > time_limit:
                break
                
        return None

    # Try to improve by 2-out/2-in (same length, rearrangement)
    def try_2out_2in(seq, unused, time_limit=0.5):
        n = len(seq)
        if n < 2:
            return None
        
        start = time.time()
        for i in range(n - 1):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+2] if i+2 < n else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            
            if len(candidates) < 2:
                continue
            
            # Try a few random pairs
            for _ in range(20):
                x = random.choice(candidates)
                y = random.choice(candidates)
                if x == y:
                    continue
                new_seq = seq[:i] + [x, y] + seq[i+2:]
                if is_valid(new_seq):
                    return new_seq
            
            if time.time() - start > time_limit:
                break
                
        return None

    # Main loop with randomization and restarts
    while time.time() - start_time < timeout:
        unused = get_unused(seq)
        
        # Try 1-out/2-in
        new_seq = try_1out_2in(seq, unused)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
            continue
        
        # Try 2-out/2-in
        new_seq = try_2out_2in(seq, unused)
        if new_seq is not None:
            seq = new_seq
            continue
        
        # If no improvement, try a random perturbation (shake)
        # Remove a random element and try to insert a random one
        if len(seq) > 5:
            i = random.randint(0, len(seq)-1)
            removed = seq[i]
            seq_without = seq[:i] + seq[i+1:]
            
            # Try to insert a new element
            left_bound = seq_without[i-1] if i > 0 else 1
            right_bound = seq_without[i] if i < len(seq_without) else 129
            
            candidates = [x for x in range(2, 129) if left_bound < x < right_bound and x not in set(seq_without)]
            
            if candidates:
                x = random.choice(candidates)
                new_seq = seq_without[:i] + [x] + seq_without[i:]
                if is_valid(new_seq):
                    seq = new_seq
                    continue
                else:
                    # Try to re-insert the removed element or another
                    # Just reset to previous state if this fails? 
                    # For now, just accept the shake if valid, otherwise keep seq
                    pass
        
        # If stuck for a while, restart from greedy with a different seed?
        # Or just break to avoid infinite loop of shakes
        # Let's add a counter for stuck iterations
        # Simple approach: if no improvement in last 100 tries, restart
        # We don't have a counter, so let's just break if time is low
        if time.time() - start_time > timeout - 5:
            break
            
    return best_seq
