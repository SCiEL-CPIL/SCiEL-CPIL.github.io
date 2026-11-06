import time
import random

def _check_valid(seq):
    """Check if seq is strictly increasing in [2,128] and all consecutive block products are distinct."""
    if not seq:
        return False
    n = len(seq)
    # seq is assumed sorted and unique
    for i in range(n):
        if seq[i] < 2 or seq[i] > 128:
            return False
        if i > 0 and seq[i] <= seq[i-1]:
            return False
            
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _random_valid_seq(max_len=10):
    """Generate a random valid sequence of small length to start SA."""
    # Try to build a small valid sequence randomly
    seq = []
    attempts = 0
    while len(seq) < max_len and attempts < 1000:
        attempts += 1
        x = random.randint(2, 128)
        if x in seq:
            continue
        # Insert in sorted order
        new_seq = seq + [x]
        new_seq.sort()
        if _check_valid(new_seq):
            seq = new_seq
    return seq

def construct():
    """
    Use Simulated Annealing to maximize the length of the sequence.
    State: A sorted list of integers.
    Moves: Add, Remove, Replace.
    """
    start_time = time.time()
    time_limit = 55.0
    
    # Initialize with a small valid sequence
    current_seq = _random_valid_seq(5)
    if not current_seq:
        # Fallback to simple greedy if random init fails
        current_seq = [2, 3, 5, 7, 11]
        if not _check_valid(current_seq):
            current_seq = [2, 3, 5]
            
    best_seq = list(current_seq)
    best_len = len(best_seq)
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.995
    # Number of iterations per temperature step
    iterations_per_step = 50
    
    all_values = set(range(2, 129))
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        for _ in range(iterations_per_step):
            if time.time() - start_time > time_limit:
                break
                
            move_type = random.choice(['add', 'remove', 'replace'])
            new_seq = None
            
            if move_type == 'add':
                missing = all_values - set(current_seq)
                if missing:
                    x = random.choice(list(missing))
                    # Insert into sorted position
                    new_seq = current_seq + [x]
                    new_seq.sort()
            elif move_type == 'remove':
                if len(current_seq) > 1:
                    idx = random.randint(0, len(current_seq) - 1)
                    new_seq = current_seq[:idx] + current_seq[idx+1:]
            elif move_type == 'replace':
                if current_seq:
                    idx = random.randint(0, len(current_seq) - 1)
                    old_val = current_seq[idx]
                    missing = all_values - set(current_seq)
                    if missing:
                        new_val = random.choice(list(missing))
                        new_seq = current_seq[:idx] + [new_val] + current_seq[idx+1:]
                        new_seq.sort()
                        # Ensure strict increase (sort handles order, but we must ensure no duplicates)
                        # Since new_val is from missing, it's not in current_seq, so no duplicates.
                        
            if new_seq is None:
                continue
                
            # Check validity
            is_valid_new = _check_valid(new_seq)
            is_valid_cur = _check_valid(current_seq)
            
            # Objective: Maximize length. 
            # If invalid, length is -infinity (or 0 with penalty)
            # Let's use score = len(seq) if valid, else -1000
            score_new = len(new_seq) if is_valid_new else -1000
            score_cur = len(current_seq) if is_valid_cur else -1000
            
            delta_score = score_new - score_cur
            
            if delta_score > 0:
                # Accept better solution
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(best_seq)
            else:
                # Accept worse solution with probability exp(delta_score / T)
                # If both are invalid, delta is 0, prob 1.
                # If cur valid, new invalid, delta is large negative, prob ~ 0.
                if T > 0:
                    prob = min(1.0, 2.718281828 ** (delta_score / T))
                    if random.random() < prob:
                        current_seq = new_seq
                        # Don't update best if invalid
                        if is_valid_new and len(current_seq) > best_len:
                            best_seq = current_seq
                            best_len = len(best_seq)
                            
        T *= alpha
        
    return best_seq
