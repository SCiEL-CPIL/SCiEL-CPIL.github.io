import time
import random

def _check_valid(seq):
    """Check if seq is strictly increasing in [2,128] and all consecutive block products are distinct."""
    if not seq:
        return False
    n = len(seq)
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

def _greedy_init():
    """Build an initial valid sequence using greedy method."""
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _try_insert(seq, x):
    """Try to insert x into seq at any valid position. Return new seq if valid, else None."""
    if x in seq:
        return None
    n = len(seq)
    for pos in range(n + 1):
        if pos > 0 and seq[pos-1] >= x:
            continue
        if pos < n and seq[pos] <= x:
            continue
        new_seq = seq[:pos] + [x] + seq[pos:]
        if _check_valid(new_seq):
            return new_seq
    return None

def construct():
    """
    Use Greedy initialization followed by Randomized Local Search (Hill Climbing with Deletions).
    """
    start_time = time.time()
    time_limit = 55.0

    current_seq = _greedy_init()
    best_seq = list(current_seq)
    best_len = len(best_seq)
    
    # First, try pure insertion hill climbing
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        missing = [x for x in range(2, 129) if x not in current_seq]
        for x in missing:
            new_seq = _try_insert(current_seq, x)
            if new_seq:
                if len(new_seq) > best_len:
                    best_seq = new_seq
                    best_len = len(best_seq)
                current_seq = new_seq
                improved = True
                break

    # Now, try delete-insert moves to escape local optima
    # We will perform a number of iterations
    iterations = 0
    max_iterations = 10000
    
    while time.time() - start_time < time_limit and iterations < max_iterations:
        iterations += 1
        
        # Randomly delete an element
        if len(current_seq) <= 1:
            break
        del_idx = random.randint(0, len(current_seq) - 1)
        candidate_seq = current_seq[:del_idx] + current_seq[del_idx+1:]
        
        # Try to insert any missing elements into the candidate
        missing = [x for x in range(2, 129) if x not in candidate_seq]
        # Shuffle missing to randomize order of insertion
        random.shuffle(missing)
        
        new_seq = list(candidate_seq)
        for x in missing:
            inserted = _try_insert(new_seq, x)
            if inserted:
                new_seq = inserted
        
        # If the new sequence is longer than the current best, update
        if len(new_seq) > best_len:
            best_seq = new_seq
            best_len = len(best_seq)
            current_seq = new_seq
        elif len(new_seq) > len(current_seq):
            # Accept if it improves current state, even if not global best
            current_seq = new_seq
        # Else, we might want to accept with some probability (SA-like) but let's stick to Hill Climbing for simplicity and speed
        # Or just reject.

    return best_seq
