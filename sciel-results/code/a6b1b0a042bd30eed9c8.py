import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    candidates = list(range(2, 129))
    
    def evaluate(seq):
        """
        Returns (is_valid, collision_count)
        Uses integer products.
        """
        if not seq:
            return True, 0
        n = len(seq)
        seen = set()
        collisions = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in seen:
                    collisions += 1
                else:
                    seen.add(prod)
        return (collisions == 0), collisions

    def greedy_init():
        seq = []
        for x in candidates:
            test_seq = seq + [x]
            is_valid, _ = evaluate(test_seq)
            if is_valid:
                seq.append(x)
        return seq

    best_seq = greedy_init()
    best_score = len(best_seq)
    
    def score(seq):
        is_valid, coll = evaluate(seq)
        if is_valid:
            return len(seq)
        else:
            return len(seq) - 10 * coll

    current_seq = best_seq[:]
    current_score = score(current_seq)
    current_set = set(current_seq)
    
    T = 50.0
    T_min = 0.5
    cooling = 0.9995
    
    # Run SA
    while time.time() < deadline:
        if T < T_min:
            break
            
        r = random.random()
        if r < 0.3:
            # Add
            available = [x for x in candidates if x not in current_set]
            if not available:
                continue
            x = random.choice(available)
            new_seq = current_seq + [x]
            new_seq.sort()
        elif r < 0.6:
            # Remove
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        else:
            # Replace
            if not current_seq:
                continue
            available = [x for x in candidates if x not in current_set]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = random.choice(available)
            new_seq = current_seq[:idx] + [x] + current_seq[idx+1:]
            new_seq.sort()
            
        new_score = score(new_seq)
        
        delta = new_score - current_score
        if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_seq = new_seq
            current_set = set(current_seq)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
        
        T *= cooling

    # Final Polish: Try to insert any remaining numbers
    final_seq = best_seq[:]
    final_set = set(final_seq)
    for x in candidates:
        if x in final_set:
            continue
        # Find position to maintain sorted order
        pos = 0
        while pos < len(final_seq) and final_seq[pos] < x:
            pos += 1
        test_seq = final_seq[:pos] + [x] + final_seq[pos:]
        is_valid, _ = evaluate(test_seq)
        if is_valid:
            final_seq = test_seq
            final_set.add(x)
            
    return final_seq
