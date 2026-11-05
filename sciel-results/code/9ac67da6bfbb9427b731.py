import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    # 55s safety margin for 60s limit
    deadline = start_time + 55.0
    
    def greedy_init():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy_init()
    best_len = len(best_seq)
    
    def get_penalty(seq):
        if not seq:
            return 0
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        # Count duplicates
        # A sum appearing k times contributes k-1 to the penalty
        c = Counter(sums)
        penalty = 0
        for count in c.values():
            if count > 1:
                penalty += count - 1
        return penalty

    current_seq = best_seq[:]
    current_penalty = get_penalty(current_seq)
    
    # Simulated Annealing / Iterated Local Search parameters
    T = 5.0
    T_min = 0.1
    decay = 0.995
    steps_since_improvement = 0
    max_steps_no_improve = 500
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        new_seq = current_seq[:]
        
        if move_type < 0.6:
            # Add a random unused number
            used = set(current_seq)
            candidates = [x for x in range(1, 257) if x not in used]
            if candidates:
                val = random.choice(candidates)
                # Insert in sorted position
                pos = 0
                while pos < len(new_seq) and new_seq[pos] < val:
                    pos += 1
                new_seq.insert(pos, val)
        elif move_type < 0.8:
            # Remove a random number
            if current_seq:
                idx = random.randint(0, len(current_seq) - 1)
                new_seq.pop(idx)
        else:
            # Replace a random number with a different unused number
            if current_seq:
                idx = random.randint(0, len(current_seq) - 1)
                old_val = current_seq[idx]
                used = set(current_seq)
                candidates = [x for x in range(1, 257) if x not in used and x != old_val]
                if candidates:
                    new_val = random.choice(candidates)
                    new_seq.pop(idx)
                    pos = 0
                    while pos < len(new_seq) and new_seq[pos] < new_val:
                        pos += 1
                    new_seq.insert(pos, new_val)
        
        new_penalty = get_penalty(new_seq)
        
        # Acceptance criteria
        accepted = False
        if new_penalty < current_penalty:
            accepted = True
        elif new_penalty == current_penalty:
            if len(new_seq) > len(current_seq):
                accepted = True
            else:
                # Accept with probability 0.5 to explore same-penalty states
                if random.random() < 0.5:
                    accepted = True
        else:
            # new_penalty > current_penalty
            delta = new_penalty - current_penalty
            # Accept with probability exp(-delta / T)
            # Use math.exp for accuracy
            prob = 2.71828 ** (-delta / T)
            if random.random() < prob:
                accepted = True
        
        if accepted:
            current_seq = new_seq
            current_penalty = new_penalty
            
            if current_penalty == 0 and len(current_seq) > best_len:
                best_seq = current_seq[:]
                best_len = len(current_seq)
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
        
        # Update Temperature
        T *= decay
        if T < T_min:
            T = T_min
        
        # Restart logic
        if steps_since_improvement > max_steps_no_improve:
            T = 5.0
            # Jump back to best known valid state
            current_seq = best_seq[:]
            current_penalty = 0
            steps_since_improvement = 0

    return best_seq
