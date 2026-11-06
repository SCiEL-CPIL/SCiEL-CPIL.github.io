import time
import random

def construct():
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    time_limit = 140.0
    
    # Helper to calculate score
    def get_score(s_set):
        s_list = sorted(s_set)
        diffs = set()
        # Find all differences present
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                diffs.add(s_list[j] - s_list[i])
        
        good_diffs = set()
        for d in diffs:
            # Check if there exists a such that a, a+d, a+2d in s_set
            found = False
            for a in s_set:
                if (a + d) in s_set and (a + 2 * d) in s_set:
                    found = True
                    break
            if found:
                good_diffs.add(d)
        
        return len(good_diffs)

    best_score = 0
    best_set = set(range(1, 33)) # Fallback
    
    # Try to beat the interval seed
    interval_set = set(range(1, 33))
    best_score = get_score(interval_set)
    best_set = set(interval_set)
    
    random.seed(42)
    
    # Simulated Annealing
    current_set = set(interval_set)
    current_score = best_score
    
    T_start = 10.0
    T_end = 0.1
    steps = 0
    
    while time.time() - start_time < time_limit:
        T = T_start * (T_end / T_start) ** (steps / 10000.0)
        if T < T_end:
            T = T_end
        
        # Pick a random element to remove
        if not current_set:
            break
        rem_elem = random.choice(list(current_set))
        new_set = set(current_set)
        new_set.remove(rem_elem)
        
        # Pick a random element to add
        add_elem = random.randint(1, 512)
        if add_elem in new_set:
            continue
        
        new_set.add(add_elem)
        
        if len(new_set) != 32:
            continue
            
        new_score = get_score(new_set)
        
        delta_score = new_score - current_score
        
        if delta_score >= 0:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            if random.random() < pow(2.718281828, delta_score / T):
                current_set = new_set
                current_score = new_score
        
        steps += 1
        
        if steps % 100 == 0:
            if time.time() - start_time > time_limit - 5:
                break

    # Final polish: Hill climbing
    for _ in range(50):
        improved = False
        for _ in range(1000):
            if time.time() - start_time > time_limit - 2:
                break
            rem_elem = random.choice(list(current_set))
            add_elem = random.randint(1, 512)
            if add_elem in current_set:
                continue
            new_set = set(current_set)
            new_set.remove(rem_elem)
            new_set.add(add_elem)
            new_score = get_score(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
        if not improved:
            break

    return list(best_set)
