"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with greedy solution as initial state
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return set(y - x for i, x in enumerate(s) for y in s[i + 1:])
    
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(get_diffs(trial)) == len(trial) * (len(trial) - 1) // 2:
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(get_diffs(trial)) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    break
                y += 1
    
    best_set = list(a)
    best_score = max(a)
    
    # Simulated annealing to improve
    temperature = 500.0
    cooling_rate = 0.9995
    
    while time.time() < deadline and temperature > 1.0:
        current = list(best_set)
        current_score = max(current)
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3 and len(current) < 32:
            # Add a new element
            max_val = max(current)
            new_val = random.randint(1, max_val + 100)
            if new_val not in current:
                trial = sorted(current + [new_val])
                trial_diffs = get_diffs(trial)
                if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2:
                    # Check all required differences are present
                    if all(d in trial_diffs for d in range(1, 31)):
                        trial_score = max(trial)
                        if trial_score < current_score or random.random() < min(1.0, temperature / max(1, current_score - trial_score)):
                            current = trial
                            current_score = trial_score
                            if current_score < best_score:
                                best_set = list(current)
                                best_score = current_score
        elif move_type < 0.6:
            # Remove an element (if we have more than needed)
            if len(current) > 31:
                idx = random.randint(0, len(current) - 1)
                if current[idx] != 0:  # Keep 0
                    trial = current[:idx] + current[idx+1:]
                    trial_diffs = get_diffs(trial)
                    if all(d in trial_diffs for d in range(1, 31)):
                        trial_score = max(trial)
                        if trial_score < current_score or random.random() < min(1.0, temperature / max(1, current_score - trial_score)):
                            current = trial
                            current_score = trial_score
                            if current_score < best_score:
                                best_set = list(current)
                                best_score = current_score
        elif move_type < 0.85:
            # Shift an element
            if len(current) > 1:
                idx = random.randint(1, len(current) - 1)  # Don't move 0
                old_val = current[idx]
                shift = random.randint(-20, 20)
                new_val = old_val + shift
                if new_val > 0 and new_val not in current:
                    trial = list(current)
                    trial[idx] = new_val
                    trial = sorted(trial)
                    trial_diffs = get_diffs(trial)
                    if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2 and all(d in trial_diffs for d in range(1, 31)):
                        trial_score = max(trial)
                        if trial_score < current_score or random.random() < min(1.0, temperature / max(1, current_score - trial_score)):
                            current = trial
                            current_score = trial_score
                            if current_score < best_score:
                                best_set = list(current)
                                best_score = current_score
        else:
            # Swap two elements
            if len(current) > 2:
                idx1, idx2 = random.sample(range(1, len(current)), 2)
                trial = list(current)
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                trial = sorted(trial)
                trial_diffs = get_diffs(trial)
                if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2 and all(d in trial_diffs for d in range(1, 31)):
                    trial_score = max(trial)
                    if trial_score < current_score or random.random() < min(1.0, temperature / max(1, current_score - trial_score)):
                        current = trial
                        current_score = trial_score
                        if current_score < best_score:
                            best_set = list(current)
                            best_score = current_score
        
        temperature *= cooling_rate
    
    # Final verification
    final_diffs = get_diffs(best_set)
    if (len(final_diffs) == len(best_set) * (len(best_set) - 1) // 2 and 
        all(d in final_diffs for d in range(1, 31)) and
        0 in best_set and
        len(best_set) <= 32 and
        max(best_set) <= 10**6):
        return best_set
    else:
        return a  # Fallback to greedy solution

# EVOLVE-BLOCK-END
