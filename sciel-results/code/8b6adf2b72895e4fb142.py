import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        for d in range(1, 256):
            # Shift mask right by d and 2d
            # If the bitwise AND of the original, shifted by d, and shifted by 2d is non-zero,
            # then there is at least one triplet.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        # Iterate through bits 1 to 512
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def greedy_construct():
        """
        Greedily construct a set by adding elements that maximize the number of new 3-term APs.
        """
        current_set = set()
        current_mask = 0
        current_score = 0
        
        # Candidate pool: all numbers 1..512
        # To speed up, we might just pick from a random subset or all. 
        # 512 is small enough.
        candidates = list(range(1, MAX_VAL + 1))
        random.shuffle(candidates)
        
        # We want to pick N elements.
        # Instead of checking all 512 at each step (32*512*256 ops ~ 4M, which is fine),
        # we can just iterate through candidates and pick the best one.
        
        for _ in range(N):
            best_candidate = -1
            best_new_score = -1
            
            # We can restrict candidates to a subset to save time? 
            # Let's try all candidates for the first few, then random subset.
            # Actually, 32 iterations * 512 candidates * score_cost.
            # score_cost is ~256 bit ops. 32*512*256 ~ 4M ops. This is fast in Python?
            # Maybe a bit slow. Let's sample 50 candidates per step.
            
            sample_size = min(100, len(candidates))
            sample = random.sample(candidates, sample_size)
            
            for c in sample:
                if c in current_set:
                    continue
                new_mask = current_mask | (1 << c)
                new_score = score_mask(new_mask)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_candidate = c
            
            if best_candidate != -1:
                current_set.add(best_candidate)
                current_mask |= (1 << best_candidate)
                current_score = best_new_score
                # Remove from candidates to avoid re-evaluation? 
                # Not strictly necessary if we check `if c in current_set`, but good for perf.
                # We don't remove from the master list, just rely on the set check.
                
        return list(current_set)

    # Generate initial candidates
    candidates = []

    # Random starts
    for _ in range(20):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)

    # Structured starts
    # Arithmetic progression
    candidates.append(list(range(1, 1 + 31 * 16, 16)))
    # Dense at start
    candidates.append(list(range(1, 33)))
    # Dense at end
    candidates.append(list(range(481, 513)))
    
    # Greedy starts
    candidates.append(greedy_construct())
    candidates.append(greedy_construct())

    # Evaluate initial candidates
    best_score = -1
    best_list = None
    best_mask = 0

    for lst in candidates:
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # Simulated Annealing with restarts
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)

    # Parameters
    T_init = 15.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Restart logic
    restart_interval = 10.0 # seconds
    next_restart = start_time + restart_interval
    no_improve_steps = 0
    max_no_improve = 500 # steps

    while time.time() < deadline:
        # Check for restart
        if time.time() > next_restart or no_improve_steps > max_no_improve:
            # Restart cooling schedule
            T = T_init
            next_restart = time.time() + restart_interval
            no_improve_steps = 0
            # Optionally, perturb the best solution slightly to escape local optima
            # But SA with high T should handle this.
            # Let's keep the current state if it's good, or reset to best found?
            # Resetting to best found is safer.
            current_list = best_list[:]
            current_score = best_score
            current_mask = best_mask
            current_set = set(current_list)
            continue

        # Generate a neighbor
        # 50% chance: Local move (small perturbation)
        # 50% chance: Global move (random replacement)
        
        if random.random() < 0.5:
            # Local move: Pick an element, try to move it to a nearby unused value
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            
            # Try a few nearby values
            found_new_val = False
            for _ in range(10):
                offset = random.randint(-15, 15)
                if offset == 0: continue
                new_val = old_val + offset
                if 1 <= new_val <= MAX_VAL and new_val not in current_set:
                    found_new_val = True
                    break
            
            if not found_new_val:
                # Fallback to global move
                while True:
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in current_set:
                        continue
                    break
        else:
            # Global move
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            while True:
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in current_set:
                    continue
                break

        # Calculate delta score
        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
        new_score = score_mask(new_mask)
        delta = new_score - current_score

        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)

        if accept:
            # Update state
            current_list[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            current_score = new_score
            current_mask = new_mask

            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                best_mask = current_mask
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_list
# EVOLVE-BLOCK-END
