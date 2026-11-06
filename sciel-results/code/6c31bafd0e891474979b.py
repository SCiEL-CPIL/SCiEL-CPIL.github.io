"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_construct():
    """Fast greedy: add x if it keeps all sums at count <= 2."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        # New sums: x+y for y in chosen, plus 2*x
        new_sums = [x + y for y in chosen]
        if chosen:
            new_sums.append(2 * x)
        else:
            new_sums.append(2 * x)
        
        # Check if adding x would cause any sum to exceed count 2
        ok = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    return chosen


def _count_violations(chosen):
    """Return number of sums that have count > 2."""
    if not chosen:
        return 0
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return sum(1 for c in counts.values() if c > 2)


def _objective(chosen):
    """Objective: maximize size, minimize violations. Return (size, -violations)."""
    if not chosen:
        return (0, 0)
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    violations = sum(1 for c in counts.values() if c > 2)
    return (len(chosen), -violations)


def _is_valid(chosen):
    """Check if chosen is a valid B2[2] set."""
    if not chosen:
        return True
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return all(c <= 2 for c in counts.values())


def _local_search(start_set, deadline, seed=42):
    """Simulated annealing / local search to improve a B2[2] set."""
    rng = random.Random(seed)
    chosen = list(start_set)
    
    # Compute initial violation count
    def compute_violations(chosen):
        if len(chosen) < 2:
            return 0
        counts = {}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        return sum(1 for c in counts.values() if c > 2)
    
    current_violations = compute_violations(chosen)
    current_size = len(chosen)
    best_set = list(chosen)
    best_size = current_size
    best_violations = current_violations
    
    # Temperature schedule
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    step = 0
    while T > min_T:
        if time.time() > deadline - 5:
            break
        
        # Random move: add, remove, or swap
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random element not in chosen
            chosen_set = set(chosen)
            candidates = [x for x in range(1, 1001) if x not in chosen_set]
            if not candidates:
                break
            x = rng.choice(candidates)
            new_chosen = chosen + [x]
        elif move_type < 0.7:
            # Remove a random element
            if not chosen:
                continue
            idx = rng.randrange(len(chosen))
            new_chosen = chosen[:idx] + chosen[idx+1:]
        else:
            # Swap: remove one, add another
            if not chosen:
                continue
            idx = rng.randrange(len(chosen))
            removed = chosen[idx]
            chosen_set = set(chosen)
            candidates = [x for x in range(1, 1001) if x not in chosen_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_chosen = chosen[:idx] + [x] + chosen[idx+1:]
        
        new_violations = compute_violations(new_chosen)
        new_size = len(new_chosen)
        
        # Acceptance: prefer larger valid sets
        # Score: size * 1000 - violations * 10
        cur_score = current_size * 1000 - current_violations * 10
        new_score = new_size * 1000 - new_violations * 10
        delta_score = new_score - cur_score
        
        if delta_score >= 0 or rng.random() < np.exp(delta_score / T):
            chosen = new_chosen
            current_violations = new_violations
            current_size = new_size
            
            if new_violations == 0 and new_size > best_size:
                best_size = new_size
                best_violations = 0
                best_set = list(chosen)
            elif new_violations < best_violations or (new_violations == best_violations and new_size > best_size):
                best_size = new_size
                best_violations = new_violations
                best_set = list(chosen)
        
        T *= cooling
        step += 1
    
    # Return best valid set found, or best set overall
    if best_violations == 0:
        return best_set
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    greedy_set = _greedy_construct()
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    if time.time() > deadline - 10:
        return best_set
    
    # Phase 2: Local search with simulated annealing
    # Try multiple restarts with different seeds
    rng_master = random.Random(12345)
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        seed = rng_master.randint(0, 10**9)
        
        # Create a perturbed starting point from greedy
        start_set = list(greedy_set)
        # Randomly remove a few elements to create diversity
        if start_set:
            k = min(len(start_set), rng_master.randint(1, 10))
            for _ in range(k):
                if start_set:
                    idx = rng_master.randrange(len(start_set))
                    start_set.pop(idx)
        
        result = _local_search(start_set, deadline, seed=seed)
        
        if len(result) > best_size and _is_valid(result):
            best_size = len(result)
            best_set = list(result)
        elif len(result) > best_size:
            # Check if it's valid
            if _is_valid(result):
                best_size = len(result)
                best_set = list(result)
    
    return best_set

# EVOLVE-BLOCK-END
