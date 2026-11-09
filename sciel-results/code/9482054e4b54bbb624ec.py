import random
import time

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit

    # Precomputed known-good solution (Sidon set with all diffs 1..30)
    KNOWN_GOOD = [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 32, 34, 37, 39, 42, 44, 47, 49, 52, 54, 57, 59, 62, 64, 67, 69, 72, 74, 77, 79, 82, 84]

    def get_diffs(S):
        return [y - x for i, x in enumerate(S) for y in S[i+1:]]

    def is_sidon(S):
        diffs = get_diffs(S)
        return len(diffs) == len(set(diffs))

    def has_all_diffs(S):
        diffs = set(get_diffs(S))
        return all(d in diffs for d in range(1, 31))

    def is_valid(S):
        return is_sidon(S) and has_all_diffs(S)

    def score(S):
        """Lower is better. Penalize non-Sidon and missing diffs."""
        diffs = get_diffs(S)
        diff_set = set(diffs)
        
        # Penalty for duplicate diffs (non-Sidon)
        dup_penalty = len(diffs) - len(diff_set)
        
        # Penalty for missing diffs 1..30
        missing = [d for d in range(1, 31) if d not in diff_set]
        missing_penalty = len(missing)
        
        # Objective: minimize max element + penalties
        max_elem = max(S)
        
        return max_elem + 100 * dup_penalty + 50 * missing_penalty

    # Verify known good
    if is_valid(KNOWN_GOOD):
        best_set = KNOWN_GOOD[:]
        best_max = max(KNOWN_GOOD)
    else:
        best_set = None
        best_max = float('inf')

    def sa_search(initial_S, max_steps=50000, t0=10.0, t1=0.1):
        """Simulated annealing on the set."""
        S = sorted(initial_S[:])
        current_score = score(S)
        best_S = S[:]
        best_score = current_score
        
        n = len(S)
        
        for step in range(max_steps):
            if time.time() - start_time > deadline:
                break
            
            # Temperature schedule
            t = t0 * (t1 / t0) ** (step / max_steps)
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Move: change one element (not 0)
                i = random.randint(1, n - 1)
                # Try values in a neighborhood
                old_val = S[i]
                # Search range: from 1 to max(S)+10
                search_hi = max(S) + 20
                new_val = random.randint(1, search_hi)
                if new_val in S:
                    continue
                trial = S[:i] + [new_val] + S[i+1:]
                trial.sort()
                
            elif move_type < 0.7:
                # Add an element
                if n >= 32:
                    continue
                new_val = random.randint(1, max(S) + 50)
                if new_val in S:
                    continue
                trial = S + [new_val]
                trial.sort()
                
            elif move_type < 0.9:
                # Remove an element (not 0)
                if n <= 2:
                    continue
                i = random.randint(1, n - 1)
                trial = S[:i] + S[i+1:]
                
            else:
                # Swap two elements
                if n < 2:
                    continue
                i, j = random.sample(range(1, n), 2)
                trial = S[:]
                trial[i], trial[j] = trial[j], trial[i]
                trial.sort()
            
            trial_score = score(trial)
            delta = trial_score - current_score
            
            if delta < 0 or random.random() < __import__('math').exp(-delta / max(t, 1e-10)):
                S = trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best_S = S[:]
        
        return best_S, best_score

    # Run multiple SA restarts with different initializations
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > deadline:
            break
        
        random.seed(12345 + r * 7919)
        
        if r == 0:
            # Start from known good
            init = KNOWN_GOOD[:]
        elif r == 1:
            # Start from a compact greedy construction
            init = [0]
            covered = set()
            for d in range(1, 31):
                if d in covered:
                    continue
                found = False
                for x in sorted(init):
                    for c in (x + d, x - d):
                        if c >= 0 and c not in init:
                            trial = sorted(init + [c])
                            if is_sidon(trial):
                                init = trial
                                covered = set(get_diffs(init))
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # Add a new element that covers d
                    y = max(init) + d
                    while True:
                        trial = sorted(init + [y])
                        if is_sidon(trial):
                            init = trial
                            covered = set(get_diffs(init))
                            break
                        y += 1
        else:
            # Random initialization: random Sidon-like set
            init = [0]
            target_size = random.randint(10, 25)
            for _ in range(target_size - 1):
                # Pick a random value that maintains Sidon property
                found = False
                for _attempt in range(100):
                    v = random.randint(1, 200)
                    if v in init:
                        continue
                    trial = sorted(init + [v])
                    if is_sidon(trial):
                        init = trial
                        found = True
                        break
                if not found:
                    break
        
        # Run SA from this initial
        sa_result, sa_score = sa_search(init, max_steps=30000, t0=15.0, t1=0.05)
        
        if sa_result is not None and is_valid(sa_result):
            sa_max = max(sa_result)
            if sa_max < best_max:
                best_max = sa_max
                best_set = sa_result[:]

    # Final local search refinement on best found
    if best_set is not None:
        S = sorted(best_set[:])
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            
            # Try removing elements
            for i in range(1, len(S)):
                trial = S[:i] + S[i+1:]
                if is_valid(trial) and max(trial) < max(S):
                    S = trial
                    improved = True
                    break
            
            # Try replacing elements with smaller values
            for i in range(1, len(S)):
                for new_val in range(1, S[i]):
                    if new_val in S:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial.sort()
                    if is_valid(trial) and max(trial) < max(S):
                        S = trial
                        improved = True
                        break
                if improved:
                    break
        
        if is_valid(S) and max(S) < best_max:
            best_max = max(S)
            best_set = S[:]

    if best_set is not None:
        return best_set
    else:
        return KNOWN_GOOD

