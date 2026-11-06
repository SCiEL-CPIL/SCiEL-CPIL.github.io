"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _largest_sidon_subset(s):
    """Find the largest subset of s with all pair sums (including diagonals) unique."""
    s = sorted(s)
    n = len(s)
    best = 0
    # Try all subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        ok = False
                        break
                    sums.add(sm)
                if not ok:
                    break
            if ok:
                best = size
                found = True
                break
        if found:
            break
    return best

def _evaluate(s):
    """Return the largest Sidon subset size for set s."""
    return _largest_sidon_subset(s)

def _count_conflicts(subset):
    """Count the number of duplicate pair sums in subset."""
    sums = {}
    conflicts = 0
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            sm = subset[i] + subset[j]
            if sm in sums:
                conflicts += 1
            else:
                sums[sm] = 1
    return conflicts

def _random_candidate():
    return sorted(random.sample(range(129), 12))

def _local_search(current, deadline, max_iter=500):
    """Hill-climbing: try swapping one element with another from [0,128]."""
    current = sorted(current)
    best_score = _evaluate(current)
    best_set = list(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        improved = False
        # Shuffle positions for diversity
        positions = list(range(12))
        random.shuffle(positions)
        for i in positions:
            if time.time() > deadline:
                break
            # Try a random sample of replacements for speed
            candidates = list(all_vals - in_set)
            random.shuffle(candidates)
            for new_val in candidates:
                candidate = list(current)
                candidate[i] = new_val
                candidate = sorted(candidate)
                score = _evaluate(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate
                    in_set = set(current)
                    in_set.discard(current[i])
                    in_set.add(new_val)
                    current = best_set
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        if not improved:
            break
    
    return best_set, best_score

def _local_search_limited(current, deadline):
    """Limited local search with fewer iterations."""
    return _local_search(current, deadline, max_iter=100)

def _simulated_annealing(current, deadline, max_iter=2000):
    """Simulated annealing to escape local optima."""
    current = sorted(current)
    best_set = list(current)
    best_score = _evaluate(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    temp = 2.0
    cooling = 0.995
    
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        if time.time() > deadline:
            break
        
        # Pick a random position and a random replacement
        i = random.randint(0, 11)
        old_val = current[i]
        new_val = random.choice(list(all_vals - in_set))
        
        candidate = list(current)
        candidate[i] = new_val
        candidate = sorted(candidate)
        
        new_score = _evaluate(candidate)
        delta = new_score - best_score
        
        if delta <= 0 or random.random() < pow(2.718, -delta / temp):
            # Accept
            in_set.discard(old_val)
            in_set.add(new_val)
            current = candidate
            if new_score < best_score:
                best_score = new_score
                best_set = list(current)
        
        temp *= cooling
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s safety margin before 60s limit
    
    random.seed(42)
    
    best_set = list(range(12))
    best_score = _evaluate(best_set)
    
    # Try several random restarts with local search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        candidate = _random_candidate()
        score = _evaluate(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate
            # Local search from this candidate
            best_set, best_score = _local_search_limited(best_set, deadline)
        
        # Also try simulated annealing from current best periodically
        if restarts % 3 == 0 and time.time() < deadline:
            sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iter=500)
            if sa_score < best_score:
                best_score = sa_score
                best_set = sa_set
    
    # Final polish: try more aggressive local search
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iter=200)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
