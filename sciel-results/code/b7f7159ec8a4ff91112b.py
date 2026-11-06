"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from collections import defaultdict

_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds before the 90s kill

def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return 90.0 - _now() - _DEADLINE_MARGIN

def _is_sidon_subset(subset):
    """Check if a sorted tuple of integers has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size_exact(ambient):
    """Exhaustively compute the largest subset of ambient with unique pair sums (incl. diagonals)."""
    ambient = sorted(ambient)
    n = len(ambient)
    # Use bitmasks for subsets; n is at most 12 so 2^12 = 4096 subsets
    # We want the max size k such that some k-subset is Sidon
    # Try from large to small
    for k in range(n, 0, -1):
        for combo in itertools.combinations(range(n), k):
            subset = tuple(ambient[i] for i in combo)
            if _is_sidon_subset(subset):
                return k
    return 1  # single element is always Sidon

def _count_conflict_hyperedges(ambient):
    """Count the number of quadruples (a,b,c,d) with a+b=c+d where the pairs are distinct.
    This serves as a proxy for how 'hard' the set is for Sidon extraction."""
    ambient = sorted(ambient)
    n = len(ambient)
    # Count all pair sums and how many times each appears
    sum_counts = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            sum_counts[s] += 1
    # A conflict exists when a sum appears more than once
    # Number of conflicting pairs = sum over sums of C(count, 2)
    conflicts = 0
    for s, c in sum_counts.items():
        if c > 1:
            conflicts += c * (c - 1) // 2
    return conflicts

def _local_search(ambient, max_iterations=50000):
    """Stochastic local search to minimize max Sidon subset size."""
    ambient = list(sorted(ambient))
    n = len(ambient)
    # Precompute the set of all values [0, 128]
    all_values = set(range(129))
    
    # Evaluate current
    current_sidon_size = _max_sidon_size_exact(ambient)
    current_conflicts = _count_conflict_hyperedges(ambient)
    
    best_ambient = list(ambient)
    best_sidon_size = current_sidon_size
    best_conflicts = current_conflicts
    
    # We want to MINIMIZE the max Sidon subset size
    # Use conflicts as a secondary objective (more conflicts = harder)
    
    temperature = 1.0
    min_temperature = 0.01
    
    for it in range(max_iterations):
        if _time_left() < 2.0:
            break
        
        # Random swap: pick two positions, replace one with a random value from [0,128]
        i = random.randint(0, n - 1)
        new_val = random.randint(0, 128)
        
        if new_val in ambient:
            continue
        
        old_val = ambient[i]
        ambient[i] = new_val
        ambient_sorted = sorted(ambient)
        
        new_conflicts = _count_conflict_hyperedges(ambient_sorted)
        
        # Accept if it improves (more conflicts) or with some probability
        delta_conf = new_conflicts - current_conflicts
        
        if delta_conf > 0:
            # Better: accept
            pass
        elif delta_conf == 0:
            # Neutral: accept with some probability
            if random.random() > 0.5:
                # Revert
                ambient[i] = old_val
                continue
        else:
            # Worse: accept with probability based on temperature
            if random.random() < temperature * 0.1:
                pass
            else:
                # Revert
                ambient[i] = old_val
                continue
        
        # Periodically evaluate exact Sidon size (expensive)
        if it % 50 == 0 or new_conflicts > best_conflicts:
            new_sidon_size = _max_sidon_size_exact(ambient_sorted)
            
            if new_sidon_size < best_sidon_size or (new_sidon_size == best_sidon_size and new_conflicts > best_conflicts):
                best_ambient = list(ambient_sorted)
                best_sidon_size = new_sidon_size
                best_conflicts = new_conflicts
                current_sidon_size = new_sidon_size
            else:
                # If exact evaluation shows no improvement, revert
                ambient[i] = old_val
                continue
            
            current_conflicts = new_conflicts
        else:
            current_conflicts = new_conflicts
        
        # Cool down
        temperature = max(min_temperature, temperature * 0.999)
    
    return best_ambient, best_sidon_size

def _greedy_construct():
    """Construct a set by greedily adding elements that create many conflicts."""
    ambient = []
    all_values = list(range(129))
    random.shuffle(all_values)
    
    # Start with a small random seed
    for v in all_values[:4]:
        ambient.append(v)
    
    # Greedily add elements that maximize conflict count
    while len(ambient) < 12:
        if _time_left() < 2.0:
            break
        best_val = None
        best_conflicts = _count_conflict_hyperedges(sorted(ambient))
        for v in all_values:
            if v in ambient:
                continue
            test_ambient = ambient + [v]
            c = _count_conflict_hyperedges(sorted(test_ambient))
            if c > best_conflicts:
                best_conflicts = c
                best_val = v
        if best_val is not None:
            ambient.append(best_val)
        else:
            # No improvement possible, add random
            remaining = [v for v in all_values if v not in ambient]
            if remaining:
                ambient.append(random.choice(remaining))
            else:
                break
    
    return sorted(ambient[:12])

def construct():
    """Return exactly 12 distinct integers in [0,128] that minimize the largest Sidon subset size."""
    random.seed(42)  # Reproducibility
    
    best_result = None
    best_score = 12  # Max possible Sidon subset size
    
    # Multiple strategies
    # Strategy 1: Random starts with local search
    for attempt in range(200):
        if _time_left() < 5.0:
            break
        
        # Random 12-element set
        ambient = random.sample(range(129), 12)
        
        # Local search
        result_ambient, result_sidon = _local_search(ambient, max_iterations=30000)
        
        if result_sidon < best_score:
            best_score = result_sidon
            best_result = result_ambient
        
        # If we found a very good result, try to improve
        if best_score <= 4:
            break
    
    # Strategy 2: Greedy construction
    if _time_left() > 3.0:
        for attempt in range(50):
            if _time_left() < 3.0:
                break
            ambient = _greedy_construct()
            if len(ambient) == 12:
                sidon_size = _max_sidon_size_exact(ambient)
                if sidon_size < best_score:
                    best_score = sidon_size
                    best_result = ambient
    
    # Final evaluation
    if best_result is not None and len(best_result) == 12:
        # Verify
        assert len(set(best_result)) == 12
        assert all(0 <= x <= 128 for x in best_result)
        return list(best_result)
    
    # Fallback
    return list(range(12))

# EVOLVE-BLOCK-END
