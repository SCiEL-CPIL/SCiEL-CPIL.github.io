import random
import time
import itertools
from functools import lru_cache

def _is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(s):
    """Find the exact size of the largest Sidon subset of s using branch-and-bound."""
    s = sorted(s)
    n = len(s)
    
    # Quick check: if n <= 1, return n
    if n <= 1:
        return n
    
    # For small sets, just check directly
    if n <= 8:
        # Check from n down to 1
        for size in range(n, 0, -1):
            for combo in itertools.combinations(s, size):
                if _is_sidon(combo):
                    return size
        return 1
    
    # Branch and bound for larger sets
    # We'll use a greedy upper bound based on sum collisions
    
    # Precompute all pair sums
    # For a subset to be Sidon of size k, it needs C(k,2)+k = k(k+1)/2 unique sums
    # The total number of possible sums from s is bounded
    
    # Use a recursive approach with pruning
    best = [1]
    
    def can_extend(subset, start_idx):
        """Try to extend the current Sidon subset starting from start_idx."""
        if not _is_sidon(subset):
            return False
        if len(subset) > best[0]:
            best[0] = len(subset)
        return len(subset)
    
    # Try to find a large Sidon subset using backtracking
    # Start with each element as a seed
    for i in range(n):
        # Try building a Sidon set starting with s[i]
        stack = [(i, [s[i]])]
        while stack:
            idx, current = stack.pop()
            if len(current) > best[0]:
                best[0] = len(current)
            # Try adding elements after idx
            for j in range(idx + 1, n):
                new_current = current + [s[j]]
                if _is_sidon(new_current):
                    stack.append((j, new_current))
    
    return best[0]

def count_sum_collisions(s):
    """Count the number of 4-tuples (a,b,c,d) with a<=b, c<=d, a+b=c+d, {a,b}!={c,d}."""
    s = sorted(s)
    n = len(s)
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            sum_count[sm] = sum_count.get(sm, 0) + 1
    # For each sum that appears k times, it creates C(k,2) collision pairs
    collisions = 0
    for k in sum_count.values():
        if k > 1:
            collisions += k * (k - 1) // 2
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin before 90s limit
    
    best_set = None
    best_score = float('inf')
    
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            return best_score <= 4  # If max Sidon size <= 4, we're done
        return False
    
    random.seed(42)
    
    # Structured candidates designed to maximize sum collisions
    candidates = [
        # Dense cluster: many small differences create sum collisions
        list(range(12)),
        # AP step 2
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Very dense with one outlier
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        # Clustered pairs
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14],
        # Symmetric around center
        [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        # Quadratic-like
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
        # Two tight clusters
        [0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65],
        # Arithmetic progression with step 1
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        # Mixed dense
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        # Dense with gaps
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],
    ]
    
    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = largest_sidon_subset_size(cand)
            if update_best(cand, score):
                return best_set
    
    # Local search to minimize largest Sidon subset size
    # Start from best structured candidate
    current_set = sorted(best_set) if best_set else sorted(random.sample(range(129), 12))
    current_score = largest_sidon_subset_size(current_set)
    
    if update_best(current_set, current_score):
        return best_set
    
    # Simulated annealing with collision-based scoring
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.1
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Initialize from best or random
        if restart == 0:
            current_set = sorted(best_set) if best_set else sorted(random.sample(range(129), 12))
        else:
            # Perturb best solution
            current_set = best_set.copy()
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 11)
                base = current_set[idx]
                offset = random.randint(-8, 8)
                new_val = max(0, min(128, base + offset))
                if new_val not in current_set:
                    current_set[idx] = new_val
            current_set = sorted(set(current_set))
            if len(current_set) < 12:
                existing = set(current_set)
                for v in range(129):
                    if v not in existing and len(current_set) < 12:
                        current_set.append(v)
                current_set = sorted(current_set[:12])
        
        current_score = largest_sidon_subset_size(current_set)
        
        if update_best(current_set, current_score):
            return best_set
        
        # Local search with simulated annealing
        T = 2.0
        no_improve = 0
        while T > min_T and time.time() < deadline and no_improve < 100:
            # Generate neighbor: replace one element
            new_set = current_set.copy()
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            
            # Prefer values that create more sum collisions
            if random.random() < 0.5:
                # Pick a value close to existing elements to create collisions
                base = random.choice(current_set)
                offset = random.randint(-6, 6)
                new_val = max(0, min(128, base + offset))
            elif random.random() < 0.5:
                # Pick a value that creates a specific collision
                a = random.choice(current_set)
                b = random.choice(current_set)
                c = random.choice(current_set)
                new_val = b + c - a
                if 0 <= new_val <= 128:
                    new_val = int(new_val)
                else:
                    new_val = random.randint(0, 128)
            else:
                new_val = random.randint(0, 128)
            
            if new_val != old_val and new_val not in new_set:
                new_set[idx] = new_val
                new_set.sort()
                
                if len(set(new_set)) == 12:
                    new_score = largest_sidon_subset_size(new_set)
                    delta = new_score - current_score
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / max(T, 1e-9)):
                        current_set = new_set
                        current_score = new_score
                        no_improve = 0
                        
                        if update_best(current_set, current_score):
                            return best_set
                    else:
                        no_improve += 1
                    
                    T *= cooling_rate
            else:
                no_improve += 1
                T *= cooling_rate
    
    return best_set
