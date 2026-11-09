import random
import time
import itertools

def _is_sidon_general(subset):
    """Check if a subset of any size is Sidon (all pair sums including diagonals unique)."""
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
    """Find the size of the largest Sidon subset of s (up to size 5)."""
    s = sorted(s)
    n = len(s)
    # Check from size 5 down to 1
    for size in range(min(n, 5), 0, -1):
        for combo in itertools.combinations(s, size):
            if _is_sidon_general(combo):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit
    
    best_set = None
    best_score = float('inf')
    
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            return best_score <= 3  # If max Sidon size <= 3, we're done
        return False
    
    random.seed(42)
    
    # Structured candidates designed to maximize sum collisions
    candidates = [
        # Dense cluster: many small differences create sum collisions
        list(range(12)),
        # AP step 1 offset
        list(range(1, 13)),
        # Two dense clusters
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # AP step 2
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # AP step 2 offset
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        # Mixed: dense + AP
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        # Very dense
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        # Clustered
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14],
        # Symmetric around center
        [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        # Quadratic-like (creates sum structure)
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
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
    
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.05
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize from best or random
        if restart == 0:
            current_set = sorted(best_set) if best_set else sorted(random.sample(range(129), 12))
        else:
            # Perturb best solution
            current_set = best_set.copy()
            for _ in range(random.randint(1, 4)):
                idx = random.randint(0, 11)
                base = current_set[idx]
                offset = random.randint(-5, 5)
                new_val = max(0, min(128, base + offset))
                if new_val not in current_set:
                    current_set[idx] = new_val
            current_set = sorted(set(current_set))
            if len(current_set) < 12:
                # Fill in missing values
                existing = set(current_set)
                for v in range(129):
                    if v not in existing and len(current_set) < 12:
                        current_set.append(v)
                current_set = sorted(current_set[:12])
        
        current_score = largest_sidon_subset_size(current_set)
        
        if update_best(current_set, current_score):
            return best_set
        
        # Local search with simulated annealing
        T = 1.0
        no_improve = 0
        while T > min_T and time.time() < deadline and no_improve < 80:
            # Generate neighbor: replace one element
            new_set = current_set.copy()
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            
            # Prefer values that create more sum collisions
            if random.random() < 0.6:
                # Pick a value close to existing elements
                base = random.choice(current_set)
                offset = random.randint(-4, 4)
                new_val = max(0, min(128, base + offset))
            elif random.random() < 0.5:
                # Pick a value that creates a specific collision
                # Try to make new_val + a == b + c for some a,b,c in set
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
