import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def is_sidon(subset):
        """Check if subset has all unordered pair sums unique including diagonals (2x)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(s):
        """Find the largest Sidon subset of s using exhaustive search."""
        n = len(s)
        # Try sizes from n down to 3
        for k in range(n, 2, -1):
            for comb in combinations(s, k):
                if is_sidon(comb):
                    return k
        return 3  # any 3 distinct elements form a Sidon set
    
    def objective(s):
        """Return the max Sidon subset size (lower is better)."""
        return max_sidon_size(s)
    
    # Initial candidate: try to build a set with lots of 4-sum collisions
    # Start with a structured set
    best = None
    best_score = 12
    
    # Try several random initializations and local search
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate initial set
        if trial == 0:
            # Structured: multiples that create many a+d=b+c patterns
            s = sorted(random.sample(range(0, 129), 12))
        else:
            s = sorted(random.sample(range(0, 129), 12))
        
        current_score = objective(s)
        if best is None or current_score < best_score:
            best = list(s)
            best_score = current_score
        
        if best_score <= 3:
            return best
        
        # Local search: swap elements
        for step in range(2000):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            # Pick a new value
            new_val = random.randint(0, 128)
            if new_val in s:
                continue
            
            old_val = s[i]
            s[i] = new_val
            s.sort()
            
            new_score = objective(s)
            
            if new_score <= current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = list(s)
                    if best_score <= 3:
                        return best
            else:
                # Revert
                s[i] = old_val
                s.sort()
        
        if best_score <= 3:
            return best
    
    if best is None:
        best = list(range(12))
    
    return best
