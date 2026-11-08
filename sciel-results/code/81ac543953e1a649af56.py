"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) on C6.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Keep within 120s limit with safety margin
    
    # Seed program gives [0, 1, 3]
    # Let's compute its score as baseline
    def compute_score(U):
        U = sorted(set(U))
        n = len(U)
        if n == 0 or max(U) == 0:
            return 0.0
        # Compute sumset U+U
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set U-U
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        # The formula uses |U-U| and |U+U|
        # But U-U includes negative numbers. The problem says "nonnegative integers" for U
        # but U-U can have negative differences.
        # Let's re-read: "score is the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)"
        # So we need |U-U| and |U+U|
        size_diff = len(diffset)
        size_sum = len(sumset)
        max_u = max(U)
        if max_u == 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        ratio = size_diff / size_sum
        if ratio <= 0:
            return 0.0
        score = 1 + math.log(ratio) / denom
        return score
    
    # Baseline score
    baseline = [0, 1, 3]
    best_score = compute_score(baseline)
    best_set = list(baseline)
    
    # The seed score is the baseline. We want to beat it.
    # Let's try some known constructions and random search.
    
    # Strategy: Use a random set and try to optimize it.
    # The ratio |U-U|/|U+U| is what we want to maximize (since log is monotone),
    # and we want to minimize log(2*max(U)+1) in the denominator, so we want small max(U)
    # but large ratio.
    
    # For a set of size n, the maximum possible |U+U| is n(n+1)/2 (if all sums distinct)
    # and |U-U| is at most n^2 - n + 1 (if all differences distinct and non-overlapping).
    # Actually |U-U| = 2*|U+U_nonneg| - 1 where U+U_nonneg is the set of nonneg differences.
    
    # Let's try small sets first to understand the structure.
    
    # Try all subsets of {0, 1, ..., 10} of various sizes? That's 2^11 = 2048, feasible.
    # But we need to be careful about time.
    
    # Let's try a different approach: generate random sets and score them.
    random.seed(42)
    
    def random_set(n, max_val):
        s = {0}
        for _ in range(n - 1):
            s.add(random.randint(1, max_val))
        return sorted(s)
    
    # Try various random sets
    for trial in range(5000):
        if time.time() > deadline - 10.0:
            break
        n = random.randint(3, 15)
        max_val = random.randint(5, 50)
        s = random_set(n, max_val)
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Try structured sets: arithmetic progressions
    for n in range(3, 20):
        for a in range(1, 10):
            for d in range(1, 20):
                if time.time() > deadline - 10.0:
                    break
                s = [a * i for i in range(n)]
                # Ensure 0 is in the set
                if 0 not in s:
                    s.insert(0, 0)
                score = compute_score(s)
                if score > best_score:
                    best_score = score
                    best_set = list(s)
    
    # Try geometric-like sets
    for base in range(2, 10):
        for n in range(3, 15):
            if time.time() > deadline - 10.0:
                break
            s = [0]
            for i in range(1, n):
                s.append(base ** i)
            score = compute_score(s)
            if score > best_score:
                best_score = score
                best_set = list(s)
    
    # Local search: try to modify the best set
    def local_search(s, max_iter=100):
        s = sorted(set(s))
        current_score = compute_score(s)
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            improved = False
            # Try adding a random element
            if len(s) < 30:
                max_val = max(s) + random.randint(1, 10)
                new_s = s + [random.randint(1, max_val)]
                new_score = compute_score(new_s)
                if new_score > current_score:
                    s = sorted(set(new_s))
                    current_score = new_score
                    improved = True
            
            # Try removing a random element (but keep 0)
            if len(s) > 3:
                idx = random.choice(range(1, len(s)))
                new_s = s[:idx] + s[idx+1:]
                new_score = compute_score(new_s)
                if new_score > current_score:
                    s = new_s
                    current_score = new_score
                    improved = True
            
            # Try replacing a random element
            if len(s) > 3:
                idx = random.choice(range(1, len(s)))
                old_val = s[idx]
                new_val = random.randint(1, max(s) + 5)
                if new_val not in s:
                    new_s = s[:idx] + [new_val] + s[idx+1:]
                    new_score = compute_score(new_s)
                    if new_score > current_score:
                        s = sorted(set(new_s))
                        current_score = new_score
                        improved = True
            
            if not improved:
                # Perturb the set
                idx = random.choice(range(1, len(s)))
                s[idx] = s[idx] + random.randint(-3, 3)
                s = sorted(set(x for x in s if x >= 0))
                if 0 not in s:
                    s.insert(0, 0)
        
        return s, current_score
    
    if time.time() < deadline - 15.0:
        improved_set, improved_score = local_search(best_set, max_iter=50)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
