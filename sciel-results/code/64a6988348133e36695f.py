import random
import itertools

def construct():
    def count_collisions(s):
        """Count the number of duplicate unordered pair sums (including diagonals) in set s."""
        s = sorted(s)
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        # Count duplicates
        from collections import Counter
        c = Counter(sums)
        # Number of "extra" collisions = total pairs - number of distinct sums
        return sum(v - 1 for v in c.values() if v > 1)
    
    def largest_sidon_size(s):
        """Find the largest Sidon subset size in s."""
        s = sorted(s)
        n = len(s)
        best = 1
        # Try all subsets from largest to smallest
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(s, size):
                subset = list(subset)
                sums = set()
                valid = True
                for i in range(len(subset)):
                    for j in range(i, len(subset)):
                        sm = subset[i] + subset[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                    if not valid:
                        break
                if valid:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    def optimize(s):
        """Randomized local search to minimize largest Sidon subset size."""
        s = list(s)
        current_size = largest_sidon_size(s)
        
        no_improvement_count = 0
        max_no_improvement = 20
        max_iterations = 100
        
        for iteration in range(max_iterations):
            if no_improvement_count > max_no_improvement:
                break
            
            # Pick a random element to change
            i = random.randint(0, 11)
            old_val = s[i]
            
            # Try a few random new values
            best_new_val = None
            best_new_size = current_size
            
            # Sample candidates
            # We want to avoid values already in the set
            available = [v for v in range(129) if v not in s]
            if not available:
                break
            candidates = random.sample(available, 10)
            
            for new_val in candidates:
                s[i] = new_val
                new_size = largest_sidon_size(s)
                if new_size < best_new_size:
                    best_new_size = new_size
                    best_new_val = new_val
                s[i] = old_val
            
            if best_new_val is not None:
                s[i] = best_new_val
                current_size = best_new_size
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        return s

    # Start with several random initializations and pick the best
    random.seed(42)
    best_set = None
    best_size = 100  # Initialize with a large number
    
    for trial in range(100):
        # Random initial set
        s = random.sample(range(129), 12)
        # Optimize
        s = optimize(s)
        size = largest_sidon_size(s)
        
        if size < best_size:
            best_size = size
            best_set = s
            
        # If we hit a very low size (e.g., 4), we can stop early
        if best_size <= 4:
            break
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return list(best_set)
