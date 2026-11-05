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
        """Greedily swap elements to maximize collisions."""
        s = list(s)
        for iteration in range(200):
            improved = False
            for i in range(12):
                best_swap = None
                best_score = count_collisions(s)
                # Try swapping s[i] with each possible value in [0, 128]
                for new_val in range(129):
                    if new_val in s and new_val != s[i]:
                        continue
                    if new_val == s[i]:
                        continue
                    s[i] = new_val
                    if len(set(s)) != 12:
                        s[i] = s[i]  # restore
                        continue
                    score = count_collisions(s)
                    if score > best_score:
                        best_score = score
                        best_swap = new_val
                if best_swap is not None:
                    s[i] = best_swap
                    improved = True
            if not improved:
                break
        return s
    
    # Start with several random initializations and pick the best
    random.seed(42)
    best_set = None
    best_collision = -1
    
    for trial in range(50):
        # Random initial set
        s = random.sample(range(129), 12)
        # Optimize
        s = optimize(s)
        col = count_collisions(s)
        if col > best_collision:
            best_collision = col
            best_set = s
    
    # Also try structured initializations
    structured_starts = [
        list(range(12)),
        list(range(0, 129, 11))[:12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 124],
    ]
    for start in structured_starts:
        if len(set(start)) == 12 and all(0 <= x <= 128 for x in start):
            s = optimize(list(start))
            col = count_collisions(s)
            if col > best_collision:
                best_collision = col
                best_set = s
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return list(best_set)
