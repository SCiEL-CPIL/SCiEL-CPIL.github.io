import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_golomb(n):
        """
        Construct a Golomb ruler with n marks using a greedy algorithm.
        This minimizes the diameter, which helps the denominator in the score.
        """
        marks = [0]
        for _ in range(n - 1):
            best_pos = None
            best_min_diff = float('inf')
            # Try positions from 1 upwards
            # Optimization: We can limit the search range
            max_search = sum(marks) * 2 + 10 if marks else 10
            for pos in range(1, max_search + 1):
                # Check if adding pos creates a duplicate difference
                valid = True
                min_diff = float('inf')
                for m in marks:
                    d = abs(pos - m)
                    if d == 0:
                        continue
                    # Check if this difference already exists
                    # We need to check against all existing differences
                    # For efficiency, we can precompute or just check O(n^2)
                    # Since n is small (10-50), O(n^3) is fine for greedy step
                    for m2 in marks:
                        if m != m2:
                            if abs(m - m2) == d:
                                valid = False
                                break
                    if not valid:
                        break
                    min_diff = min(min_diff, d)
                
                if valid:
                    # Golomb rulers maximize the minimum difference? No, they just require distinct differences.
                    # Greedy usually picks the smallest valid position to keep diameter small.
                    if pos < best_min_diff or best_pos is None:
                        best_pos = pos
                        best_min_diff = pos # Just tracking the position
                        # Actually, we want the smallest valid position
                        break 
            if best_pos is None:
                # If we didn't find one in the range, just extend
                best_pos = marks[-1] + 1
            marks.append(best_pos)
        
        return sorted(marks)

    # Initial candidates: Mix of structured sets
    candidates = []
    
    # 1. Greedy Golomb rulers of various sizes
    for n in [10, 15, 20, 30]:
        candidates.append(greedy_golomb(n))
        
    # 2. Geometric progressions (small sumsets)
    for r in [2, 3, 5]:
        for n in [10, 15, 20]:
            U = [0]
            cur = 1
            for _ in range(n-1):
                U.append(cur)
                cur *= r
            candidates.append(U)
            
    # 3. Squares (Golomb-like)
    for n in [10, 20, 30]:
        candidates.append([i*i for i in range(n)])
        
    # 4. Random sets
    for _ in range(10):
        n = random.randint(10, 30)
        max_val = random.randint(100, 5000)
        samples = random.sample(range(1, max_val + 1), n - 1)
        samples.append(0)
        candidates.append(sorted(samples))

    # Evaluate initial candidates
    scored = [(score_set(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_set = scored[0]

    # Local Search Refinement
    # Strategy: Try to add or remove elements to improve the score.
    # Since score depends on |U-U|/|U+U| and max(U), we want to:
    # - Increase |U-U| (add elements far apart)
    # - Decrease |U+U| (add elements that create sum collisions? Hard to do greedily)
    # - Decrease max(U) (remove large elements if they don't contribute enough to ratio)
    
    current_set = set(best_set)
    current_score = best_score
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        current_score = score_set(list(current_set))
        
        # 1. Try removing elements
        # Remove the largest element first? Or the one that contributes least?
        # Let's try removing the largest element and see if score improves.
        if len(current_set) > 5:
            max_elem = max(current_set)
            new_set = current_set - {max_elem}
            new_score = score_set(list(new_set))
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                no_improvement_count = 0
                continue
        
        # 2. Try adding elements
        # Candidate positions:
        # - Near the max (to increase range slightly)
        # - Randomly in the range
        # - Gaps in the set
        
        max_val = max(current_set) if current_set else 0
        min_val = min(current_set) if current_set else 0
        
        # Generate candidate additions
        candidates_add = set()
        
        # Add elements just above max
        for delta in range(1, 10):
            candidates_add.add(max_val + delta)
            
        # Add random elements in range
        for _ in range(50):
            candidates_add.add(random.randint(1, max_val * 2))
            
        # Add elements to fill gaps?
        sorted_set = sorted(current_set)
        for i in range(len(sorted_set) - 1):
            gap = sorted_set[i+1] - sorted_set[i]
            if gap > 2:
                candidates_add.add(sorted_set[i] + gap // 2)
        
        # Remove 0 from candidates as it's already there
        candidates_add.discard(0)
        
        best_add_score = current_score
        best_add_elem = None
        
        for elem in candidates_add:
            if elem in current_set:
                continue
            new_set = current_set | {elem}
            new_score = score_set(list(new_set))
            if new_score > best_add_score:
                best_add_score = new_score
                best_add_elem = elem
        
        if best_add_elem is not None:
            current_set.add(best_add_elem)
            current_score = best_add_score
            improved = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            # If we can't add, maybe we should try a more aggressive removal?
            # Or try swapping?
            # Let's try swapping the largest element with a smaller one.
            if len(current_set) > 5:
                max_elem = max(current_set)
                # Try replacing max_elem with a random smaller element
                for _ in range(10):
                    new_elem = random.randint(1, max_elem)
                    if new_elem in current_set:
                        continue
                    new_set = (current_set - {max_elem}) | {new_elem}
                    new_score = score_set(list(new_set))
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
        
        if not improved:
             # Random perturbation to escape local minima
             if random.random() < 0.5:
                 # Remove a random element
                 if len(current_set) > 5:
                     elem_to_remove = random.choice(list(current_set - {0}))
                     current_set.remove(elem_to_remove)
                     current_score = score_set(list(current_set))
                 # Add a random element
                 new_elem = random.randint(1, max(current_set) * 2 + 10)
                 current_set.add(new_elem)
                 current_score = score_set(list(current_set))
             
             no_improvement_count += 1

    return sorted(list(current_set))
