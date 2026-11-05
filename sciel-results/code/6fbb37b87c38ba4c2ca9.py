import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s limit for safety (90s total, but be conservative)
    
    max_val = 256
    numbers = list(range(2, max_val + 1))
    n = len(numbers)
    
    # Precompute conflict matrix
    # conflict[i][j] = True if lcm(numbers[i], numbers[j]) <= max_val (conflict)
    # We need lcm > 256 for all distinct pairs, so conflict means NOT allowed together
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.lcm(numbers[i], numbers[j]) <= max_val:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weights: 1/number
    weights = [Fraction(1, x) for x in numbers]
    
    # Greedy approach
    def greedy(candidates, selected_set):
        """Greedy: add candidates in order of decreasing weight if compatible with all selected"""
        # candidates: list of indices
        # selected_set: set of indices already selected
        result = set(selected_set)
        sorted_cands = sorted(candidates, key=lambda i: -weights[i])
        for c in sorted_cands:
            if c in result:
                continue
            # Check compatibility with all in result
            compatible = True
            for s in result:
                if conflict[c][s]:
                    compatible = False
                    break
            if compatible:
                result.add(c)
        return result
    
    def score_set(indices):
        """Compute exact reciprocal sum for a set of indices"""
        total = Fraction(0, 1)
        for i in indices:
            total += weights[i]
        return total
    
    def is_valid_set(indices):
        """Check if all pairs in indices are compatible"""
        idx_list = list(indices)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if conflict[idx_list[i]][idx_list[j]]:
                    return False
        return True
    
    # Strategy 1: Global greedy
    all_indices = list(range(n))
    s1 = greedy(all_indices, set())
    best_score = score_set(s1)
    best_set = s1
    
    # Strategy 2: Local search - try to add elements
    def local_search(current_set, time_limit=2.0):
        """Try to improve the set by adding elements or swapping"""
        current = set(current_set)
        current_score = score_set(current)
        
        # Repeatedly try to add the best compatible element
        improved = True
        while improved:
            improved = False
            # Find the best element to add
            best_add = None
            best_gain = Fraction(0, 1)
            for i in range(n):
                if i in current:
                    continue
                # Check if i is compatible with all in current
                compatible = True
                for s in current:
                    if conflict[i][s]:
                        compatible = False
                        break
                if compatible:
                    gain = weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
            if best_add is not None:
                current.add(best_add)
                improved = True
        
        # Try swaps: remove one, add two
        # This is more complex, so do a simpler version
        # Try removing one element and see if we can add more
        for remove_i in list(current):
            if time.time() > time_limit:
                break
            temp = set(current)
            temp.remove(remove_i)
            # Now try to add elements
            temp_improved = True
            while temp_improved:
                temp_improved = False
                best_add = None
                best_gain = Fraction(0, 1)
                for i in range(n):
                    if i in temp:
                        continue
                    compatible = True
                    for s in temp:
                        if conflict[i][s]:
                            compatible = False
                            break
                    if compatible:
                        gain = weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = i
                if best_add is not None:
                    temp.add(best_add)
                    temp_improved = True
            
            temp_score = score_set(temp)
            if temp_score > current_score:
                current = temp
                current_score = temp_score
        
        return current, current_score
    
    # Apply local search to the initial greedy result
    if time.time() < deadline:
        improved_set, improved_score = local_search(s1, time_limit=5.0)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Strategy 3: Multiple random restarts with different orderings
    if time.time() < deadline:
        for restart in range(50):
            if time.time() > deadline:
                break
            # Randomly shuffle candidates
            shuffled = all_indices[:]
            random.shuffle(shuffled)
            # Weighted shuffle: bias towards smaller numbers
            shuffled.sort(key=lambda i: (random.random() * 0.5 - 0.25, -weights[i]))
            s = greedy(shuffled, set())
            # Apply local search
            improved_s, improved_s_score = local_search(s, time_limit=1.0)
            if improved_s_score > best_score:
                best_score = improved_s_score
                best_set = improved_s
    
    # Strategy 4: Simulated annealing-like approach
    if time.time() < deadline:
        # Start from best found
        current = set(best_set)
        current_score = best_score
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Random perturbation: add or remove a random element
            action = random.choice(['add', 'remove', 'swap'])
            temp = set(current)
            
            if action == 'add':
                # Add a random element (even if it causes conflict, we'll fix)
                i = random.randint(0, n - 1)
                if i not in temp:
                    temp.add(i)
            elif action == 'remove':
                if temp:
                    # Remove a random element
                    to_remove = random.choice(list(temp))
                    temp.remove(to_remove)
            else:  # swap
                if temp:
                    # Remove one random, add one random
                    to_remove = random.choice(list(temp))
                    temp.remove(to_remove)
                    i = random.randint(0, n - 1)
                    if i not in temp:
                        temp.add(i)
            
            # Make valid: remove conflicting elements (keep higher weight ones)
            # Simple approach: repeatedly remove the lower-weight element from each conflict
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if i not in temp:
                        continue
                    for j in range(i + 1, n):
                        if j not in temp:
                            continue
                        if conflict[i][j]:
                            # Remove the one with lower weight
                            if weights[i] <= weights[j]:
                                temp.remove(i)
                            else:
                                temp.remove(j)
                            changed = True
                            break
                    if changed:
                        break
            
            if not temp:
                continue
            
            temp_score = score_set(temp)
            delta = temp_score - current_score
            
            # Accept if better, or with some probability if worse
            if delta >= 0:
                current = temp
                current_score = temp_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current
            else:
                # Simple acceptance: always accept worse solutions occasionally
                # This is very crude, but let's try
                if random.random() < 0.1:
                    current = temp
                    current_score = temp_score
    
    # Ensure result is non-empty
    if not best_set:
        # Pick the smallest number (highest weight)
        best_set = {0}
        best_score = weights[0]
    
    # Convert to list of actual numbers
    result = [numbers[i] for i in best_set]
    
    # Final validation
    assert len(result) > 0, "Result must be non-empty"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.lcm(result[i], result[j]) > max_val, f"Conflict: {result[i]}, {result[j]}"
    
    return result
