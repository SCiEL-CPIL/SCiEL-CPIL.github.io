import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def is_sidon_subset(subset):
        seen = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                sm = subset[i] + subset[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def find_sidon_4(ambient):
        # Returns a Sidon subset of size 4 if exists, else None
        for subset in itertools.combinations(ambient, 4):
            if is_sidon_subset(subset):
                return subset
        return None

    def max_sidon_size(ambient):
        # We only care if it's 3 or >= 4
        if find_sidon_4(ambient):
            # It might be larger, but for scoring, 4 is better than 5, etc.
            # But the task says MINIMIZE.
            # If we find 4, score is 1.0 (if seed is 4).
            # If we find 3, score is 1.33.
            # So we want to eliminate all 4s.
            return 4 # Placeholder, we know it's at least 4
        return 3

    best_set = list(range(12))
    best_score = 4 # Lower is better
    
    # Initial candidates
    candidates = [
        list(range(12)),
        list(range(11)) + [20],
        list(range(10)) + [20, 30],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 20],
    ]
    
    # Add some random candidates
    for _ in range(10):
        if time.time() > deadline: break
        cand = random.sample(range(129), 12)
        candidates.append(cand)
        
    # Local search function
    def local_search(initial_set):
        current_set = list(initial_set)
        current_set.sort()
        # Try to eliminate Sidon 4-subsets
        for step in range(1000):
            if time.time() > deadline:
                break
            
            s4 = find_sidon_4(current_set)
            if s4 is None:
                return current_set, 3
            
            # Pick a random element from the Sidon 4-subset to modify
            x = random.choice(s4)
            # Try to replace x with a new value
            replaced = False
            # Try values close to x
            for delta in range(1, 10):
                for sign in [-1, 1]:
                    new_x = x + sign * delta
                    if 0 <= new_x <= 128 and new_x not in current_set:
                        # Check if replacing x with new_x breaks this specific Sidon 4-subset
                        # Actually, we just need to check if the NEW set has a Sidon 4-subset
                        # But that's expensive.
                        # Heuristic: Does new_x create a collision in s4?
                        # Let's just try the swap and check the whole set.
                        new_set = [v for v in current_set if v != x]
                        new_set.append(new_x)
                        new_set.sort()
                        
                        # Check if new_set is better
                        # We check if it has a Sidon 4-subset.
                        # If it does, we might still accept it if it's "closer" to being Sidon-free?
                        # No, we want to eliminate them.
                        # If new_set has NO Sidon 4-subset, we win.
                        if find_sidon_4(new_set) is None:
                            return new_set, 3
                        
                        # If it still has Sidon 4-subsets, we can accept it if it has "fewer" or "different" ones?
                        # For now, let's just accept if it doesn't increase the "badness".
                        # Since we can't easily measure badness, let's just accept random valid swaps
                        # and hope to stumble onto a solution.
                        # But we need to avoid infinite loops.
                        # Let's only accept if the specific s4 is broken.
                        # Does new_set contain a Sidon 4-subset that includes new_x and the other 3 from s4?
                        # Let's check if the specific s4 is broken in new_set.
                        # The other 3 elements are s4 \ {x}
                        others = [v for v in s4 if v != x]
                        test_subset = others + [new_x]
                        if not is_sidon_subset(test_subset):
                            # This swap broke the specific collision pattern of s4
                            # It might have created a new one, but let's take the step.
                            current_set = new_set
                            replaced = True
                            break
                if replaced:
                    break
            
            if not replaced:
                # Try a random swap from the whole set
                x = random.choice(current_set)
                new_x = random.randint(0, 128)
                if new_x not in current_set:
                    new_set = [v for v in current_set if v != x]
                    new_set.append(new_x)
                    new_set.sort()
                    current_set = new_set

        # If we didn't find a solution with size 3, return the current best
        # We should check the max Sidon size of current_set
        # But we know it has at least one Sidon 4-subset if we didn't return early
        # Actually, we might have lost track.
        # Let's just return current_set. The evaluator will compute the score.
        # But we want to return the BEST set we found.
        return current_set, 4

    best_found_set = best_set
    best_found_score = 4
    
    for cand in candidates:
        if time.time() > deadline:
            break
        s, score = local_search(cand)
        if score < best_found_score:
            best_found_score = score
            best_found_set = s
            if score == 3:
                return best_found_set
                
    return best_found_set
