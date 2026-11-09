import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    def check_pair(a, b):
        """Return True if (a*b) % (a+b) != 0 (i.e., compatible)."""
        return (a * b) % (a + b) != 0
    
    def can_add(num, s):
        """Check if num can be added to set s."""
        for x in s:
            if x == num:
                continue
            if not check_pair(x, num):
                return False
        return True
    
    # Start with all odd numbers (guaranteed valid since sum of two odds is even,
    # product is odd, so even number can't divide odd number)
    odds = set(x for x in range(1, 257) if x % 2 == 1)
    best_set = set(odds)
    best_size = len(best_set)
    
    evens = [x for x in range(2, 257) if x % 2 == 0]
    
    # Multiple greedy attempts with different random orderings
    # Also try adding powers of 2 first in some attempts
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        current = set(odds)
        
        # Create a shuffled list of evens
        shuffled_evens = list(evens)
        random.shuffle(shuffled_evens)
        
        # In some attempts, prioritize powers of 2
        if attempt % 3 == 0:
            powers_of_2 = [2, 4, 8, 16, 32, 64, 128]
            other_evens = [e for e in shuffled_evens if e not in powers_of_2]
            # Try powers of 2 first
            for p in powers_of_2:
                if can_add(p, current):
                    current.add(p)
            shuffled_evens = other_evens
        
        for e in shuffled_evens:
            if time.time() > deadline:
                break
            if can_add(e, current):
                current.add(e)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search: try to add more elements by removing conflicting ones
    # and then re-adding
    for ls_iter in range(100):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        remaining = [x for x in range(1, 257) if x not in current]
        random.shuffle(remaining)
        
        improved = False
        for r in remaining:
            if time.time() > deadline:
                break
            # Try adding r; if it conflicts, see if we can remove one conflicting element
            conflicts = [x for x in current if not check_pair(x, r)]
            if not conflicts:
                current.add(r)
                improved = True
            elif len(conflicts) == 1:
                # Try removing the single conflict and adding r
                c = conflicts[0]
                new_set = current - {c}
                new_set.add(r)
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
                elif len(new_set) == len(current):
                    # Neutral swap - accept with some probability to explore
                    if random.random() < 0.3:
                        current = new_set
                        improved = True
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            improved = True
        
        if not improved:
            # Try a few more iterations with different shuffles before giving up
            if ls_iter < 20:
                continue
            break
    
    return list(best_set)
