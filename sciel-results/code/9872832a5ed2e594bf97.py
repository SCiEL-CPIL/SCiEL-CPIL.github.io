import time
import random

N = 256

def construct():
    # Deadline guard
    deadline = time.time() + 25.0
    
    def is_valid_sidon(s_list):
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                val = s_list[i] + s_list[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def is_maximal(s_list):
        # s_list is a list of integers
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        
        s_set = set(s_list)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if adding x breaks Sidon property
            # New sums: 2x and x+a for a in s
            if 2 * x in sums:
                continue
            can_add = True
            for a in s_list:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True

    def greedy_maximal(perm):
        selected = []
        sums = set()
        for x in perm:
            # Check if x can be added
            if 2 * x in sums:
                continue
            
            can_add = True
            for a in selected:
                if (x + a) in sums:
                    can_add = False
                    break
            
            if can_add:
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected

    def peel_maximal(s_list):
        # Try to remove elements while keeping the set maximal
        current = list(s_list)
        changed = True
        # Randomize removal order to escape local minima
        while changed and time.time() < deadline:
            changed = False
            # Create a random order of indices
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(current):
                    continue
                x = current[idx]
                # Try removing x
                candidate = current[:idx] + current[idx+1:]
                if is_maximal(candidate):
                    current = candidate
                    changed = True
        return current

    # Initial best: Use a known good set or a greedy one
    # Known good: {1, 3, 7, 15, 31, 63, 127}
    known_set = [1, 3, 7, 15, 31, 63, 127]
    if is_valid_sidon(known_set) and is_maximal(known_set):
        best_set = list(known_set)
        best_size = len(best_set)
    else:
        # Fallback greedy
        perm = list(range(1, N + 1))
        best_set = greedy_maximal(perm)
        best_set = peel_maximal(best_set)
        best_size = len(best_set)

    # Search loop
    while time.time() < deadline:
        # Generate a random permutation
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        # Construct a maximal Sidon set
        candidate = greedy_maximal(perm)
        
        # Peel it down to a smaller maximal set
        candidate = peel_maximal(candidate)
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            
            # If we find a very small set, we might want to continue
            # but 6 or 7 is likely the minimum for N=256
            if best_size <= 6:
                # We can try to polish this specific set further
                # But peel_maximal already does local search
                pass

    return best_set
