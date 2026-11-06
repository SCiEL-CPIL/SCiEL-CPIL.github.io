import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Optimized score calculation using bitset for O(1) membership
    def get_score(S):
        # Build a boolean array for fast membership
        # Values are in 1..512, so we use a 513-element list
        present = [False] * 513
        for x in S:
            present[x] = True
        
        S_sorted = sorted(S)
        n = len(S_sorted)
        d_set = set()
        
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if c <= 512 and present[c]:
                    d_set.add(d)
        return len(d_set)

    # Better initial set generation
    def generate_initial_sets():
        sets = []
        
        # 1. Simple interval
        sets.append(set(range(1, 33)))
        
        # 2. Structured grid: use multiples of small numbers
        # Try sets based on arithmetic progressions with different step sizes
        for step in [1, 2, 3, 4, 5]:
            s = set()
            for i in range(32):
                val = 1 + i * step
                if val <= 512:
                    s.add(val)
            if len(s) == 32:
                sets.append(s)
        
        # 3. Sums of two squares
        candidates = set()
        for i in range(1, 23):
            for j in range(1, 23):
                val = i*i + j*j
                if 1 <= val <= 512:
                    candidates.add(val)
        if len(candidates) >= 32:
            sets.append(set(random.sample(sorted(candidates), 32)))
        
        # 4. Grid-like structure with varied spacing
        for base in [8, 10, 12, 16]:
            grid_set = set()
            for i in range(1, 6):
                for j in range(1, 7):
                    val = i * base + j
                    if 1 <= val <= 512:
                        grid_set.add(val)
            if len(grid_set) >= 32:
                sets.append(set(random.sample(sorted(grid_set), 32)))
        
        # 5. Random sets
        for _ in range(3):
            sets.append(set(random.sample(range(1, 513), 32)))
        
        return sets

    best_set = None
    best_score = -1
    
    # Find best initial set
    init_sets = generate_initial_sets()
    for s in init_sets:
        sc = get_score(s)
        if sc > best_score:
            best_score = sc
            best_set = set(s)
    
    current_set = set(best_set)
    current_score = best_score
    
    # Simulated Annealing
    start_temp = 8.0
    end_temp = 0.005
    cooling_rate = 0.997
    
    temp = start_temp
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0:
        # Generate neighbor
        current_list = list(current_set)
        elem_to_replace = random.choice(current_list)
        
        if random.random() < 0.4:
            # Random replacement
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
        elif random.random() < 0.7:
            # Local nudge (small perturbation)
            delta = random.randint(-8, 8)
            if delta == 0:
                continue
            new_val = elem_to_replace + delta
            if new_val < 1 or new_val > 512:
                continue
            if new_val in current_set:
                continue
        else:
            # Swap with another element
            other_elem = random.choice(current_list)
            if other_elem == elem_to_replace:
                continue
            # Try to find a new value for other_elem
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
            # Apply swap: remove elem_to_replace and other_elem, add new_val and elem_to_replace
            # Actually, let's just do single replacements for simplicity
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
            # Remove elem_to_replace, add new_val
            pass
        
        # Apply move
        new_set = set(current_set)
        new_set.remove(elem_to_replace)
        new_set.add(new_val)
        
        new_score = get_score(new_set)
        delta_score = new_score - current_score
        
        if delta_score >= 0:
            current_set = new_set
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            if temp > 0:
                prob = math.exp(delta_score / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            no_improve_count += 1
        
        # Cool down
        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            temp = max(temp, 2.0)
            # Perturb the current set
            for _ in range(5):
                cl = list(current_set)
                el = random.choice(cl)
                nv = random.randint(1, 512)
                if nv not in current_set:
                    current_set.remove(el)
                    current_set.add(nv)
            current_score = get_score(current_set)
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)

    # Final polish: local search with tighter tolerance
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_temp = 0.01
        polish_set = set(best_set)
        polish_score = best_score
        polish_no_improve = 0
        
        while time.time() < polish_deadline and polish_no_improve < 200:
            cl = list(polish_set)
            el = random.choice(cl)
            nv = random.randint(1, 512)
            if nv in polish_set:
                continue
            
            new_set = set(polish_set)
            new_set.remove(el)
            new_set.add(nv)
            new_score = get_score(new_set)
            
            if new_score > polish_score:
                polish_set = new_set
                polish_score = new_score
                polish_no_improve = 0
                if polish_score > best_score:
                    best_score = polish_score
                    best_set = set(polish_set)
            else:
                polish_no_improve += 1
        
        # Try systematic local moves
        if time.time() < polish_deadline:
            for _ in range(100):
                if time.time() >= polish_deadline:
                    break
                cl = list(best_set)
                el = random.choice(cl)
                for nv in range(max(1, el - 15), min(513, el + 16)):
                    if nv == el or nv in best_set:
                        continue
                    new_set = set(best_set)
                    new_set.remove(el)
                    new_set.add(nv)
                    new_score = get_score(new_set)
                    if new_score > best_score:
                        best_set = new_set
                        best_score = new_score
                        break

    return sorted(best_set)
