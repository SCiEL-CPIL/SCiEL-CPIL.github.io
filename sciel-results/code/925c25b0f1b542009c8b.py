import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_score(S):
        S_set = set(S)
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                if b + d in S_set:
                    d_set.add(d)
        return len(d_set)

    def generate_initial_sets():
        sets = []
        
        # 1. Interval
        sets.append(set(range(1, 33)))
        
        # 2. Grid-like structures with various strides
        for stride in [5, 10, 20, 50]:
            points = []
            for i in range(1, 512 // stride + 1):
                for j in range(1, stride + 1):
                    val = i * stride + j
                    if 1 <= val <= 512:
                        points.append(val)
            if len(points) >= 32:
                sets.append(set(random.sample(points, 32)))
        
        # 3. Sums of two squares
        candidates = set()
        for i in range(1, 25):
            for j in range(1, 25):
                val = i*i + j*j
                if 1 <= val <= 512:
                    candidates.add(val)
        if len(candidates) >= 32:
            sets.append(set(random.sample(sorted(candidates), 32)))
            
        # 4. Random sets
        for _ in range(10):
            sets.append(set(random.sample(range(1, 513), 32)))
            
        return sets

    # Find the best initial set
    init_sets = generate_initial_sets()
    best_init_set = max(init_sets, key=get_score)
    best_init_score = get_score(best_init_set)
    
    best_set = set(best_init_set)
    best_score = best_init_score
    
    # Simulated Annealing
    current_set = set(best_init_set)
    current_score = best_init_score
    
    # Parameters
    start_temp = 15.0
    end_temp = 0.1
    cooling_rate = 0.9995
    
    temp = start_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if time.time() > deadline - 5.0:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Single swap
            elem_to_replace = random.choice(list(current_set))
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
            new_set = set(current_set)
            new_set.remove(elem_to_replace)
            new_set.add(new_val)
        elif move_type < 0.8:
            # Double swap
            elems = list(current_set)
            idx1, idx2 = random.sample(range(len(elems)), 2)
            e1, e2 = elems[idx1], elems[idx2]
            new_vals = set()
            while len(new_vals) < 2:
                new_vals.add(random.randint(1, 512))
            remaining = current_set - {e1, e2}
            valid_new_vals = [v for v in new_vals if v not in remaining]
            if len(valid_new_vals) < 2:
                continue
            new_set = set(current_set)
            new_set.remove(e1)
            new_set.remove(e2)
            new_set.update(valid_new_vals)
        else:
            # Perturbation: replace 3 random elements
            elems = list(current_set)
            to_replace = random.sample(elems, 3)
            remaining = current_set - set(to_replace)
            new_vals = set()
            while len(new_vals) < 3:
                v = random.randint(1, 512)
                if v not in remaining:
                    new_vals.add(v)
            new_set = set(current_set)
            for e in to_replace:
                new_set.remove(e)
            new_set.update(new_vals)

        new_score = get_score(new_set)
        delta = new_score - current_score
        
        if delta >= 0:
            current_set = new_set
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            if temp > 0:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            no_improve_count += 1
            
        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp
            
        if no_improve_count > max_no_improve:
            current_set = set(best_set)
            current_score = best_score
            temp = start_temp
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)

    return sorted(best_set)
