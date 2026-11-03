import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # Compute sumset and difference set
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        size_s = len(sums)
        size_d = len(diffs)
        
        if size_s == 0:
            return 0.0
            
        ratio = size_d / size_s
        if ratio <= 0:
            return 0.0
            
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    best_set = [0, 1, 3]
    best_score = calc_score(best_set)
    
    # Simulated annealing / local search
    random.seed(42)
    
    def local_search(U, max_steps=2000, initial_temp=1.0, cooling_rate=0.999):
        """Local search with simulated annealing"""
        U = sorted(set(U))
        current_score = calc_score(U)
        best_local = U[:]
        best_local_score = current_score
        temp = initial_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                return best_local, best_local_score
            
            # Random move: add, remove, or shift an element
            move_type = random.random()
            new_U = U[:]
            
            if move_type < 0.3 and len(U) < 30:
                # Add a new element
                max_u = max(U) if U else 0
                new_elem = random.randint(1, max_u + 10)
                if new_elem not in new_U:
                    new_U.append(new_elem)
            elif move_type < 0.5 and len(U) > 3:
                # Remove an element (keep 0)
                removable = [x for x in new_U if x != 0]
                if removable:
                    new_U.remove(random.choice(removable))
            elif move_type < 0.8:
                # Shift an element
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)  # Don't move 0
                    old_val = new_U[idx]
                    shift = random.randint(-5, 5)
                    new_val = max(0, old_val + shift)
                    if new_val not in new_U or new_val == old_val:
                        new_U[idx] = new_val
                    else:
                        new_U[idx] = new_val
                        if len(new_U) > 1:
                            # Try to find a spot where it doesn't duplicate
                            while new_val in [x for x in new_U if x != new_U[idx]]:
                                new_val = max(0, new_val + random.choice([-1, 1]))
                            new_U[idx] = new_val
            else:
                # Replace an element with a random one
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)
                    max_u = max(U) if U else 10
                    new_U[idx] = random.randint(1, max_u + 5)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U or len(new_U) < 2:
                continue
                
            new_score = calc_score(new_U)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                U = new_U
                current_score = new_score
                if current_score > best_local_score:
                    best_local = U[:]
                    best_local_score = current_score
            
            temp *= cooling_rate
        
        return best_local, best_local_score

    # Try multiple starting points
    initial_configs = []
    
    # Geometric progressions
    for r in range(2, 8):
        for n in range(4, 10):
            U = [r**i for i in range(n)]
            if 0 not in U:
                U.insert(0, 0)
            initial_configs.append(sorted(set(U)))
    
    # Golomb-like rulers (distinct differences)
    for n in range(4, 12):
        # Simple Golomb construction
        U = []
        for i in range(n):
            U.append(i * (n - i))
        if 0 not in U:
            U.insert(0, 0)
        initial_configs.append(sorted(set(U)))
    
    # Random dense sets
    for _ in range(20):
        n = random.randint(5, 15)
        max_val = random.randint(n * 2, n * 5)
        U = [0] + sorted(random.sample(range(1, max_val + 1), min(n - 1, max_val)))
        initial_configs.append(U)
    
    # Run local search from each starting point
    for i, init_U in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        result_U, result_score = local_search(init_U, max_steps=1000)
        
        if result_score > best_score:
            best_score = result_score
            best_set = result_U
    
    # Final refinement: try to improve the best set found
    if time.time() < deadline - 10:
        result_U, result_score = local_search(best_set, max_steps=3000, initial_temp=0.5, cooling_rate=0.9995)
        if result_score > best_score:
            best_score = result_score
            best_set = result_U
    
    return best_set
