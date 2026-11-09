import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin (150s limit)
    
    def evaluate(S):
        """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S."""
        S_set = set(S)
        valid_diffs = set()
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def evaluate_fast(S):
        """Faster evaluation using set operations."""
        S_set = set(S)
        valid_diffs = set()
        for a in S:
            for b in S:
                if b <= a:
                    continue
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def random_init():
        return random.sample(range(1, 513), 32)

    def structured_init():
        """Create a structured set with good progression properties."""
        # Use a mix of arithmetic progressions
        # Base: start at 1, add elements to cover many differences
        pts = [1]
        # Add elements to create many 3-term APs
        # Try a geometric-like spread with some arithmetic structure
        for i in range(1, 32):
            # Mix of linear and quadratic growth
            base = 1 + i * 16  # Spread across 1..512
            jitter = random.randint(-3, 3)
            val = max(1, min(512, base + jitter))
            if val not in pts:
                pts.append(val)
            else:
                # Find a nearby unused value
                for offset in range(1, 10):
                    for dv in [offset, -offset]:
                        v = val + dv
                        if 1 <= v <= 512 and v not in pts:
                            pts.append(v)
                            break
                    if len(pts) == 32:
                        break
            if len(pts) == 32:
                break
        # Fill remaining if needed
        while len(pts) < 32:
            v = random.randint(1, 512)
            if v not in pts:
                pts.append(v)
        return pts[:32]

    def dense_cluster_init():
        """Create clusters that form many APs within themselves."""
        pts = []
        # Create 4 clusters of 8 elements each, each an arithmetic progression
        cluster_starts = [1, 130, 259, 388]
        for cs in cluster_starts:
            for k in range(8):
                val = cs + k * 16
                if 1 <= val <= 512 and val not in pts:
                    pts.append(val)
        while len(pts) < 32:
            v = random.randint(1, 512)
            if v not in pts:
                pts.append(v)
        return pts[:32]

    # Initial candidates
    candidates = [
        list(range(1, 33)),       # Simple AP
        structured_init(),         # Structured
        dense_cluster_init(),      # Clusters
        random_init(),
        random_init(),
        random_init(),
        random_init(),
    ]
    
    # Evaluate initial candidates
    best_set = None
    best_score = -1
    for c in candidates:
        s = evaluate(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 50.0  # Initial temperature
    T_min = 0.5
    cooling_rate = 0.9995
    
    while time.time() < start_time + timeout:
        # Check time budget
        if time.time() - start_time > timeout - 5:
            break
            
        # Generate a neighbor
        move_type = random.random()
        new_set = current_set[:]
        
        if move_type < 0.4:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
        elif move_type < 0.7:
            # Change one element to a new random position
            i = random.randrange(32)
            new_val = random.randint(1, 512)
            if new_val in new_set:
                continue
            new_set[i] = new_val
        elif move_type < 0.85:
            # Move one element to a nearby position
            i = random.randrange(32)
            old_val = new_set[i]
            delta = random.randint(-20, 20)
            new_val = old_val + delta
            if 1 <= new_val <= 512 and new_val not in new_set:
                new_set[i] = new_val
            else:
                continue
        else:
            # Add and remove: remove one, add another
            i = random.randrange(32)
            removed = new_set.pop(i)
            new_val = random.randint(1, 512)
            attempts = 0
            while new_val in new_set and attempts < 20:
                new_val = random.randint(1, 512)
                attempts += 1
            if new_val in new_set:
                new_set.append(removed)
                continue
            new_set.append(new_val)
        
        # Check distinctness
        if len(set(new_set)) != 32:
            continue
            
        new_score = evaluate(new_set)
        
        # Acceptance criterion (simulated annealing)
        delta_score = new_score - current_score
        if delta_score >= 0 or random.random() < np.exp(delta_score / T):
            current_score = new_score
            current_set = new_set
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Occasionally restart with a new random init if stuck
        if random.random() < 0.001:
            new_init = random_init()
            new_init_score = evaluate(new_init)
            if new_init_score > current_score:
                current_set = new_init
                current_score = new_init_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            T = 50.0  # Reheat
    
    return best_set
