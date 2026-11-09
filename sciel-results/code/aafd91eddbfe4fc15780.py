import time
import random
import sys

N_POINTS = 16
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    
    # 1. Generate initial valid linear triple system
    # We use a greedy approach with random restarts to find a dense partial Steiner triple system.
    # A dense system (many triples) generally has a smaller independence number.
    
    all_triples = []
    for i in range(N_POINTS):
        for j in range(i+1, N_POINTS):
            for k in range(j+1, N_POINTS):
                all_triples.append((i, j, k))
    
    random.seed(42)
    
    best_initial_system = []
    best_size = 0
    
    # Multiple random restarts to find a large linear triple system
    for _ in range(50):
        shuffled_triples = all_triples[:]
        random.shuffle(shuffled_triples)
        selected = []
        used_pairs = set()
        
        for t in shuffled_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            if all(p not in used_pairs for p in pairs):
                selected.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_initial_system = selected[:]
            
    current_system = best_initial_system[:]
    
    if not current_system:
        # Fallback
        current_system = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    # Precompute masks for the current system
    def get_masks(system):
        return [ (1<<t[0]) | (1<<t[1]) | (1<<t[2]) for t in system ]
    
    current_masks = get_masks(current_system)
    
    # 2. Define a fast heuristic for independence number
    # We use a local search to find a large independent set.
    # An independent set is a set of points containing no triple.
    # We start with all points and iteratively remove points from triples that are fully contained.
    # To make it fast, we use bitmasks.
    
    def heuristic_alpha(system, masks):
        S_mask = (1 << N_POINTS) - 1
        # We want to find a large S such that no mask in masks is a subset of S.
        # Greedy removal:
        # While there exists a mask m in masks such that (m & S_mask) == m:
        #     Pick one such m.
        #     Remove a point p from m that is in S_mask.
        #     To be smart, remove the point that appears in the most other "active" triples.
        
        # For speed, we'll do a few iterations of this.
        
        # Precompute point counts for current S?
        # Or just pick a random point from the violating triple.
        
        # Let's try a more robust heuristic:
        # 1. Start with S = all points.
        # 2. Repeat until no violation:
        #    Find a violating triple t.
        #    For each point p in t, count how many other triples in the system contain p and are currently "active" (subset of S).
        #    Remove the point p with the highest count.
        
        # This is O(|T|^2) per step in worst case, but |T| is small (~40).
        
        # Optimization: We can just pick the first violating triple and remove a random point from it.
        # This is much faster and might be good enough for SA.
        
        # Let's try the "remove point in most triples" heuristic but optimized.
        
        changed = True
        while changed:
            changed = False
            # Find any violating triple
            viol_idx = -1
            for i, m in enumerate(masks):
                if (m & S_mask) == m:
                    viol_idx = i
                    break
                    
            if viol_idx == -1:
                break
                
            # Remove a point from this triple
            m = masks[viol_idx]
            points = [i for i in range(N_POINTS) if (m >> i) & 1]
            
            # Pick point to remove
            # Heuristic: remove the point that is in the most other triples that are subsets of S_mask
            # But this is expensive. Let's just remove a random point from the triple.
            # Or, remove the point that appears in the most triples overall (precomputed static weight).
            
            # Let's use a static weight: degree of point in the hypergraph.
            # Precompute degrees?
            # For now, random removal is very fast.
            
            p = random.choice(points)
            S_mask &= ~(1 << p)
            changed = True
            
        return bin(S_mask).count('1')
        
    # Let's try a slightly better heuristic:
    # Instead of random removal, we can try to find a maximal independent set using a greedy algorithm on the complement?
    # No, the current one is fine for SA.
    
    # However, the heuristic_alpha above is not deterministic if it uses random.choice.
    # For SA, we need a consistent cost function.
    # Let's make it deterministic: remove the point with the smallest index? Or highest degree?
    
    # Let's precompute point degrees for the current system.
    def compute_degrees(system):
        degrees = [0]*N_POINTS
        for t in system:
            for p in t:
                degrees[p] += 1
        return degrees
        
    current_degrees = compute_degrees(current_system)
    
    def deterministic_alpha(system, masks, degrees):
        S_mask = (1 << N_POINTS) - 1
        changed = True
        while changed:
            changed = False
            viol_idx = -1
            for i, m in enumerate(masks):
                if (m & S_mask) == m:
                    viol_idx = i
                    break
                    
            if viol_idx == -1:
                break
                
            m = masks[viol_idx]
            points = [i for i in range(N_POINTS) if (m >> i) & 1]
            
            # Remove point with highest degree
            # If tie, lowest index
            best_p = min(points, key=lambda p: (-degrees[p], p))
            S_mask &= ~(1 << (best_p))
            changed = True
            
        return bin(S_mask).count('1')
        
    # Initial cost
    current_cost = deterministic_alpha(current_system, current_masks, current_degrees)
    best_system = current_system[:]
    best_cost = current_cost
    
    # 3. Simulated Annealing
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Precompute all possible triples for random selection
    # We already have all_triples
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
            
        if not current_system:
            break
            
        # Generate a move: swap a triple
        idx1 = random.randint(0, len(current_system) - 1)
        old_triple = current_system[idx1]
        
        # Remove old_triple
        temp_system = current_system[:]
        temp_system.pop(idx1)
        temp_masks = get_masks(temp_system)
        
        # Find a new triple that is compatible with temp_system
        # Compatibility: no shared pair.
        
        found_new = False
        new_triple = None
        attempts = 0
        
        # To speed up, we can precompute used pairs in temp_system?
        # That's O(|T|) per move.
        
        used_pairs = set()
        for t in temp_system:
            used_pairs.add((t[0], t[1]))
            used_pairs.add((t[0], t[2]))
            used_pairs.add((t[1], t[2]))
            
        while attempts < 20: # Limit attempts
            candidate = random.choice(all_triples)
            c_pairs = [(candidate[0], candidate[1]), (candidate[0], candidate[2]), (candidate[1], candidate[2])]
            valid = True
            for cp in c_pairs:
                if cp in used_pairs:
                    valid = False
                    break
            if valid:
                new_triple = candidate
                found_new = True
                break
            attempts += 1
            
        if found_new:
            # Perform move
            current_system[idx1] = new_triple
            current_masks = get_masks(current_system)
            current_degrees = compute_degrees(current_system)
            
            new_cost = deterministic_alpha(current_system, current_masks, current_degrees)
            
            # Accept if better or with probability exp(-(delta)/T)
            delta = new_cost - current_cost
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_system = current_system[:]
            else:
                if delta == 0:
                    current_cost = new_cost
                else:
                    if delta < 10 * T: # Rough cutoff
                        if random.random() < pow(2.718281828, -delta / T):
                            current_cost = new_cost
        else:
            pass
            
        T *= cooling_rate
        
    return best_system

# Fallback in case of any error
if __name__ == "__main__":
    sys = construct()
    print(f"System size: {len(sys)}")
    # Verify linearity
    for i in range(len(sys)):
        for j in range(i+1, len(sys)):
            common = set(sys[i]) & set(sys[j])
            if len(common) > 1:
                print("Invalid system!")
                break
