import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096 possible subsets
    
    # Precompute subset relationships for speed? 
    # Actually, direct bitmask check (a & b) == a is very fast in Python.
    # Let's stick to direct checks but optimize the loop.
    
    # Initial solution: A random set of 40 distinct subsets
    # Better initial solution: Start with a "dense" structure
    # Let's use a greedy construction to get a good starting point, 
    # then run simulated annealing to improve it.
    
    # 1. Greedy Initialization
    family = []
    family_set = set()
    
    # Add empty set and full set first? Or just random?
    # Let's try a greedy approach to build a strong initial chain-like structure
    # that maximizes immediate gains.
    
    # Start with a random subset to break symmetry, or just start with empty?
    # Let's start with a few random ones to diversify, then greedy.
    
    # Actually, a known good structure for this problem involves sets of similar sizes
    # that are nested. 
    # Let's just use a simple greedy: pick the set that adds the most comparables.
    
    # To speed up initialization, we can just pick random distinct sets.
    # But greedy is better for SA start.
    
    # Greedy step:
    # We have 4096 candidates. We pick 40.
    # For each step, we evaluate gain of each candidate against current family.
    # This is O(m * N * m) = O(40 * 4096 * 40) ~ 6.5M ops per step.
    # 40 steps -> 260M ops. Might be too slow in Python for 85s?
    # 260M simple ops in Python is likely ~20-30s.
    # Let's optimize the gain calculation.
    
    # Alternative: Random initialization is O(1). Then SA does the work.
    # SA with good move generation can find good solutions.
    # Let's try Random Init + SA.
    
    # Generate 40 distinct random subsets
    all_subsets = list(range(N))
    random.shuffle(all_subsets)
    family = all_subsets[:m]
    family_set = set(family)
    
    # Calculate initial score
    def calc_score(fam):
        score = 0
        m = len(fam)
        for i in range(m):
            a = fam[i]
            for j in range(i+1, m):
                b = fam[j]
                if (a & b) == a or (a & b) == b:
                    score += 1
        return score

    current_score = calc_score(family)
    best_score = current_score
    best_family = family[:]
    
    # Simulated Annealing
    # Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.999  # Cool slowly
    T_min = 1e-5
    
    # Move: Pick a random index i, replace family[i] with a random subset s
    # Not in family.
    
    # To make moves more effective, we can bias the new subset to be similar to the old one?
    # Or just random.
    
    # Optimization: Precompute comparability matrix?
    # 4096x4096 matrix is 16M entries. Too much memory/time to precompute fully.
    # But we can precompute subset relationships for the 4096 sets?
    # Actually, (a & b) == a is very fast.
    
    # Let's run the loop
    t = start_time
    steps = 0
    
    while t < deadline - 1.0:
        # Check time every 1000 steps to avoid overhead
        if steps % 1000 == 0:
            t = time.time()
            if t > deadline - 1.0:
                break
        
        # Generate move
        i = random.randint(0, m - 1)
        old_s = family[i]
        
        # Pick a new random subset
        new_s = random.randint(0, N - 1)
        
        # Ensure distinctness
        if new_s in family_set:
            # Try a few times to find a valid new_s
            for _ in range(10):
                new_s = random.randint(0, N - 1)
                if new_s not in family_set:
                    break
            else:
                # If we still collide, skip this step
                steps += 1
                continue
        
        # Calculate delta score
        # Remove old_s, add new_s
        # Delta = (gain(new_s) - gain(old_s))
        # gain(s) = number of sets in family (excluding s itself) comparable to s
        
        # Calculate gain of old_s against current family (excluding old_s)
        gain_old = 0
        for k in range(m):
            if k == i:
                continue
            s_k = family[k]
            if (old_s & s_k) == old_s or (old_s & s_k) == s_k:
                gain_old += 1
                
        # Calculate gain of new_s against current family (excluding old_s, which is being replaced)
        # Note: family still contains old_s at index i.
        # We compare new_s with all s_k where k != i.
        gain_new = 0
        for k in range(m):
            if k == i:
                continue
            s_k = family[k]
            if (new_s & s_k) == new_s or (new_s & s_k) == s_k:
                gain_new += 1
        
        delta = gain_new - gain_old
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # P = exp(delta / T)
            # delta is negative
            # If T is very small, exp(neg/T) is 0
            if T < 1e-5:
                accept = False
            else:
                # Optimize: if delta is very negative, just reject
                if delta < -10 * T:
                    accept = False
                else:
                    accept = random.random() < (2.718281828 ** (delta / T))
        
        if accept:
            # Perform move
            family[i] = new_s
            family_set.remove(old_s)
            family_set.add(new_s)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
        
        # Cool down
        T *= cooling_rate
        
        steps += 1
        
    # Return best family found
    # Convert bitmasks to lists
    result = []
    for s in best_family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    return result

