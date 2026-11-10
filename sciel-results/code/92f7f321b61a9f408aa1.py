"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = 128
    targets = list(range(128, 257))
    num_targets = len(targets)  # 129
    
    # Precompute for each target t, the list of valid unordered pairs (a, b) with a+b=t, a<=b, a,b in [0,128]
    # For target t, a ranges from max(0, t-128) to min(128, t//2)
    # b = t - a
    target_pairs = {}
    for t in targets:
        a_min = max(0, t - 128)
        a_max = min(128, t // 2)
        pairs = []
        for a in range(a_min, a_max + 1):
            b = t - a
            if a <= b and 0 <= a <= 128 and 0 <= b <= 128:
                pairs.append((a, b))
        target_pairs[t] = pairs
    
    # Objective: minimize max multiplicity over targets [128, 256]
    # Multiplicity of target t = number of unordered pairs (a,b) in A with a+b=t, a<=b
    
    def compute_multiplicities(A_set):
        """Compute multiplicity for each target given set A."""
        mults = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        for i, t in enumerate(targets):
            a_min = max(0, t - 128)
            a_max = min(128, t // 2)
            count = 0
            for a in range(a_min, a_max + 1):
                b = t - a
                if a <= b and a in A_set and b in A_set:
                    count += 1
            mults[i] = count
        return mults
    
    def max_mult(A_set):
        mults = compute_multiplicities(A_set)
        return int(mults.max())
    
    def is_valid(A_set):
        """Check every target has at least one representation."""
        for t in targets:
            a_min = max(0, t - 128)
            a_max = min(128, t // 2)
            found = False
            for a in range(a_min, a_max + 1):
                b = t - a
                if a <= b and a in A_set and b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Start with a good initial solution: the seed [64, 128]
    # Let's try to find a better initial solution
    best_A = set(range(64, 129))
    best_score = max_mult(best_A)
    
    # Try a few random initializations to find a good starting point
    random.seed(42)
    for init_iter in range(50):
        # Random subset of size around 40-60
        size = random.randint(35, 65)
        A_set = set(random.sample(range(129), size))
        # Ensure validity by adding elements if needed
        # Simple repair: for each uncovered target, add a random valid pair element
        for t in targets:
            a_min = max(0, t - 128)
            a_max = min(128, t // 2)
            found = False
            for a in range(a_min, a_max + 1):
                b = t - a
                if a <= b and a in A_set and b in A_set:
                    found = True
                    break
            if not found:
                # Add a random valid pair
                valid_pairs = target_pairs[t]
                if valid_pairs:
                    a, b = random.choice(valid_pairs)
                    A_set.add(a)
                    A_set.add(b)
        
        if is_valid(A_set):
            score = max_mult(A_set)
            if score < best_score:
                best_score = score
                best_A = set(A_set)
        
        if time.time() > deadline - 70.0:
            break
    
    # Simulated Annealing
    current_A = set(best_A)
    current_score = best_score
    
    # Precompute pair lists for efficiency
    # For each element v, precompute which targets it participates in and with what partner
    # Actually, let's use a more efficient delta update
    
    def get_affected_targets(v, A_set):
        """Get targets affected by adding/removing v."""
        affected = []
        for t in targets:
            a_min = max(0, t - 128)
            a_max = min(128, t // 2)
            if a_min <= v <= a_max:
                b = t - v
                if v <= b and b in A_set:
                    affected.append(t)
            elif a_min <= (t - v) <= a_max:
                a = t - v
                if a <= v and a in A_set:
                    affected.append(t)
        return affected
    
    # Simpler: just recompute multiplicities when needed
    # But for SA, we want fast delta updates
    
    # Let's use a different approach: maintain multiplicities array
    # mult[t_idx] = number of representations for target t
    
    def init_mults(A_set):
        mults = np.zeros(num_targets, dtype=np.int32)
        for i, t in enumerate(targets):
            a_min = max(0, t - 128)
            a_max = min(128, t // 2)
            count = 0
            for a in range(a_min, a_max + 1):
                b = t - a
                if a <= b and a in A_set and b in A_set:
                    count += 1
            mults[i] = count
        return mults
    
    current_mults = init_mults(current_A)
    current_max = int(current_mults.max())
    
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.995
    max_iters = 500000
    iter_count = 0
    
    while time.time() < deadline - 5.0 and iter_count < max_iters:
        iter_count += 1
        
        # Choose a random element to toggle
        v = random.randint(0, 128)
        
        if v in current_A:
            # Try removing v
            # Check if removal keeps validity
            # Compute delta multiplicities
            new_mults = current_mults.copy()
            for i, t in enumerate(targets):
                a_min = max(0, t - 128)
                a_max = min(128, t // 2)
                # Count pairs involving v
                if a_min <= v <= a_max:
                    b = t - v
                    if v <= b and b in current_A:
                        new_mults[i] -= 1
                else:
                    a = t - v
                    if a_min <= a <= a_max and a <= v and a in current_A:
                        new_mults[i] -= 1
            
            # Check validity: all mults must be >= 1
            if new_mults.min() >= 1:
                new_max = int(new_mults.max())
                delta = new_max - current_max
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_A.remove(v)
                    current_mults = new_mults
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_A = set(current_A)
            else:
                # Invalid, skip
                pass
        else:
            # Try adding v
            new_mults = current_mults.copy()
            for i, t in enumerate(targets):
                a_min = max(0, t - 128)
                a_max = min(128, t // 2)
                if a_min <= v <= a_max:
                    b = t - v
                    if v <= b and b in current_A:
                        new_mults[i] += 1
                else:
                    a = t - v
                    if a_min <= a <= a_max and a <= v and a in current_A:
                        new_mults[i] += 1
            
            new_max = int(new_mults.max())
            delta = new_max - current_max
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A.add(v)
                current_mults = new_mults
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_A = set(current_A)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final polish: try local search on best_A
    # Try removing elements that don't affect validity
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for v in sorted(best_A):
            test_A = set(best_A)
            test_A.remove(v)
            if is_valid(test_A):
                test_score = max_mult(test_A)
                if test_score <= best_score:
                    best_A = test_A
                    best_score = test_score
                    improved = True
                    break
    
    return list(best_A)
# EVOLVE-BLOCK-END
