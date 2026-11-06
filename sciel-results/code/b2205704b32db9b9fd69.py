import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin
    
    T_min = 128
    T_max = 256
    N_max = 129  # elements from 0 to 128
    
    def compute_counts(A_set):
        """Compute representation counts for all sums in [T_min, T_max]."""
        A_list = sorted(A_set)
        n = len(A_list)
        counts = np.zeros(T_max + 1, dtype=np.int32)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if T_min <= s <= T_max:
                counts[s] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > T_max:
                    break
                if s >= T_min:
                    counts[s] += 1
        return counts
    
    def get_max_count(counts):
        return int(np.max(counts[T_min:T_max+1]))
    
    def is_valid(counts):
        return np.all(counts[T_min:T_max+1] > 0)
    
    def quick_eval(A_set):
        """Quick evaluation returning (valid, max_count)."""
        counts = compute_counts(A_set)
        valid = is_valid(counts)
        if not valid:
            return False, float('inf')
        return True, get_max_count(counts)
    
    best_A = None
    best_max = float('inf')
    
    def update_best(A_set, max_count):
        nonlocal best_A, best_max
        if max_count < best_max:
            best_max = max_count
            best_A = set(A_set)
    
    def try_improve(A_initial, label=""):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        
        # Repair if needed
        counts = compute_counts(A_set)
        if not is_valid(counts):
            for _ in range(100):
                gaps = np.where(counts[T_min:T_max+1] == 0)[0] + T_min
                if len(gaps) == 0:
                    break
                best_y = None
                best_gain = -1
                candidates = [y for y in range(N_max) if y not in A_set]
                random.shuffle(candidates)
                for y in candidates[:30]:
                    gain = 0
                    for s in gaps:
                        a = s - y
                        if a in A_set or a == y:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                if best_y is not None and best_gain > 0:
                    A_set.add(best_y)
                    counts = compute_counts(A_set)
                else:
                    break
            
            if not is_valid(counts):
                return
        
        current_max = get_max_count(counts)
        update_best(A_set, current_max)
        
        # Local search with numpy-accelerated evaluation
        A_cur = set(A_set)
        max_cur = current_max
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 1.0:
            # Pick a random element to remove and a random element to add
            A_list = list(A_cur)
            x = random.choice(A_list)
            y = random.randint(0, N_max - 1)
            if y in A_cur:
                continue
            
            # Simulate swap
            A_new = A_cur - {x}
            A_new.add(y)
            
            valid, m_new = quick_eval(A_new)
            if not valid:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
            
            if m_new < max_cur:
                A_cur = A_new
                max_cur = m_new
                no_improve = 0
                update_best(A_cur, max_cur)
            elif m_new == max_cur:
                # Accept with small probability to explore
                if random.random() < 0.1:
                    A_cur = A_new
                    # max_cur unchanged
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        # Multi-move perturbation to escape local minima
        if time.time() < deadline - 3.0:
            A_perturb = set(A_cur)
            for _ in range(random.randint(2, 5)):
                x = random.choice(list(A_perturb))
                A_perturb.remove(x)
                y = random.randint(0, N_max - 1)
                if y not in A_perturb:
                    A_perturb.add(y)
            
            # Try to repair and improve from perturbed state
            counts_p = compute_counts(A_perturb)
            if is_valid(counts_p):
                m_p = get_max_count(counts_p)
                if m_p < max_cur:
                    A_cur = A_perturb
                    max_cur = m_p
                    update_best(A_cur, max_cur)
                    # Continue local search from new state
                    no_improve = 0
                    while time.time() < deadline - 1.0 and no_improve < max_no_improve:
                        A_list = list(A_cur)
                        x = random.choice(A_list)
                        y = random.randint(0, N_max - 1)
                        if y in A_cur:
                            continue
                        A_new = A_cur - {x}
                        A_new.add(y)
                        valid, m_new = quick_eval(A_new)
                        if not valid:
                            no_improve += 1
                            continue
                        if m_new < max_cur:
                            A_cur = A_new
                            max_cur = m_new
                            no_improve = 0
                            update_best(A_cur, max_cur)
                        elif m_new == max_cur and random.random() < 0.1:
                            A_cur = A_new
                        else:
                            no_improve += 1
    
    # 1. Start with a good seed: interval [64, 128]
    A_seed = set(range(64, 129))
    try_improve(A_seed, "seed")
    
    # 2. Try different random initializations with various sizes
    if time.time() < deadline - 10:
        for N in [40, 50, 60, 70]:
            for trial in range(2):
                if time.time() > deadline - 5:
                    break
                random.seed(trial * 1000 + N)
                A_rand = set(random.sample(range(N_max), N))
                try_improve(A_rand, f"rand_{N}_{trial}")
    
    # 3. Try structured initializations
    if time.time() < deadline - 10:
        # Arithmetic progressions
        for step in [2, 3, 4]:
            if time.time() > deadline - 5:
                break
            A_ap = set(range(0, N_max, step))
            try_improve(A_ap, f"ap_{step}")
        
        # Symmetric sets
        A_sym = set(range(0, 65)) | set(range(64, 129))
        try_improve(A_sym, "sym")
    
    # 4. Final polish: try to reduce max further with more aggressive swaps
    if best_A is not None and time.time() < deadline - 2.0:
        A_cur = set(best_A)
        max_cur = best_max
        no_improve = 0
        while time.time() < deadline - 0.5 and no_improve < 2000:
            A_list = list(A_cur)
            x = random.choice(A_list)
            y = random.randint(0, N_max - 1)
            if y in A_cur:
                continue
            A_new = A_cur - {x}
            A_new.add(y)
            valid, m_new = quick_eval(A_new)
            if not valid:
                no_improve += 1
                continue
            if m_new < max_cur:
                A_cur = A_new
                max_cur = m_new
                no_improve = 0
                update_best(A_cur, max_cur)
            elif m_new == max_cur and random.random() < 0.05:
                A_cur = A_new
            else:
                no_improve += 1
    
    if best_A is None:
        return list(range(64, 129))
    
    return list(best_A)
