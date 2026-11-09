import random
import time
import itertools

def is_sidon_4(s0, s1, s2, s3):
    """Fast check if 4 elements form a Sidon set (all 10 pair sums unique)."""
    sums = [
        2*s0, s0+s1, s0+s2, s0+s3,
        2*s1, s1+s2, s1+s3,
        2*s2, s2+s3,
        2*s3
    ]
    return len(set(sums)) == 10

def is_sidon_3(a, b, c):
    """Check if 3 elements form a Sidon set (all 6 pair sums unique)."""
    sums = [2*a, a+b, a+c, 2*b, b+c, 2*c]
    return len(set(sums)) == 6

def count_sidon_4_subsets(s):
    """Count number of Sidon 4-subsets in set s. Returns 0 if none."""
    count = 0
    n = len(s)
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    if is_sidon_4(s[i], s[j], s[k], s[l]):
                        count += 1
    return count

def find_first_sidon_4(s):
    """Find first Sidon 4-subset, return tuple or None."""
    n = len(s)
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    if is_sidon_4(s[i], s[j], s[k], s[l]):
                        return (s[i], s[j], s[k], s[l])
    return None

def find_first_sidon_3(s):
    """Find first Sidon 3-subset, return tuple or None."""
    n = len(s)
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                if is_sidon_3(s[i], s[j], s[k]):
                    return (s[i], s[j], s[k])
    return None

def exact_max_sidon_size(s):
    """Exactly compute the largest Sidon subset size in s."""
    n = len(s)
    # Check from size 5 down to 2
    for k in range(5, 1, -1):
        for combo in itertools.combinations(s, k):
            # Check Sidon
            sums = set()
            for i in range(k):
                for j in range(i, k):
                    sm = combo[i] + combo[j]
                    if sm in sums:
                        break
                    sums.add(sm)
                else:
                    continue
                break
            else:
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = None
    best_count = float('inf')
    best_exact = float('inf')
    
    def update_best(s, cnt, exact):
        nonlocal best_set, best_count, best_exact
        # Primary: minimize exact max Sidon size
        # Secondary: minimize count of Sidon-4s (proxy for robustness)
        if exact < best_exact or (exact == best_exact and cnt < best_count):
            best_exact = exact
            best_count = cnt
            best_set = list(s)
            return True
        return False
    
    # Initial candidates
    random.seed(42)
    
    # Try AP
    ap = list(range(12))
    cnt_ap = count_sidon_4_subsets(ap)
    ex_ap = exact_max_sidon_size(ap)
    update_best(ap, cnt_ap, ex_ap)
    
    # Try structured: multiples of some step
    for step in [2, 3, 5, 7]:
        cand = [i * step for i in range(12) if i * step <= 128]
        if len(cand) == 12:
            cnt_c = count_sidon_4_subsets(cand)
            ex_c = exact_max_sidon_size(cand)
            update_best(cand, cnt_c, ex_c)
    
    # Random initializations
    for _ in range(20):
        cand = random.sample(range(129), 12)
        cnt_c = count_sidon_4_subsets(cand)
        ex_c = exact_max_sidon_size(cand)
        if update_best(cand, cnt_c, ex_c):
            if best_exact <= 2:
                return best_set
    
    # Simulated Annealing main loop
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() < deadline:
        # Random restart with some probability
        if random.random() < 0.05 or current_set is None:
            current_set = set(random.sample(range(129), 12))
            current_list = list(current_set)
            current_count = count_sidon_4_subsets(current_list)
            current_exact = exact_max_sidon_size(current_list)
            T = 1.0
            if update_best(current_list, current_count, current_exact):
                if best_exact <= 2:
                    return best_set
            continue
        
        # Generate neighbor: replace one or two elements
        move_type = random.choice([0, 1, 1])  # 0=single, 1=double
        new_list = list(current_set)
        
        if move_type == 0:
            idx = random.randint(0, 11)
            old_val = new_list[idx]
            new_val = random.randint(0, 128)
            if new_val in new_list and new_val != old_val:
                continue
            new_list[idx] = new_val
        else:
            idx1 = random.randint(0, 11)
            idx2 = (idx1 + random.randint(1, 11)) % 12
            old1 = new_list[idx1]
            old2 = new_list[idx2]
            new1 = random.randint(0, 128)
            new2 = random.randint(0, 128)
            # Ensure distinctness
            if new1 in new_list and new1 != old1:
                continue
            new_list[idx1] = new1
            if new2 in new_list and new2 != old2:
                continue
            new_list[idx2] = new2
        
        new_count = count_sidon_4_subsets(new_list)
        new_exact = exact_max_sidon_size(new_list)
        
        # Acceptance criterion
        delta_count = new_count - current_count
        delta_exact = new_exact - current_exact
        
        accept = False
        if delta_exact < 0 or (delta_exact == 0 and delta_count < 0):
            accept = True
        elif delta_exact == 0 and delta_count == 0:
            accept = random.random() < 0.3
        else:
            # Accept worse with probability exp(-delta/T)
            if delta_exact > 0:
                # Much worse, unlikely to accept
                if random.random() < pow(2.718, -10.0 / max(T, 1e-9)):
                    accept = True
            else:
                # Same exact, worse count
                if random.random() < pow(2.718, -delta_count / max(T, 1e-9)):
                    accept = True
        
        if accept:
            current_set = set(new_list)
            current_count = new_count
            current_exact = new_exact
            if update_best(new_list, new_count, new_exact):
                if best_exact <= 2:
                    return best_set
        
        T *= cooling_rate
        
        # If stuck, do targeted move
        if T < min_T:
            # Try to break a specific Sidon subset
            if current_exact >= 4:
                sidon_4 = find_first_sidon_4(list(current_set))
                if sidon_4:
                    # Replace one element from this subset
                    idx_to_move = random.choice(range(4))
                    old_val = sidon_4[idx_to_move]
                    list_cs = list(current_set)
                    pos = list_cs.index(old_val)
                    
                    for _ in range(30):
                        new_val = random.randint(0, 128)
                        if new_val in current_set and new_val != old_val:
                            continue
                        new_list = list(current_set)
                        new_list[pos] = new_val
                        new_count = count_sidon_4_subsets(new_list)
                        new_exact = exact_max_sidon_size(new_list)
                        if new_exact < current_exact or (new_exact == current_exact and new_count < current_count):
                            current_set = set(new_list)
                            current_count = new_count
                            current_exact = new_exact
                            if update_best(new_list, new_count, new_exact):
                                if best_exact <= 2:
                                    return best_set
                            break
                    T = 1.0  # Reset temperature after targeted move
            
            elif current_exact == 3:
                # Try to break a Sidon 3-subset
                sidon_3 = find_first_sidon_3(list(current_set))
                if sidon_3:
                    idx_to_move = random.choice(range(3))
                    old_val = sidon_3[idx_to_move]
                    list_cs = list(current_set)
                    pos = list_cs.index(old_val)
                    
                    for _ in range(30):
                        new_val = random.randint(0, 128)
                        if new_val in current_set and new_val != old_val:
                            continue
                        new_list = list(current_set)
                        new_list[pos] = new_val
                        new_count = count_sidon_4_subsets(new_list)
                        new_exact = exact_max_sidon_size(new_list)
                        if new_exact < current_exact:
                            current_set = set(new_list)
                            current_count = new_count
                            current_exact = new_exact
                            if update_best(new_list, new_count, new_exact):
                                if best_exact <= 2:
                                    return best_set
                            break
                    T = 1.0
            
            else:
                T = 1.0
    
    # Final polish: if we have a good candidate, try to improve further
    if best_set is not None:
        # Verify and potentially polish
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            cand = list(best_set)
            idx = random.randint(0, 11)
            old_val = cand[idx]
            new_val = random.randint(0, 128)
            if new_val in cand and new_val != old_val:
                continue
            cand[idx] = new_val
            cnt_c = count_sidon_4_subsets(cand)
            ex_c = exact_max_sidon_size(cand)
            if ex_c < best_exact or (ex_c == best_exact and cnt_c < best_count):
                best_exact = ex_c
                best_count = cnt_c
                best_set = cand
                if best_exact <= 2:
                    break
    
    # Ensure we return exactly 12 distinct integers in [0,128]
    if best_set is None:
        best_set = list(range(12))
    
    # Validate
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
