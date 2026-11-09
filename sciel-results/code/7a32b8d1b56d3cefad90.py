import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs
    aps = []
    ap_by_elem = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            p = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(p)
            idx = len(aps) - 1
            ap_by_elem[p[0]].append(idx)
            ap_by_elem[p[1]].append(idx)
            ap_by_elem[p[2]].append(idx)
            ap_by_elem[p[3]].append(idx)

    num_aps = len(aps)
    
    # For each element, precompute the other 3 elements in each AP containing it
    ap_others = [[] for _ in range(N)]
    for i, (a, b, c, d) in enumerate(aps):
        ap_others[a].append((b, c, d))
        ap_others[b].append((a, c, d))
        ap_others[c].append((a, b, d))
        ap_others[d].append((a, b, c))

    def count_violations(x):
        """Count number of violated 4-APs (all 4 elements present)."""
        violations = 0
        for (a, b, c, d) in aps:
            if (x >> a) & 1 and (x >> b) & 1 and (x >> c) & 1 and (x >> d) & 1:
                violations += 1
        return violations

    def delta_violations(x, flip_bit):
        """Compute change in violation count if we flip bit flip_bit in state x."""
        other_count = 0
        for (o1, o2, o3) in ap_others[flip_bit]:
            if (x >> o1) & 1 and (x >> o2) & 1 and (x >> o3) & 1:
                other_count += 1
        if (x >> flip_bit) & 1:
            return -other_count  # Clearing the bit reduces violations
        else:
            return other_count  # Setting the bit increases violations

    def walksat(x, max_steps=50000, p_flip=0.8):
        """WalkSAT-style local search to minimize violations."""
        current_violations = count_violations(x)
        best_x = x
        best_violations = current_violations
        rng = random.Random()
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if current_violations == 0:
                return x, 0
            
            # Find violated APs
            violated_aps = []
            for i in range(num_aps):
                a, b, c, d = aps[i]
                if (x >> a) & 1 and (x >> b) & 1 and (x >> c) & 1 and (x >> d) & 1:
                    violated_aps.append((a, b, c, d))
            
            if not violated_aps:
                break
            
            # Pick a random violated AP
            a, b, c, d = rng.choice(violated_aps)
            
            # With probability p_flip, pick the best flip (minimizes delta)
            # Otherwise, pick a random flip from the AP
            if rng.random() < p_flip:
                best_bit = None
                best_delta = None
                for bit in (a, b, c, d):
                    dv = delta_violations(x, bit)
                    if best_delta is None or dv < best_delta:
                        best_delta = dv
                        best_bit = bit
                flip_bit = best_bit
            else:
                flip_bit = rng.choice((a, b, c, d))
            
            # Compute delta BEFORE flipping
            dv = delta_violations(x, flip_bit)
            
            # Flip the bit
            x ^= (1 << flip_bit)
            current_violations += dv
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_x = x
            
            # Occasionally recompute to fix any drift
            if step % 2000 == 0:
                current_violations = count_violations(x)
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_x = x
        
        # Return the best found
        if count_violations(best_x) < count_violations(x):
            return best_x, count_violations(best_x)
        return x, count_violations(x)

    def polish(x):
        """Try to add any element that doesn't create a 4-AP."""
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if not (x >> i) & 1:
                    safe = True
                    for (o1, o2, o3) in ap_others[i]:
                        if (x >> o1) & 1 and (x >> o2) & 1 and (x >> o3) & 1:
                            safe = False
                            break
                    if safe:
                        x |= (1 << i)
                        changed = True
        return x

    # Phase 1: Generate diverse initial solutions via random greedy
    random.seed(123)
    best_x = 0
    best_size = 0
    
    for init_i in range(50):
        if time.time() - start_time > time_limit:
            break
        
        x = 0
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            safe = True
            for (o1, o2, o3) in ap_others[i]:
                if (x >> o1) & 1 and (x >> o2) & 1 and (x >> o3) & 1:
                    safe = False
                    break
            if safe:
                x |= (1 << i)
        
        x = polish(x)
        sz = bin(x).count('1')
        if sz > best_size:
            best_size = sz
            best_x = x
    
    # Phase 2: WalkSAT-style local search from multiple starting points
    starting_points = [best_x]
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        x = 0
        for i in range(N):
            if random.random() < 0.3:
                x |= (1 << i)
        starting_points.append(x)
    
    for sp in starting_points:
        if time.time() - start_time > time_limit:
            break
        
        x, violations = walksat(sp, max_steps=30000, p_flip=0.7)
        x = polish(x)
        sz = bin(x).count('1')
        
        if sz > best_size:
            best_size = sz
            best_x = x
    
    # Phase 3: Final polish and more WalkSAT
    best_x = polish(best_x)
    best_size = bin(best_x).count('1')
    
    if time.time() - start_time < time_limit - 5:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            x, violations = walksat(best_x, max_steps=20000, p_flip=0.6)
            x = polish(x)
            sz = bin(x).count('1')
            if sz > best_size:
                best_size = sz
                best_x = x
    
    best_x = polish(best_x)
    best_size = bin(best_x).count('1')
    
    result = [i + 1 for i in range(N) if (best_x >> i) & 1]
    return sorted(result)
