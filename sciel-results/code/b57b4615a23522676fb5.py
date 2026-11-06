import time
import random
import sys
import numpy as np
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    arr = np.array(A, dtype=np.int64)
    # Upper triangle indices (i <= j)
    iu, ju = np.triu_indices(n)
    sums = arr[iu] + arr[ju]
    prods = arr[iu] * arr[ju]
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def compute_metrics_partial(A, idx):
    """Compute metric when only element at idx might change. Returns (n_sums, n_prods)."""
    n = len(A)
    arr = np.array(A, dtype=np.int64)
    iu, ju = np.triu_indices(n)
    sums = arr[iu] + arr[ju]
    prods = arr[iu] * arr[ju]
    return len(np.unique(sums)), len(np.unique(prods))

def smooth_numbers(limit, max_exponent=6):
    """Generate smooth numbers (numbers with only small prime factors) up to limit."""
    primes = [2, 3, 5, 7, 11, 13]
    smooths = set()
    
    def gen(idx, current):
        if current > limit:
            return
        smooths.add(current)
        if idx >= len(primes):
            return
        p = primes[idx]
        while current * p <= limit:
            current *= p
            gen(idx, current)
            if current > limit:
                break
        gen(idx + 1, current)
    
    gen(0, 1)
    return sorted([s for s in smooths if 1 <= s <= limit])

def greedy_collision_construct(smooth_pool, n_target=20, deadline=None):
    """
    Greedily build a set of n_target elements from smooth_pool by maximizing
    collision density: at each step, pick the candidate that, when added to the
    current set, causes the most sum/product collisions (i.e., minimizes the
    increase in |A+A| and |A*A|).
    
    This models a maximum-weight-clique-like selection where edge weights
    represent collision benefit.
    """
    if deadline is not None and time.time() > deadline:
        return list(smooth_pool[:n_target])
    
    # Start with a few "anchor" elements that have high internal collision
    # Use small smooth numbers as anchors (they collide heavily)
    small_anchors = [x for x in smooth_pool if x <= 30][:5]
    if len(small_anchors) < 3:
        small_anchors = smooth_pool[:3]
    
    current_set = small_anchors[:]
    
    # Precompute current sums and prods
    def get_sums_prods(S):
        n = len(S)
        s_arr = np.array(S, dtype=np.int64)
        iu, ju = np.triu_indices(n)
        sums = s_arr[iu] + s_arr[ju]
        prods = s_arr[iu] * s_arr[ju]
        return set(sums.tolist()), set(prods.tolist())
    
    cur_sums, cur_prods = get_sums_prods(current_set)
    
    while len(current_set) < n_target:
        if deadline is not None and time.time() > deadline:
            break
        
        best_candidate = None
        best_score = float('inf')
        
        # Evaluate candidates: prefer smooth numbers not already in set
        in_set = set(current_set)
        candidates = [x for x in smooth_pool if x not in in_set]
        
        if not candidates:
            # Fallback: random numbers
            candidates = random.sample(range(1, 10001), min(50, 10000 - len(in_set)))
            candidates = [x for x in candidates if x not in in_set]
        
        # For efficiency, evaluate a subset of candidates (top by "smoothness")
        # Smooth numbers with more factorizations tend to collide more
        # Prioritize candidates that are multiples/divisors of existing elements
        scored_candidates = []
        for c in candidates[:200]:  # Limit evaluation for speed
            # Quick heuristic: count how many existing elements share a factor with c
            shared_factors = 0
            for e in current_set:
                if c % e == 0 or e % c == 0:
                    shared_factors += 1
                elif math_gcd(c, e) > 1:
                    shared_factors += 0.5
            scored_candidates.append((shared_factors, c))
        
        scored_candidates.sort(key=lambda x: -x[0])
        
        # Evaluate top candidates fully
        eval_count = min(30, len(scored_candidates))
        for _, c in scored_candidates[:eval_count]:
            new_set = current_set + [c]
            n = len(new_set)
            s_arr = np.array(new_set, dtype=np.int64)
            iu, ju = np.triu_indices(n)
            new_sums = s_arr[iu] + s_arr[ju]
            new_prods = s_arr[iu] * s_arr[ju]
            n_sums = len(np.unique(new_sums))
            n_prods = len(np.unique(new_prods))
            score = max(n_sums, n_prods)
            if score < best_score:
                best_score = score
                best_candidate = c
        
        if best_candidate is None:
            # Pick a random candidate
            best_candidate = random.choice(candidates) if candidates else random.randint(1, 10000)
            while best_candidate in in_set:
                best_candidate = random.randint(1, 10000)
        
        current_set.append(best_candidate)
        cur_sums, cur_prods = get_sums_prods(current_set)
    
    return current_set[:n_target]

def math_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def compute_metrics_fast(A):
    """Faster metric computation using incremental approach for single-element changes."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))

def generate_structured_candidates(deadline=None):
    """Generate diverse structured initial candidates."""
    candidates = []
    
    # 1. Arithmetic progression
    ap = list(range(1, 21))
    candidates.append(ap)
    
    # 2. Smooth numbers (many factorizations -> product collisions)
    smooths = smooth_numbers(10000)
    if len(smooths) >= 20:
        candidates.append(smooths[:20])
        small_smooths = [s for s in smooths if s <= 500]
        if len(small_smooths) >= 20:
            candidates.append(small_smooths[:20])
        mid_smooths = [s for s in smooths if 100 <= s <= 2000]
        if len(mid_smooths) >= 20:
            candidates.append(mid_smooths[:20])
    
    # 3. Greedy collision-dense construction from smooth pool
    if deadline is not None and time.time() < deadline - 20:
        greedy_set = greedy_collision_construct(smooths, 20, deadline=deadline - 15)
        if len(greedy_set) == 20:
            candidates.append(greedy_set)
    
    # 4. Geometric progression
    gp = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    while len(gp) < 20:
        new_val = random.randint(4097, 10000)
        if new_val not in gp:
            gp.append(new_val)
    candidates.append(gp)
    
    # 5. Mix of small consecutive and smooth numbers
    mixed = list(range(1, 11))
    smooths_large = [s for s in smooths if s > 10]
    random.shuffle(smooths_large)
    for s in smooths_large:
        if len(mixed) >= 20:
            break
        if s not in mixed:
            mixed.append(s)
    candidates.append(mixed)
    
    # 6. Random set
    random_set = random.sample(range(1, 10001), 20)
    candidates.append(random_set)
    
    # 7. Powers of 2 with some variations
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    extra = random.sample(range(513, 10001), 10)
    candidates.append(powers2 + extra)
    
    # 8. Consecutive even numbers
    even_ap = list(range(2, 41, 2))
    candidates.append(even_ap)
    
    # 9. Numbers with many divisors (highly composite-like)
    highly_composite = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]
    candidates.append(highly_composite)
    
    # 10. Logarithmic spacing
    import math
    log_spaced = []
    for i in range(20):
        val = int(round(math.exp(i * 0.3)))
        if val < 1:
            val = 1
        if val > 10000:
            val = 10000
        if val not in log_spaced:
            log_spaced.append(val)
    while len(log_spaced) < 20:
        new_val = random.randint(1, 10000)
        if new_val not in log_spaced:
            log_spaced.append(new_val)
    candidates.append(log_spaced[:20])
    
    # 11. Two-cluster structure
    cluster1 = list(range(1, 11))
    cluster2 = [x * 100 for x in range(1, 11)]
    candidates.append(cluster1 + cluster2)
    
    return candidates

def simulated_annealing(A, start_temp, end_temp, start_time, deadline, best_A, best_score):
    """Run simulated annealing from initial state A with improved move generation."""
    current_A = A[:]
    current_score = compute_metrics(current_A)
    
    if current_score < best_score:
        best_score = current_score
        best_A = current_A[:]
    
    n_steps = 0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = min(1.0, elapsed / max(1.0, total_time))
        temp = max(end_temp, start_temp * (0.997 ** n_steps))
        
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        r = random.random()
        if r < 0.25:
            delta = random.randint(-5, 5)
            new_val = old_val + delta
        elif r < 0.50:
            delta = random.randint(-30, 30)
            new_val = old_val + delta
        elif r < 0.70:
            candidates = []
            for other in current_A:
                if other == old_val:
                    continue
                for k in [2, 3, 4, 5]:
                    v = other * k
                    if 1 <= v <= 10000 and v != old_val:
                        candidates.append(v)
                for k in [2, 3, 4, 5]:
                    if other % k == 0:
                        v = other // k
                        if 1 <= v <= 10000 and v != old_val:
                            candidates.append(v)
            if candidates:
                new_val = random.choice(candidates)
            else:
                delta = random.randint(-100, 100)
                new_val = old_val + delta
        elif r < 0.85:
            delta = random.randint(-200, 200)
            new_val = old_val + delta
        else:
            new_val = random.randint(1, 5000)
        
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current_A and new_val != old_val:
            continue
        
        current_A[i] = new_val
        new_score = compute_metrics(current_A)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            delta_score = new_score - current_score
            if temp > 0.01 and delta_score < 50 and random.random() < min(1.0, (-delta_score) / temp):
                current_score = new_score
                no_improve += 1
            else:
                current_A[i] = old_val
                no_improve += 1
        
        n_steps += 1
        
        if no_improve > max_no_improve:
            current_A = best_A[:]
            current_score = best_score
            no_improve = 0
            n_steps = 0
    
    return best_A, best_score

def final_polish(A, deadline):
    """Final high-precision polish with multiple passes and structure-aware moves."""
    current_A = A[:]
    current_score = compute_metrics(current_A)
    
    windows = [50, 30, 20, 10, 5]
    
    for window in windows:
        if time.time() > deadline:
            break
        for i in range(20):
            if time.time() > deadline:
                break
            old_val = current_A[i]
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                
                current_A[i] = new_val
                new_score = compute_metrics(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    old_val = new_val
                else:
                    current_A[i] = old_val
    
    for i in range(20):
        if time.time() > deadline:
            break
        old_val = current_A[i]
        for j in range(20):
            if i == j:
                continue
            other = current_A[j]
            for k in [2, 3, 4, 5, 6]:
                new_val = other * k
                if 1 <= new_val <= 10000 and new_val != old_val and new_val not in current_A:
                    current_A[i] = new_val
                    new_score = compute_metrics(current_A)
                    if new_score < current_score:
                        current_score = new_score
                        old_val = new_val
                    else:
                        current_A[i] = old_val
            for k in [2, 3, 4, 5, 6]:
                if other % k == 0:
                    new_val = other // k
                    if 1 <= new_val <= 10000 and new_val != old_val and new_val not in current_A:
                        current_A[i] = new_val
                        new_score = compute_metrics(current_A)
                        if new_score < current_score:
                            current_score = new_score
                            old_val = new_val
                        else:
                            current_A[i] = old_val
    
    return current_A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate structured candidates (includes greedy collision construction)
    candidates = generate_structured_candidates(deadline=deadline)
    
    best_A = None
    best_score = float('inf')
    scored_candidates = []
    
    for cand in candidates:
        if len(cand) != 20:
            continue
        score = compute_metrics(cand)
        scored_candidates.append((score, cand))
        if score < best_score:
            best_score = score
            best_A = cand[:]
    
    if best_A is None:
        return list(range(1, 21))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    start_temp = 50.0
    end_temp = 0.01
    
    # Run SA on the best candidate
    sa_deadline = deadline - 12.0
    best_A, best_score = simulated_annealing(
        best_A, start_temp, end_temp, start_time, sa_deadline, best_A, best_score
    )
    
    # Run SA on second-best if time permits
    if time.time() < deadline - 10.0 and len(scored_candidates) >= 2:
        second_score, second_best = scored_candidates[1]
        if second_best != best_A:
            best_A, best_score = simulated_annealing(
                second_best, start_temp * 0.8, end_temp, start_time, deadline - 6.0, best_A, best_score
            )
    
    # Run SA on third-best if time permits
    if time.time() < deadline - 8.0 and len(scored_candidates) >= 3:
        third_score, third_best = scored_candidates[2]
        if third_best != best_A:
            best_A, best_score = simulated_annealing(
                third_best, start_temp * 0.6, end_temp, start_time, deadline - 4.0, best_A, best_score
            )
    
    # Local refinement pass
    if time.time() < deadline - 2.0:
        current_A = best_A[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 300
        
        while time.time() < deadline - 1.5 and no_improve < max_no_improve:
            i = random.randint(0, 19)
            old_val = current_A[i]
            
            tried = set()
            found_improvement = False
            
            for _ in range(10):
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val in tried:
                    continue
                tried.add(new_val)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                
                current_A[i] = new_val
                new_score = compute_metrics(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    found_improvement = True
                    break
                else:
                    current_A[i] = old_val
            
            if not found_improvement:
                for j in range(20):
                    if i == j:
                        continue
                    other = current_A[j]
                    for k in [2, 3, 4, 5]:
                        new_val = other * k
                        if new_val in tried:
                            continue
                        tried.add(new_val)
                        
                        if 1 <= new_val <= 10000 and new_val != old_val and new_val not in current_A:
                            current_A[i] = new_val
                            new_score = compute_metrics(current_A)
                            
                            if new_score < current_score:
                                current_score = new_score
                                no_improve = 0
                                if current_score < best_score:
                                    best_score = current_score
                                    best_A = current_A[:]
                                found_improvement = True
                                break
                            else:
                                current_A[i] = old_val
                    if found_improvement:
                        break
            
            if not found_improvement:
                no_improve += 1
    
    if time.time() < deadline - 0.5:
        best_A, best_score = final_polish(best_A, deadline - 0.1)
    
    return best_A

if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Score: {compute_metrics(result)}")
