import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = list(range(128, 257))  # 129 targets
    target_set = set(TARGETS)
    
    def compute_mult(A_set):
        """Return max multiplicity over targets in [128,256]."""
        max_mult = 0
        for t in TARGETS:
            count = 0
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    count += 1
            if count == 0:
                return float('inf')  # not covered
            if count > max_mult:
                max_mult = count
        return max_mult

    def is_complete(A_set):
        """Check if all targets are covered."""
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True

    def get_missing_targets(A_set):
        missing = []
        for t in TARGETS:
            found = False
            for a in A_set:
                b = t - a
                if b < a:
                    continue
                if b in A_set:
                    found = True
                    break
            if not found:
                missing.append(t)
        return missing

    def local_search(A_set, max_iters=2000):
        """Local search to minimize max multiplicity."""
        current_mult = compute_mult(A_set)
        if current_mult == float('inf'):
            return A_set, current_mult
            
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try removing an element
            for a in list(A_set):
                new_set = A_set - {a}
                if not new_set: continue
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
                    break
                elif new_mult == current_mult and len(new_set) < len(A_set):
                    A_set = new_set
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping an element
            if len(A_set) > 0:
                a_in = random.choice(list(A_set))
                a_out = random.randint(0, N)
                if a_out in A_set:
                    continue
                new_set = A_set - {a_in} | {a_out}
                if not is_complete(new_set):
                    continue
                new_mult = compute_mult(new_set)
                if new_mult < current_mult:
                    A_set = new_set
                    current_mult = new_mult
                    improved = True
            
            if not improved:
                # Stuck. Maybe try adding an element that reduces peak?
                # Or just break out of local search for this seed
                break
        
        return A_set, current_mult

    best_A = None
    best_mult = float('inf')
    
    # Generate diverse structured seeds
    seeds = []
    
    # Seed 1: Random small sets
    for _ in range(5):
        k = random.randint(16, 24)
        A = random.sample(range(0, N+1), k)
        seeds.append(set(A))
    
    # Seed 2: Structured - even numbers + a few odds
    evens = set(range(0, N+1, 2))
    odds = set(random.sample(range(1, N, 2), 10))
    seeds.append(evens | odds)
    
    # Seed 3: Structured - low range + high range
    low = set(range(0, 33))
    high = set(range(96, N+1))
    seeds.append(low | high)
    
    # Seed 4: Structured - arithmetic progression with step 3
    ap3 = set(range(0, N+1, 3))
    # Add some to cover gaps
    for i in range(0, N+1, 2):
        if i not in ap3 and random.random() < 0.3:
            ap3.add(i)
    seeds.append(ap3)
    
    # Seed 5: Structured - clustered
    cluster1 = set(range(0, 20))
    cluster2 = set(range(60, 80))
    cluster3 = set(range(110, N+1))
    seeds.append(cluster1 | cluster2 | cluster3)
    
    # Seed 6: Random with bias toward middle
    biased = set()
    for _ in range(20):
        # Bias toward middle values
        x = int(64 + 64 * random.gauss(0, 0.3))
        x = max(0, min(N, x))
        biased.add(x)
    seeds.append(biased)
    
    # Process each seed
    for seed_set in seeds:
        if time.time() > deadline - 5:
            break
        
        # First, ensure completeness by greedily adding elements
        A_set = set(seed_set)
        
        # Greedy completion
        for _ in range(50):  # Max 50 additions
            if time.time() > deadline - 5:
                break
            
            if is_complete(A_set):
                break
            
            missing = get_missing_targets(A_set)
            if not missing:
                break
            
            # Find best element to add
            best_gain = -1
            best_elem = None
            for x in range(0, N+1):
                if x in A_set:
                    continue
                gain = 0
                for t in missing:
                    y = t - x
                    if 0 <= y <= N and y in A_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            
            if best_gain > 0 and best_elem is not None:
                A_set.add(best_elem)
            else:
                # Force add
                t0 = missing[0]
                x1 = t0 // 2
                x2 = t0 - x1
                A_set.add(x1)
                if x2 != x1:
                    A_set.add(x2)
        
        # Verify completeness
        if not is_complete(A_set):
            # Force complete
            for t in TARGETS:
                found = False
                for a in A_set:
                    b = t - a
                    if b < a:
                        continue
                    if b in A_set:
                        found = True
                        break
                if not found:
                    a = t // 2
                    b = t - a
                    A_set.add(a)
                    A_set.add(b)
        
        # Local search
        A_set, mult = local_search(A_set, max_iters=1000)
        
        if mult < best_mult:
            best_mult = mult
            best_A = list(A_set)
    
    if best_A is None:
        return list(range(64, 129))
    
    return best_A
