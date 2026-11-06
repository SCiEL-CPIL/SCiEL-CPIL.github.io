import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n_max = 128
    target_min = 128
    target_max = 256
    target_range = list(range(target_min, target_max + 1))
    
    # Helper to compute multiplicities for a given set A
    def compute_multiplicities(A):
        # A is a set of ints
        # We need counts for sums in [128, 256]
        counts = np.zeros(257, dtype=np.int32)
        A_list = sorted(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if s <= 256:
                    counts[s] += 1
        return counts

    # Helper to evaluate cost
    def evaluate(A):
        counts = compute_multiplicities(A)
        uncovered = 0
        peak = 0
        for t in target_range:
            c = counts[t]
            if c == 0:
                uncovered += 1
            else:
                if c > peak:
                    peak = c
        
        # Cost function: 
        # Priority 1: Coverage (uncovered count)
        # Priority 2: Peak multiplicity
        # Priority 3: Set size (to keep A small, though not strictly required, helps generalization)
        cost = uncovered * 100000 + peak * 1000 + len(A)
        return cost, peak, uncovered

    # Initial Construction: Greedy Coverage
    # Start with a random set of moderate size to ensure we start close to valid
    # A set of size ~30-40 usually covers [128, 256] reasonably well if spread out.
    A = set()
    rng = random.Random(42)
    
    # Strategy: Add elements that cover the most "hard" sums.
    # Initially, all sums are hard.
    # Let's just start with a random subset and repair.
    # A good heuristic for additive bases is to take a dense interval.
    # But to minimize peak, we want to avoid clustering.
    # Let's start with a random set of 40 elements.
    for _ in range(40):
        A.add(rng.randint(0, n_max))
    
    current_cost, current_peak, current_uncovered = evaluate(A)
    
    # Repair Phase: Ensure coverage
    # While uncovered > 0, add elements that cover the most uncovered sums.
    max_repair_steps = 50
    for _ in range(max_repair_steps):
        if time.time() > deadline - 10: break
        if current_uncovered == 0:
            break
            
        # Find uncovered sums
        counts = compute_multiplicities(A)
        uncovered_sums = [t for t in target_range if counts[t] == 0]
        
        if not uncovered_sums:
            break
            
        # Find best candidate to add
        best_candidate = None
        best_gain = -1
        best_new_cost = current_cost
        
        # Sample candidates to speed up
        candidates = [x for x in range(n_max + 1) if x not in A]
        rng.shuffle(candidates)
        
        for cand in candidates[:50]: # Check top 50 random candidates
            new_A = A | {cand}
            # Quick check: does it help?
            # Instead of full eval, we can just do full eval since N is small
            c, p, u = evaluate(new_A)
            if u < current_uncovered:
                # Prefer lower uncovered, then lower peak
                if best_gain == -1 or u < best_gain or (u == best_gain and c < best_new_cost):
                    best_gain = u
                    best_candidate = cand
                    best_new_cost = c
            elif u == current_uncovered:
                if c < best_new_cost:
                    best_candidate = cand
                    best_new_cost = c
                    best_gain = u # Keep track that it's not worse coverage-wise
        
        if best_candidate is not None:
            A.add(best_candidate)
            current_cost, current_peak, current_uncovered = evaluate(A)
        else:
            # If no random candidate helped, pick the one that minimizes cost globally among all candidates?
            # This is expensive. Let's just break or pick random.
            # Actually, if we have a valid set, we don't need to repair.
            # If we can't repair with random, we might be stuck. 
            # But with N=128, a set of size 40+ should cover [128, 256].
            # Let's force add a random one if stuck.
            A.add(rng.choice([x for x in range(n_max + 1) if x not in A]))
            current_cost, current_peak, current_uncovered = evaluate(A)

    # Optimization Phase: Simulated Annealing
    # Goal: Minimize peak multiplicity while maintaining coverage (uncovered == 0)
    
    current_A = set(A)
    current_cost, current_peak, current_uncovered = evaluate(current_A)
    best_A = set(current_A)
    best_cost = current_cost
    best_peak = current_peak
    
    T = 100.0
    T_min = 1.0
    alpha = 0.999
    
    while time.time() < deadline:
        if current_uncovered > 0:
            # If we broke coverage, we must repair immediately
            # Re-run repair logic quickly
            counts = compute_multiplicities(current_A)
            uncovered_sums = [t for t in target_range if counts[t] == 0]
            if uncovered_sums:
                # Try to add back a removed element or a new one
                # Simple heuristic: Add the element that covers the most uncovered sums
                best_add = None
                max_covered = -1
                for x in range(n_max + 1):
                    if x in current_A: continue
                    # How many uncovered sums does x help?
                    # Sum x+y where y in A.
                    # We need x+y to be in uncovered_sums.
                    # So y must be in uncovered_sums - x.
                    help_count = 0
                    for u in uncovered_sums:
                        y = u - x
                        if 0 <= y <= n_max and y in current_A:
                            help_count += 1
                    if help_count > max_covered:
                        max_covered = help_count
                        best_add = x
                
                if best_add is not None:
                    current_A.add(best_add)
                    current_cost, current_peak, current_uncovered = evaluate(current_A)
                continue
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.4:
            # Remove
            if len(current_A) > 10: # Keep it non-trivial
                elem_to_remove = rng.choice(list(current_A))
                new_A = set(current_A)
                new_A.remove(elem_to_remove)
            else:
                new_A = set(current_A)
        elif move_type < 0.8:
            # Add
            candidates = [x for x in range(n_max + 1) if x not in current_A]
            if candidates:
                elem_to_add = rng.choice(candidates)
                new_A = set(current_A)
                new_A.add(elem_to_add)
            else:
                new_A = set(current_A)
        else:
            # Swap
            if len(current_A) > 5:
                elem_in = rng.choice(list(current_A))
                candidates = [x for x in range(n_max + 1) if x not in current_A]
                if candidates:
                    elem_out = rng.choice(candidates)
                    new_A = set(current_A)
                    new_A.remove(elem_in)
                    new_A.add(elem_out)
                else:
                    new_A = set(current_A)
            else:
                new_A = set(current_A)
                
        new_cost, new_peak, new_uncovered = evaluate(new_A)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost <= 0:
            # Accept
            current_A = new_A
            current_cost = new_cost
            current_peak = new_peak
            current_uncovered = new_uncovered
            
            if current_uncovered == 0 and current_cost < best_cost:
                best_cost = current_cost
                best_A = set(current_A)
                best_peak = current_peak
        else:
            # Probabilistic accept
            # Avoid accepting if it breaks coverage (huge cost penalty handles this, 
            # but explicitly rejecting coverage breaks helps stability)
            if new_uncovered > 0 and current_uncovered == 0:
                # Don't accept coverage breaks easily
                if delta_cost > 1000: 
                    pass
                else:
                    p = np.exp(-delta_cost / T)
                    if rng.random() < p:
                        current_A = new_A
                        current_cost = new_cost
                        current_peak = new_peak
                        current_uncovered = new_uncovered
            else:
                p = np.exp(-delta_cost / T)
                if rng.random() < p:
                    current_A = new_A
                    current_cost = new_cost
                    current_peak = new_peak
                    current_uncovered = new_uncovered
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final check: Ensure the best_A found is valid (covers all)
    # If best_A has uncovered > 0, we need to fall back to the last valid state or repair best_A.
    # The tracking of best_A only updated when current_uncovered == 0.
    # So best_A should be valid.
    
    # One final sanity check on best_A
    c, p, u = evaluate(best_A)
    if u > 0:
        # Fallback to a known valid set if something went wrong
        # Use the simple interval set which is always valid
        return list(range(64, 129))
        
    return list(best_A)
