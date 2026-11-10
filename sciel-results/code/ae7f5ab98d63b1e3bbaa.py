"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Generate all odd divisors of base greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order
    n_divs = len(divisors)

    # Precompute indices for each modulus to speed up slicing?
    # Actually, numpy slicing `arr[r::m]` is fast.

    def get_score(cover_counts):
        """Compute number of uncovered residues."""
        return int(np.sum(cover_counts == 0))

    def greedy_construct(order=None, cost_weighted=False):
        """Greedy: moduli in specified order."""
        if order is None:
            order = divisors
        cover_counts = np.zeros(base, dtype=np.int8)
        classes = []
        
        for m in order:
            best_r = 0
            best_val = -1
            # Optimization: Only check residues if m is small enough?
            # m can be up to 51975. Looping m times is too slow if m is large.
            # However, we only have ~100 divisors.
            # If m is large, the number of residues is large.
            # We can sample or use a heuristic.
            # But for correctness and speed, let's see how many divisors are large.
            # Divisors of 51975.
            # 51975 = 3^3 * 5^2 * 7 * 11.
            # Number of divisors = (3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48.
            # Excluding 1, we have 47 divisors.
            # Max divisor is 51975.
            # Looping 51975 times * 47 divisors = ~2.4M operations.
            # Each operation is a numpy sum over ~1 element (if m is large) or more.
            # This should be fast enough if optimized.
            
            # To speed up: if m > base/2, there is only 1 residue (r=0 covers 0, m, 2m... only 0 is < base? No, 0, m... if m > base/2, only 0 is in [0, base). Wait.
            # Residues are 0 <= r < m.
            # The class covers {r, r+m, r+2m, ...} < base.
            # If m > base/2, the class covers at most 2 elements?
            # If r < base - m, it covers r and r+m.
            # If r >= base - m, it covers only r.
            
            # Let's just loop. 47 * 51975 is not huge.
            for r in range(m):
                # Count newly covered spots
                # cover_counts[r::m] == 0
                new_cov = int(np.sum(cover_counts[r::m] == 0))
                if cost_weighted:
                    val = new_cov * m
                else:
                    val = new_cov
                
                if val > best_val:
                    best_val = val
                    best_r = r
            
            classes.append([m, best_r])
            cover_counts[best_r::m] += 1
        
        return classes, get_score(cover_counts)

    def local_search(classes, max_iter=50, time_budget=None):
        """Local search using cover_counts for speed."""
        classes = [list(c) for c in classes]
        n = len(classes)
        
        # Initialize cover_counts
        cover_counts = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            cover_counts[r::m] += 1
        
        current_score = get_score(cover_counts)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            if time_budget is not None and time.time() > time_budget:
                break
            
            improved = False
            # Randomize order of classes to avoid bias
            idxs = list(range(n))
            random.shuffle(idxs)
            
            for i in idxs:
                m, r = classes[i]
                
                # Remove current class
                cover_counts[r::m] -= 1
                
                # Find best new residue
                best_r = r
                best_score = current_score
                
                # Heuristic: If m is very large, checking all r is expensive.
                # But m is a divisor.
                # We check all r.
                for r_new in range(m):
                    if r_new == r:
                        continue
                    # Tentatively add
                    cover_counts[r_new::m] += 1
                    new_score = get_score(cover_counts)
                    cover_counts[r_new::m] -= 1
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = r_new
                
                # Apply best
                if best_r != r:
                    cover_counts[r::m] += 1 # Re-add old? No, we removed it.
                    # We need to add the new one
                    cover_counts[best_r::m] += 1
                    classes[i][1] = best_r
                    current_score = best_score
                    improved = True
                else:
                    # Re-add old
                    cover_counts[r::m] += 1
            
            if not improved:
                break
        
        return classes, current_score

    def simulated_annealing(classes, start_temp=10.0, end_temp=0.1, max_iter=500):
        """Simulated annealing using cover_counts."""
        classes = [list(c) for c in classes]
        n = len(classes)
        
        cover_counts = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            cover_counts[r::m] += 1
        
        current_score = get_score(cover_counts)
        best_classes = [list(c) for c in classes]
        best_score = current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            progress = iteration / max_iter
            temp = start_temp * (end_temp / start_temp) ** progress
            
            i = random.randint(0, n - 1)
            m, r = classes[i]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            if new_r == old_r:
                continue
                
            # Remove old
            cover_counts[old_r::m] -= 1
            # Add new
            cover_counts[new_r::m] += 1
            
            new_score = get_score(cover_counts)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
                current_score = new_score
                classes[i][1] = new_r
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [list(c) for c in classes]
            else:
                # Revert
                cover_counts[new_r::m] -= 1
                cover_counts[old_r::m] += 1
        
        return best_classes, best_score

    # Initial greedy
    classes, current_score = greedy_construct()
    best_classes = [list(c) for c in classes]
    best_score = current_score

    # Local search on initial
    classes, current_score = local_search(classes, max_iter=50)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Try cost-weighted greedy
    classes, current_score = greedy_construct(cost_weighted=True)
    classes, current_score = local_search(classes, max_iter=50)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        random_divs = list(divisors)
        random.shuffle(random_divs)
        
        use_weighted = random.random() < 0.5
        classes, current_score = greedy_construct(order=random_divs, cost_weighted=use_weighted)
        
        # Local search
        classes, current_score = local_search(classes, max_iter=30)
        
        # Occasionally try simulated annealing
        if random.random() < 0.3 and time.time() < deadline - 15.0:
            classes, current_score = simulated_annealing(classes, max_iter=200)
        
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    # Final polish
    if time.time() < deadline - 5.0:
        classes, current_score = local_search(best_classes, max_iter=100)
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    return best_classes
# EVOLVE-BLOCK-END
