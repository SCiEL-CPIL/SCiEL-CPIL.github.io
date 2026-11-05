"""Combined program: greedy construction with randomized local search polish."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975
    # All divisors > 1 of base (odd divisors since base is odd)
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size ascending for greedy
    divisors_sorted = sorted(divisors)
    
    def score_of(classes):
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(base - covered.sum())
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick best residue for each."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() > deadline - 20:
                break
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(covered[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, covered
    
    def local_search(classes, covered, max_iterations=500):
        """Iteratively try to improve by changing residues or adding/removing classes."""
        if not classes:
            return classes, covered
        sc = int(base - covered.sum())
        used_moduli = set(m for m, _ in classes)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Try changing residue of existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    return classes, covered
                m, r = classes[i]
                best_r = r
                best_sc = sc
                # Try all residues if small, else sample
                if m <= 50:
                    candidates = list(range(m))
                else:
                    candidates = [r] + [random.randint(0, m - 1) for _ in range(30)]
                for tr in candidates:
                    if tr == r:
                        continue
                    # Compute gain/loss
                    old_covered = covered[r::m]
                    new_covered = covered[tr::m]
                    # Gain = newly covered, Loss = no longer covered
                    # We need to compute this carefully
                    # Temporarily change
                    classes[i][1] = tr
                    new_covered_arr = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        new_covered_arr[rj::mj] = 1
                    new_sc = int(base - new_covered_arr.sum())
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_r = tr
                        covered = new_covered_arr
                    else:
                        classes[i][1] = r
                if best_r != r:
                    classes[i][1] = best_r
                    sc = best_sc
                    improved = True
            
            # Try adding an unused modulus
            if time.time() < deadline - 15:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    # Pick a few candidates
                    random.shuffle(unused)
                    for m in unused[:5]:
                        if time.time() > deadline - 10:
                            break
                        best_r = 0
                        best_gain = -1
                        for _ in range(min(20, m)):
                            r = random.randint(0, m - 1)
                            gain = int(np.sum(covered[r::m] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_r = r
                        # Also check r=0 and r=1 explicitly
                        for r in [0, 1]:
                            if r < m:
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        if best_gain > 0:
                            classes.append([m, best_r])
                            covered[best_r::m] = 1
                            used_moduli.add(m)
                            sc = int(base - covered.sum())
                            improved = True
        
        return classes, covered

    best_classes = []
    best_score = base + 1
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_greedy = 15
    for attempt in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            order = divisors_sorted
        elif attempt == 1:
            order = sorted(divisors, reverse=True)
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)
        
        classes, covered = greedy_construct(order)
        sc = int(base - covered.sum())
        if sc < best_score:
            best_score = sc
            best_classes = classes[:]
        
        # Phase 2: Local search on this solution
        if time.time() < deadline - 20:
            classes_ls, covered_ls = local_search(classes, covered, max_iterations=100)
            sc_ls = int(base - covered_ls.sum())
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]
    
    # Phase 3: More aggressive local search on the best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered_best[r::m] = 1
        best_classes, covered_best = local_search(best_classes, covered_best, max_iterations=300)
        final_sc = int(base - covered_best.sum())
        if final_sc < best_score:
            best_score = final_sc
    
    if not best_classes:
        best_classes = [[3, 0]]
    
    return best_classes
