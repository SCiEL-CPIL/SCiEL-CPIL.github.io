import time
import random
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    base = 51975

    # Efficient divisor enumeration
    def get_divisors(n):
        divs = []
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                other = n // i
                if other != i:
                    divs.append(other)
        return sorted(divs)

    divisors = get_divisors(base)
    divisors_sorted = divisors  # already sorted

    def compute_coverage(classes):
        """Compute coverage array from list of classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return covered

    def score_from_covered(covered):
        return int(base - covered.sum())

    def greedy_construct(order, covered, time_budget):
        """
        Greedy construction with adaptive look-ahead.
        At each step, select the class with highest marginal gain,
        using probabilistic selection based on gain relative to remaining budget.
        """
        classes = []
        used = set()
        remaining = list(order)
        
        while remaining and time.time() < time_budget:
            # For each candidate modulus, find best residue
            best_m = None
            best_r = None
            best_gain = -1
            second_best_m = None
            second_best_r = None
            second_best_gain = -1
            
            # Sort candidates by potential (larger moduli first)
            candidates = sorted(remaining, key=lambda m: -m)
            
            # Limit candidates to top 40 for speed
            if len(candidates) > 40:
                candidates = candidates[:40]
                
            for m in candidates:
                if m in used:
                    continue
                
                # Find best residue for this modulus
                if m <= 30:
                    # Try all residues
                    for r in range(m):
                        gain = int(np.sum(covered[r::m] == 0))
                        if gain > best_gain:
                            if best_gain > second_best_gain:
                                second_best_m, second_best_r, second_best_gain = best_m, best_r, best_gain
                            best_m, best_r, best_gain = m, r, gain
                        elif gain > second_best_gain:
                            second_best_m, second_best_r, second_best_gain = m, r, gain
                else:
                    # Sample residues + deterministic checks
                    residues_to_try = list(range(0, m, max(1, m // 25)))
                    residues_to_try = list(set(residues_to_try + [random.randint(0, m - 1) for _ in range(15)]))
                    
                    for r in residues_to_try:
                        gain = int(np.sum(covered[r::m] == 0))
                        if gain > best_gain:
                            if best_gain > second_best_gain:
                                second_best_m, second_best_r, second_best_gain = best_m, best_r, best_gain
                            best_m, best_r, best_gain = m, r, gain
                        elif gain > second_best_gain:
                            second_best_m, second_best_r, second_best_gain = m, r, gain
            
            if best_m is None:
                break
            
            # Probabilistic selection: prefer best, but occasionally take second best
            # This adds diversity to the search
            if second_best_m is not None and second_best_gain > 0:
                ratio = second_best_gain / (best_gain + 1e-10)
                # With probability proportional to how close second is, take it
                if random.random() < 0.15 * ratio:
                    chosen_m, chosen_r = second_best_m, second_best_r
                else:
                    chosen_m, chosen_r = best_m, best_r
            else:
                chosen_m, chosen_r = best_m, best_r
            
            classes.append([chosen_m, chosen_r])
            used.add(chosen_m)
            covered[chosen_r::chosen_m] = 1
            remaining = [m for m in remaining if m != chosen_m]
        
        return classes, covered

    def local_search(classes, covered, time_budget):
        """
        Local search with multiple move types:
        1. Change residue of existing class
        2. Add unused modulus
        3. Remove weak class
        4. Swap two classes
        """
        if not classes:
            return classes, covered

        sc = score_from_covered(covered)
        used_moduli = set(m for m, _ in classes)
        no_improve_count = 0
        max_no_improve = 50
        
        while no_improve_count < max_no_improve and time.time() < time_budget:
            improved = False
            
            # --- Move 1: Change residue of existing classes ---
            for i in range(len(classes)):
                if time.time() > time_budget:
                    return classes, covered
                m, r = classes[i]
                
                # Generate candidate residues
                if m <= 50:
                    candidates = [tr for tr in range(m) if tr != r]
                else:
                    # Sample + strategic residues
                    candidates = set()
                    candidates.add(0)
                    if r != 0:
                        candidates.add(r)
                    # Add residues that cover uncovered positions
                    uncovered = np.where(covered == 0)[0]
                    if len(uncovered) > 0:
                        # Sample some uncovered positions to find good residues
                        sample_idx = random.sample(range(len(uncovered)), min(20, len(uncovered)))
                        for idx in sample_idx:
                            pos = uncovered[idx]
                            candidates.add(pos % m)
                    # Add random
                    candidates.update(random.randint(0, m - 1) for _ in range(20))
                    candidates = list(candidates)
                
                best_r = r
                best_sc = sc
                
                for tr in candidates:
                    if tr == r:
                        continue
                    
                    # Quick check: how many uncovered positions does this residue cover?
                    gain = int(np.sum(covered[tr::m] == 0))
                    if gain <= 0:
                        continue
                    
                    # Try the move
                    old_r = classes[i][1]
                    classes[i][1] = tr
                    
                    # Recompute coverage
                    new_covered = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        new_covered[rj::mj] = 1
                    new_sc = score_from_covered(new_covered)
                    
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_r = tr
                        covered = new_covered
                    else:
                        classes[i][1] = old_r
                
                if best_r != r:
                    sc = best_sc
                    improved = True
            
            # --- Move 2: Add an unused modulus ---
            if time.time() < time_budget - 5:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    # Sort by potential: larger moduli cover more
                    unused_sorted = sorted(unused, reverse=True)
                    # Try top 15
                    for m in unused_sorted[:15]:
                        if time.time() > time_budget:
                            break
                        
                        best_r = 0
                        best_gain = -1
                        
                        if m <= 40:
                            for r in range(m):
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        else:
                            # Sample residues
                            for _ in range(20):
                                r = random.randint(0, m - 1)
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                            gain0 = int(np.sum(covered[0::m] == 0))
                            if gain0 > best_gain:
                                best_gain = gain0
                                best_r = 0
                        
                        if best_gain > 0:
                            # Check if adding this improves the score
                            old_sc = sc
                            classes.append([m, best_r])
                            covered[best_r::m] = 1
                            new_sc = score_from_covered(covered)
                            
                            if new_sc < old_sc:
                                sc = new_sc
                                used_moduli.add(m)
                                improved = True
                                break  # Only add one at a time
                            else:
                                classes.pop()
            
            # --- Move 3: Remove a class that contributes little ---
            if len(classes) > 6 and time.time() < time_budget - 5:
                for i in range(len(classes)):
                    if time.time() > time_budget:
                        break
                    m, r = classes[i]
                    
                    # Check if this class uniquely covers anything
                    # Count coverage from other classes
                    count_cover = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        if j != i:
                            count_cover[rj::mj] += 1
                    
                    # Positions in r::m that are only covered by this class
                    only_this = (count_cover[r::m] == 0)
                    only_count = int(np.sum(only_this))
                    
                    if only_count == 0:
                        # Try removing
                        old_sc = sc
                        classes.pop(i)
                        used_moduli.discard(m)
                        new_covered = np.zeros(base, dtype=np.uint8)
                        for mj, rj in classes:
                            new_covered[rj::mj] = 1
                        new_sc = score_from_covered(new_covered)
                        
                        if new_sc <= old_sc:
                            covered = new_covered
                            sc = new_sc
                            improved = True
                            break
                        else:
                            classes.insert(i, [m, r])
                            used_moduli.add(m)
            
            # --- Move 4: Swap two classes (change residues) ---
            if len(classes) >= 2 and time.time() < time_budget - 5 and random.random() < 0.3:
                i, j = random.sample(range(len(classes)), 2)
                mi, ri = classes[i]
                mj, rj = classes[j]
                
                # Try swapping residues
                classes[i][1], classes[j][1] = rj, ri
                new_covered = np.zeros(base, dtype=np.uint8)
                for k, (mk, rk) in enumerate(classes):
                    new_covered[rk::mk] = 1
                new_sc = score_from_covered(new_covered)
                
                if new_sc < sc:
                    sc = new_sc
                    covered = new_covered
                    improved = True
                else:
                    classes[i][1], classes[j][1] = ri, rj
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        return classes, covered

    best_classes = []
    best_score = base + 1

    # Phase 1: Multiple greedy constructions with different orderings
    num_restarts = 30
    for attempt in range(num_restarts):
        if time.time() > deadline - 40:
            break

        if attempt == 0:
            # Deterministic: sorted by modulus ascending
            order = divisors_sorted[:]
        elif attempt == 1:
            # Deterministic: sorted by modulus descending
            order = sorted(divisors, reverse=True)
        elif attempt == 2:
            # Mix: interleave small and large
            small = divisors_sorted[:len(divisors_sorted)//2]
            large = sorted(divisors_sorted[len(divisors_sorted)//2:], reverse=True)
            order = []
            for i in range(max(len(small), len(large))):
                if i < len(small):
                    order.append(small[i])
                if i < len(large):
                    order.append(large[i])
        elif attempt == 3:
            # Priority by divisibility: prefer moduli that divide others
            order = sorted(divisors, key=lambda m: sum(1 for d in divisors if d % m == 0), reverse=True)
        elif attempt == 4:
            # Priority by coprimality: prefer moduli coprime to many others
            order = sorted(divisors, key=lambda m: sum(1 for d in divisors if gcd(m, d) == 1), reverse=True)
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)

        covered = np.zeros(base, dtype=np.uint8)
        classes, covered = greedy_construct(order, covered, deadline - 30)
        sc = score_from_covered(covered)

        if sc < best_score:
            best_score = sc
            best_classes = classes[:]

        # Local search refinement
        if time.time() < deadline - 20 and classes:
            classes_ls, covered_ls = local_search(classes, covered, time_budget=deadline - 15)
            sc_ls = score_from_covered(covered_ls)
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]

    # Phase 2: Aggressive local search on best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = compute_coverage(best_classes)
        best_classes, covered_best = local_search(best_classes, covered_best, time_budget=deadline - 5)
        final_sc = score_from_covered(covered_best)
        if final_sc < best_score:
            best_score = final_sc

    # Phase 3: Final random restarts with different seeds
    if time.time() < deadline - 20:
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 31337 + 999 + 10000)
            order = divisors_sorted[:]
            random.shuffle(order)

            covered = np.zeros(base, dtype=np.uint8)
            classes, covered = greedy_construct(order, covered, deadline - 15)
            sc = score_from_covered(covered)

            if sc < best_score:
                best_score = sc
                best_classes = classes[:]

            if time.time() < deadline - 10 and classes:
                classes_ls, covered_ls = local_search(classes, covered, time_budget=deadline - 5)
                sc_ls = score_from_covered(covered_ls)
                if sc_ls < best_score:
                    best_score = sc_ls
                    best_classes = classes_ls[:]

    if not best_classes:
        best_classes = [[3, 0]]

    return best_classes
