import time
import numpy as np
import random
from collections import deque

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    n_divs = len(divisors)

    # Precompute for each divisor: the set of residue classes it covers
    # We'll use a bitset approach for speed
    # For each divisor m, precompute the indices that each residue r covers
    # But that's too large. Instead, use numpy slicing.

    def uncovered_count(covered):
        return int(np.sum(covered == 0))

    def score_solution(classes):
        if not classes:
            return base
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return uncovered_count(covered)

    def greedy_pass(div_order, base, deadline, covered_init=None, rng=None):
        """One greedy pass: pick best (m,r) at each step from div_order."""
        covered = np.zeros(base, dtype=np.uint8) if covered_init is None else covered_init.copy()
        classes = []
        available = set(div_order)
        
        while available:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in div_order:
                if m not in available:
                    continue
                
                # Max possible coverage for this modulus
                max_possible = (base + m - 1) // m
                if max_possible <= best_gain:
                    continue
                
                # Find best residue for this modulus
                best_r_for_m = 0
                best_count_for_m = -1
                for r in range(m):
                    cnt = int(np.sum(covered[r::m] == 0))
                    if cnt > best_count_for_m:
                        best_count_for_m = cnt
                        best_r_for_m = r
                
                if best_count_for_m > best_gain:
                    best_gain = best_count_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            covered[best_r::best_m] = 1
            available.remove(best_m)
        
        return classes, covered

    def local_search(classes, base, deadline):
        """Local search: try changing residues and swapping classes."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered[r::m] = 1
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try changing residue for each class
            for idx in range(len(best_classes)):
                if time.time() > deadline - 10.0:
                    break
                m, r = best_classes[idx]
                # Try a few alternative residues
                for r2 in [0, m//2, m-1, (r+1)%m, (r-1)%m]:
                    if r2 == r:
                        continue
                    new_classes = [c[:] for c in best_classes]
                    new_classes[idx] = [m, r2]
                    new_score = score_solution(new_classes)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes = new_classes
                        covered = np.zeros(base, dtype=np.uint8)
                        for m2, r3 in best_classes:
                            covered[r3::m2] = 1
                        improved = True
                        break
                if improved:
                    break
            
            # Try swapping two classes
            if not improved and len(best_classes) >= 2:
                for i in range(len(best_classes)):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i+1, len(best_classes)):
                        new_classes = [c[:] for c in best_classes]
                        new_classes[i], new_classes[j] = new_classes[j], new_classes[i]
                        new_score = score_solution(new_classes)
                        if new_score < best_score:
                            best_score = new_score
                            best_classes = new_classes
                            improved = True
                            break
                    if improved:
                        break
        
        return best_classes, best_score

    def hill_climb(classes, base, deadline):
        """More aggressive local search with random restarts of residue changes."""
        if not classes:
            return classes, base
        
        rng = random.Random(12345)
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered[r::m] = 1
        
        improved = True
        no_improve_count = 0
        max_no_improve = 200
        
        while improved and time.time() < deadline - 15.0:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
            
            # Try changing residue for a random class to a random other residue
            if len(best_classes) > 0:
                idx = rng.randrange(len(best_classes))
                m, r = best_classes[idx]
                r2 = rng.randrange(m)
                if r2 != r:
                    new_classes = [c[:] for c in best_classes]
                    new_classes[idx] = [m, r2]
                    new_score = score_solution(new_classes)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes = new_classes
                        improved = True
                        no_improve_count = 0
        
        return best_classes, best_score

    def simulated_annealing(classes, base, deadline):
        """Simulated annealing on residue assignments."""
        if not classes:
            return classes, base
        
        rng = random.Random(777)
        current_classes = [c[:] for c in classes]
        current_score = score_solution(current_classes)
        best_classes = [c[:] for c in current_classes]
        best_score = current_score
        
        start_temp = 50.0
        end_temp = 0.5
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline - 15.0:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (step / steps)
            
            # Pick a random class and change its residue
            idx = rng.randrange(len(current_classes))
            m, r = current_classes[idx]
            r2 = rng.randrange(m)
            
            old_classes = [c[:] for c in current_classes]
            current_classes[idx] = [m, r2]
            new_score = score_solution(current_classes)
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
            else:
                current_classes = old_classes
        
        return best_classes, best_score

    def pairwise_improve(classes, base, deadline):
        """Try pairwise residue changes: change two residues at once."""
        if len(classes) < 2:
            return classes, base
        
        rng = random.Random(99999)
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        no_improve = 0
        max_no_improve = 300
        
        while improved and time.time() < deadline - 15.0:
            improved = False
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            # Pick two random classes
            i, j = rng.sample(range(len(best_classes)), 2)
            mi, ri = best_classes[i]
            mj, rj = best_classes[j]
            
            ri2 = rng.randrange(mi)
            rj2 = rng.randrange(mj)
            
            new_classes = [c[:] for c in best_classes]
            new_classes[i] = [mi, ri2]
            new_classes[j] = [mj, rj2]
            
            new_score = score_solution(new_classes)
            if new_score < best_score:
                best_score = new_score
                best_classes = new_classes
                improved = True
                no_improve = 0
        
        return best_classes, best_score

    def cover_with_constraint_propagation(base, deadline):
        """
        Backtracking search with forward checking using bitsets.
        Variables: divisors (sorted by some heuristic)
        Domains: residues 0..m-1
        Objective: minimize uncovered residues
        """
        # Order divisors: larger moduli first (cover fewer residues per class but are more specific)
        # Actually, smaller moduli cover more residues, so they might be more important
        # Let's try both orders
        div_order = sorted(divisors, key=lambda m: -m)  # descending
        
        # Precompute for each divisor m, for each residue r, the set of positions covered
        # This is expensive, so we'll use numpy slicing on the fly
        # But for constraint propagation, we need to track uncovered positions efficiently
        
        # Use a numpy array for uncovered
        uncovered = np.ones(base, dtype=np.uint8)
        selected = {}  # m -> r
        
        def backtrack(idx, current_uncovered, current_score):
            nonlocal uncovered, selected
            if time.time() > deadline - 10.0:
                return
            
            if idx >= n_divs:
                return current_uncovered.copy(), current_score
            
            m = div_order[idx]
            best_result = None
            best_score_so_far = current_score
            
            for r in range(m):
                # Compute new uncovered
                new_uncovered = current_uncovered.copy()
                new_uncovered[r::m] = 0
                new_score = int(np.sum(new_uncovered))
                
                if new_score < best_score_so_far:
                    best_score_so_far = new_score
                    best_result = (new_uncovered, new_score)
                
                # Forward checking: if new_score is already >= best known global, skip
                # (but we don't have a global best here easily, so just proceed)
                
                if time.time() > deadline - 10.0:
                    break
            
            if best_result is not None:
                return backtrack(idx + 1, best_result[0], best_result[1])
            else:
                return current_uncovered.copy(), current_score
        
        # This is too slow for full backtracking. Use a greedy approach with constraint propagation
        # instead: greedily select the (m, r) that maximizes coverage, with lookahead.
        
        # Greedy with lookahead: at each step, try each (m, r) and estimate the best
        # possible remaining coverage
        selected_classes = []
        current_uncovered = np.ones(base, dtype=np.uint8)
        
        remaining_divs = list(div_order)
        remaining_divs_set = set(remaining_divs)
        
        while remaining_divs:
            if time.time() > deadline - 10.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            best_uncovered = None
            
            for m in remaining_divs:
                for r in range(m):
                    # Tentatively select (m, r)
                    new_uncovered = current_uncovered.copy()
                    new_uncovered[r::m] = 0
                    gain = int(np.sum(current_uncovered[r::m] == 1))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
                        best_uncovered = new_uncovered
                
                if time.time() > deadline - 10.0:
                    break
            
            if best_m is None or best_gain <= 0:
                break
            
            selected_classes.append([best_m, best_r])
            current_uncovered = best_uncovered
            remaining_divs.remove(best_m)
        
        return selected_classes, int(np.sum(current_uncovered))

    def multi_start_greedy(base, deadline, n_starts=10):
        """Try multiple greedy starts with different orderings."""
        best_classes = []
        best_score = base
        rng = random.Random(55555)
        
        for s in range(n_starts):
            if time.time() > deadline - 10.0:
                break
            
            # Different ordering strategies
            if s % 3 == 0:
                order = list(divisors)  # ascending
            elif s % 3 == 1:
                order = list(reversed(divisors))  # descending
            else:
                order = list(divisors)
                rng.shuffle(order)
            
            classes, covered = greedy_pass(order, base, deadline)
            score = int(np.sum(covered == 0))
            
            if score < best_score:
                best_score = score
                best_classes = [c[:] for c in classes]
        
        return best_classes, best_score

    best_score = base
    best_classes = []
    
    # Strategy 1: Greedy with ascending divisors
    if time.time() < deadline - 30.0:
        classes, covered = greedy_pass(divisors, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Greedy with descending divisors
    if time.time() < deadline - 30.0:
        div_desc = list(reversed(divisors))
        classes, covered = greedy_pass(div_desc, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 3: Multi-start greedy
    if time.time() < deadline - 40.0:
        classes, score = multi_start_greedy(base, deadline, n_starts=8)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 4: Constraint propagation greedy
    if time.time() < deadline - 30.0:
        classes, score = cover_with_constraint_propagation(base, deadline)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Local search refinement
    if best_classes and time.time() < deadline - 20.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    # Hill climbing
    if best_classes and time.time() < deadline - 20.0:
        best_classes, best_score = hill_climb(best_classes, base, deadline)
    
    # Pairwise improvement
    if best_classes and time.time() < deadline - 15.0:
        best_classes, best_score = pairwise_improve(best_classes, base, deadline)
    
    # Simulated annealing
    if best_classes and time.time() < deadline - 15.0:
        best_classes, best_score = simulated_annealing(best_classes, base, deadline)
    
    # Final local search
    if best_classes and time.time() < deadline - 5.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    return best_classes if best_classes else []
