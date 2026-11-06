"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 512
    candidates = list(range(511, 0, -1))  # sorted by weight descending
    
    # Precompute prime factors for each candidate for faster coprime checks
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # Cache prime factor sets
    factor_cache = {}
    for a in candidates:
        factor_cache[a] = prime_factors(a)
    
    def are_coprime(a, b):
        return len(factor_cache[a] & factor_cache[b]) == 0
    
    # Greedy initial solution
    def greedy_solution(order=None):
        if order is None:
            order = candidates
        chosen = []
        for a in order:
            if all(are_coprime(a, b) for b in chosen):
                chosen.append(a)
        return chosen
    
    best_solution = greedy_solution()
    best_score = sum(1.0 / (n - a) for a in best_solution)
    
    # Improved local search: try single swaps and double swaps
    def local_search(solution, max_iterations=2000):
        solution = list(solution)
        current_score = sum(1.0 / (n - a) for a in solution)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 2:
                break
            
            # Try single element replacement
            for i, a in enumerate(solution):
                remaining = [x for j, x in enumerate(solution) if j != i]
                for b in candidates:
                    if b == a or b in remaining:
                        continue
                    if 1.0 / (n - b) <= 1.0 / (n - a):
                        continue
                    if all(are_coprime(b, x) for x in remaining):
                        new_score = current_score - 1.0 / (n - a) + 1.0 / (n - b)
                        if new_score > current_score:
                            solution = remaining + [b]
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            # If no single swap improved, try double swaps
            if not improved and iterations % 5 == 0:
                for i in range(len(solution)):
                    for j in range(i + 1, len(solution)):
                        a1, a2 = solution[i], solution[j]
                        remaining = [x for k, x in enumerate(solution) if k != i and k != j]
                        # Try to replace both with two better elements
                        for b1 in candidates:
                            if b1 == a1 or b1 == a2 or b1 in remaining:
                                continue
                            if not all(are_coprime(b1, x) for x in remaining):
                                continue
                            for b2 in candidates:
                                if b2 == a1 or b2 == a2 or b2 == b1 or b2 in remaining:
                                    continue
                                if not are_coprime(b1, b2):
                                    continue
                                if not all(are_coprime(b2, x) for x in remaining):
                                    continue
                                new_score = (current_score 
                                            - 1.0 / (n - a1) - 1.0 / (n - a2)
                                            + 1.0 / (n - b1) + 1.0 / (n - b2))
                                if new_score > current_score:
                                    solution = remaining + [b1, b2]
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
        
        return solution, current_score
    
    # Beam search: maintain multiple partial solutions
    def beam_search(beam_width=20):
        # Each beam entry: (score, chosen_list, next_index)
        # We process candidates in order of decreasing weight
        beams = [(0.0, [], 0)]
        
        for a in candidates:
            new_beams = []
            for score, chosen, _ in beams:
                # Option 1: skip a
                new_beams.append((score, list(chosen), len(candidates)))
                
                # Option 2: include a if coprime with all chosen
                if all(are_coprime(a, b) for b in chosen):
                    new_score = score + 1.0 / (n - a)
                    new_beams.append((new_score, chosen + [a], len(candidates)))
            
            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
            
            if time.time() > deadline - 5:
                break
        
        # Best solution from beam search
        best_beam = max(beams, key=lambda x: x[0])
        return best_beam[1], best_beam[0]
    
    # Run beam search
    if time.time() < deadline - 10:
        beam_sol, beam_score = beam_search(beam_width=30)
        if beam_score > best_score:
            best_score = beam_score
            best_solution = beam_sol
    
    # Local search on best solution
    if time.time() < deadline - 5:
        ls_sol, ls_score = local_search(best_solution, max_iterations=1500)
        if ls_score > best_score:
            best_score = ls_score
            best_solution = ls_sol
    
    # Try multiple random restarts
    random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 3:
            break
        
        # Create perturbed ordering
        shuffled = candidates[:]
        for i in range(len(shuffled) - 1):
            if random.random() < 0.2:
                j = min(i + random.randint(1, 5), len(shuffled) - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        chosen = greedy_solution(shuffled)
        score = sum(1.0 / (n - a) for a in chosen)
        
        if score > best_score:
            best_score = score
            best_solution = chosen
        
        # Local search from this starting point
        if time.time() < deadline - 3:
            improved_sol, improved_score = local_search(chosen, max_iterations=300)
            if improved_score > best_score:
                best_score = improved_score
                best_solution = improved_sol
    
    # Final polish
    if time.time() < deadline - 1:
        final_sol, final_score = local_search(best_solution, max_iterations=1000)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol
    
    return best_solution
# EVOLVE-BLOCK-END
