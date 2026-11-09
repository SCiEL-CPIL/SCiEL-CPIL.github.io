import time
import numpy as np
import random
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N
PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Precompute indices for each prime to speed up
    prime_indices = {}
    for p in PRIMES:
        if p <= N + 1:
            prime_indices[p] = [np.arange(r, N + 1, p) for r in range(p)]
    
    best_points = None
    best_size = 0
    
    def evaluate_solution(chosen_residues):
        """Evaluate a solution: returns (size, robustness) where
        size = number of remaining elements
        robustness = min over primes p<=size of (count of least-populated residue class)
        """
        R = np.zeros(N + 1, dtype=bool)
        for p, r in chosen_residues.items():
            indices = prime_indices[p][r]
            R[indices] = True
        
        remaining = int(np.sum(~R))
        if remaining == 0:
            return 0, 0
        
        # Compute robustness: for each prime p <= remaining, find min count in any residue class
        # that is NOT the chosen one (i.e., how many elements survive in each other class)
        robustness = float('inf')
        for p in PRIMES:
            if p > remaining:
                break
            min_surviving = N + 1
            for r in range(p):
                if r == chosen_residues.get(p, -1):
                    continue
                indices = prime_indices[p][r]
                surviving = int(np.sum(~R[indices]))
                if surviving < min_surviving:
                    min_surviving = surviving
            robustness = min(robustness, min_surviving)
        
        return remaining, robustness
    
    def greedy_initial_solution():
        """Create an initial solution using greedy with random tie-breaking"""
        m = N + 1
        active_primes = [p for p in PRIMES if p <= m]
        chosen_residues = {}
        R = np.zeros(N + 1, dtype=bool)
        
        for p in active_primes:
            min_cost = N + 1
            best_r = 0
            best_cost_list = []
            
            for r in range(p):
                indices = prime_indices[p][r]
                cost = int(np.sum(~R[indices]))
                if cost < min_cost:
                    min_cost = cost
                    best_r = r
                    best_cost_list = [r]
                    if cost == 0:
                        break
                elif cost == min_cost:
                    best_cost_list.append(r)
            
            if len(best_cost_list) > 1:
                best_r = random.choice(best_cost_list)
            
            indices = prime_indices[p][best_r]
            R[indices] = True
            chosen_residues[p] = best_r
        
        return chosen_residues
    
    def simulated_annealing(initial_solution, max_iter=5000):
        """Apply simulated annealing to improve the solution"""
        current_solution = dict(initial_solution)
        current_size, current_rob = evaluate_solution(current_solution)
        
        best_solution = dict(current_solution)
        best_size = current_size
        best_rob = current_rob
        
        T = 10.0
        cooling = 0.995
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random prime to change
            active_primes = [p for p in PRIMES if p <= current_size]
            if not active_primes:
                break
            
            p = random.choice(active_primes)
            old_r = current_solution[p]
            
            # Pick a new random residue class different from old
            new_r = random.randint(0, p - 1)
            while new_r == old_r:
                new_r = random.randint(0, p - 1)
            
            # Try the change
            current_solution[p] = new_r
            new_size, new_rob = evaluate_solution(current_solution)
            
            # Objective: maximize size, then maximize robustness
            # Delta in primary objective (size)
            delta_size = new_size - current_size
            # Delta in secondary objective (robustness)
            delta_rob = new_rob - current_rob
            
            # Acceptance criterion
            if delta_size > 0:
                accept = True
            elif delta_size == 0 and delta_rob >= 0:
                accept = True
            else:
                # Compute energy difference
                # Primary: size decrease is bad
                # Secondary: robustness decrease is bad
                energy_delta = -delta_size * 1000.0 - delta_rob * 10.0
                if energy_delta < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-energy_delta / max(T, 0.001))
                    accept = random.random() < accept_prob
            
            if accept:
                current_size = new_size
                current_rob = new_rob
                if current_size > best_size or (current_size == best_size and current_rob > best_rob):
                    best_size = current_size
                    best_rob = current_rob
                    best_solution = dict(current_solution)
            else:
                # Revert
                current_solution[p] = old_r
            
            T *= cooling
        
        return best_solution, best_size, best_rob
    
    # Run multiple trials with simulated annealing
    trial_count = 0
    while time.time() < deadline:
        # Create initial solution
        initial_sol = greedy_initial_solution()
        
        # Apply simulated annealing
        improved_sol, sa_size, sa_rob = simulated_annealing(initial_sol, max_iter=3000)
        
        if sa_size > best_size:
            best_size = sa_size
            # Reconstruct points from solution
            R = np.zeros(N + 1, dtype=bool)
            for p, r in improved_sol.items():
                indices = prime_indices[p][r]
                R[indices] = True
            best_points = np.where(~R)[0].tolist()
        
        trial_count += 1
        
        # If we've done several trials, vary the initial solution more
        if trial_count % 3 == 0:
            # Try a different greedy strategy: process primes in random order
            m = N + 1
            active_primes = [p for p in PRIMES if p <= m]
            random.shuffle(active_primes)
            chosen_residues = {}
            R = np.zeros(N + 1, dtype=bool)
            
            for p in active_primes:
                min_cost = N + 1
                best_r = 0
                best_cost_list = []
                
                for r in range(p):
                    indices = prime_indices[p][r]
                    cost = int(np.sum(~R[indices]))
                    if cost < min_cost:
                        min_cost = cost
                        best_r = r
                        best_cost_list = [r]
                        if cost == 0:
                            break
                    elif cost == min_cost:
                        best_cost_list.append(r)
                
                if len(best_cost_list) > 1:
                    best_r = random.choice(best_cost_list)
                
                indices = prime_indices[p][best_r]
                R[indices] = True
                chosen_residues[p] = best_r
            
            initial_sol = chosen_residues
            improved_sol, sa_size, sa_rob = simulated_annealing(initial_sol, max_iter=3000)
            
            if sa_size > best_size:
                best_size = sa_size
                R = np.zeros(N + 1, dtype=bool)
                for p, r in improved_sol.items():
                    indices = prime_indices[p][r]
                    R[indices] = True
                best_points = np.where(~R)[0].tolist()
        
        if trial_count > 0 and trial_count % 10 == 0:
            # Periodic check: if no improvement in last few trials, increase exploration
            pass
    
    if best_points is not None:
        return best_points
    
    # Fallback: return empty or minimal valid set
    return [0]
