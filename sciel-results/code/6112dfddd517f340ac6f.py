import time
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _sieve(1000)
    L_MAX = 10000 # Reduced L_MAX for speed, as L is likely < 10000
    
    # Precompute Fraction costs
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    def compute_L_from_counts(cover_count):
        L = 0
        # Use local variable for speed
        cc = cover_count
        for i in range(1, L_MAX + 1):
            if cc[i] > 0:
                L = i
            else:
                break
        return L

    def update_cover_count(cover_count, p, r, delta):
        start_k = r
        if start_k == 0:
            start_k = p
        # Local variable access is faster
        cc = cover_count
        for k in range(start_k, L_MAX + 1, p):
            cc[k] += delta

    def greedy_build(deadline):
        cover_count = [0] * (L_MAX + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= L_MAX:
            if time.time() > deadline:
                break
            if cover_count[n] > 0:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            
            # Heuristic: Pick largest p that fits
            # This saves budget
            for p in reversed(primes):
                if p in used_primes:
                    continue
                cost = prime_costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                best_p = p
                best_r = r
                break
            
            if best_p is None:
                break
                
            used_primes.add(best_p)
            budget -= prime_costs[best_p]
            classes.append([best_p, best_r])
            
            update_cover_count(cover_count, best_p, best_r, 1)
            
            n += 1
            
        return classes, budget, cover_count

    # 1. Initial Greedy
    classes, remaining_budget, cover_count = greedy_build(start_time + 5.0)
    
    current_L = compute_L_from_counts(cover_count)
    best_L = current_L
    best_classes = [list(c) for c in classes]
    used_primes = set(p for p, r in classes)
    
    # 2. Optimization
    # We will use a simple hill climbing approach
    # To avoid getting stuck, we can occasionally do a "random restart" of residues?
    # No, let's just do hill climbing.
    
    # Precompute list of primes for random selection
    primes_list = primes
    
    while time.time() < deadline - 2:
        # Try a batch of moves
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            
            # Random move type
            # Weight towards 'change' and 'swap'
            r = random.random()
            if r < 0.5:
                move_type = 'change'
            elif r < 0.8:
                move_type = 'swap'
            elif r < 0.9:
                move_type = 'add'
            else:
                move_type = 'remove'
            
            # Backup
            old_classes = [list(c) for c in classes]
            old_cover = cover_count[:]
            old_budget = remaining_budget
            old_used = set(used_primes)
            
            valid = True
            
            if move_type == 'change':
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                p, r_old = classes[idx]
                r_new = random.randint(0, p - 1)
                if r_new == r_old:
                    continue
                
                update_cover_count(cover_count, p, r_old, -1)
                update_cover_count(cover_count, p, r_new, 1)
                classes[idx] = [p, r_new]
                
            elif move_type == 'remove':
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                p, r = classes[idx]
                
                update_cover_count(cover_count, p, r, -1)
                remaining_budget += prime_costs[p]
                classes.pop(idx)
                used_primes.discard(p)
                
            elif move_type == 'add':
                # Find available primes
                # Optimization: Maintain a list of available primes?
                # For now, filter.
                available = [p for p in primes_list if p not in used_primes]
                if not available:
                    continue
                p = random.choice(available)
                cost = prime_costs[p]
                if remaining_budget < cost:
                    continue
                
                # Find first hole
                first_hole = -1
                for i in range(1, L_MAX + 1):
                    if cover_count[i] == 0:
                        first_hole = i
                        break
                
                if first_hole != -1:
                    r = first_hole % p
                else:
                    r = random.randint(0, p - 1)
                
                update_cover_count(cover_count, p, r, 1)
                remaining_budget -= cost
                classes.append([p, r])
                used_primes.add(p)
                
            elif move_type == 'swap':
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                p_old, r_old = classes[idx]
                
                # Find available primes (excluding p_old effectively, or allowing same p with diff r? 
                # If same p, it's just a change. Let's allow different p.)
                available = [p for p in primes_list if p not in used_primes or p == p_old]
                if not available:
                    continue
                
                # Prefer different prime
                if len(available) > 1:
                    candidates = [p for p in available if p != p_old]
                    if candidates:
                        p_new = random.choice(candidates)
                    else:
                        p_new = available[0]
                else:
                    p_new = available[0]
                
                cost_new = prime_costs[p_new]
                cost_old = prime_costs[p_old]
                
                if remaining_budget - cost_old + cost_new > 1:
                    continue
                
                # Find first hole
                first_hole = -1
                for i in range(1, L_MAX + 1):
                    if cover_count[i] == 0:
                        first_hole = i
                        break
                
                if first_hole != -1:
                    r_new = first_hole % p_new
                else:
                    r_new = random.randint(0, p_new - 1)
                
                update_cover_count(cover_count, p_old, r_old, -1)
                update_cover_count(cover_count, p_new, r_new, 1)
                
                remaining_budget -= cost_new
                remaining_budget += cost_old
                
                classes[idx] = [p_new, r_new]
                used_primes.discard(p_old)
                used_primes.add(p_new)
            
            # Evaluate
            new_L = compute_L_from_counts(cover_count)
            
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_classes = [list(c) for c in classes]
            else:
                # Revert
                classes = old_classes
                cover_count = old_cover
                remaining_budget = old_budget
                used_primes = old_used

    # Final check
    total_cost = sum(prime_costs[p] for p, r in best_classes)
    if total_cost > 1:
        # Should not happen
        return classes
    return best_classes

# EVOLVE-BLOCK-END
