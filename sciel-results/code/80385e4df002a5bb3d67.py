import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    
    best_solution = []
    best_size = 0
    
    # We will try multiple random orders of moduli
    # and a greedy assignment
    
    # Optimization: Precompute gcds? Not strictly necessary for 127 items.
    
    def is_disjoint(r1, m1, r2, m2):
        d = gcd(m1, m2)
        return (r1 - r2) % d != 0
        
    def try_greedy(order):
        selected = [] # list of [r, m]
        for m in order:
            # Try to find a residue r in [0, m-1] such that it is disjoint from all in selected
            # To make it deterministic or fast, we can just iterate r
            # But we can also randomize r order
            
            # Optimization: If selected is empty, any r works.
            # Otherwise, check constraints.
            
            # We want to maximize success. 
            # A good heuristic for r is to pick one that is "far" from existing residues?
            # Or just the first valid one.
            
            # To allow different outcomes, we can shuffle the candidate residues
            candidates = list(range(m))
            random.shuffle(candidates)
            
            for r in candidates:
                valid = True
                for [r_s, m_s] in selected:
                    if not is_disjoint(r, m, r_s, m_s):
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    break
        return selected

    # Initial deterministic attempt with descending moduli (larger moduli have more residues, maybe easier to fit?)
    # Or ascending?
    # Let's try a few fixed orders and random orders.
    
    orders = [
        list(reversed(moduli)),
        list(moduli),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True), # More bits -> more divisors?
        sorted(moduli, key=lambda x: x.bit_count())
    ]
    
    # Add some random orders
    for _ in range(100):
        o = list(moduli)
        random.shuffle(o)
        orders.append(o)
        
    # If time permits, add more
    while time.time() < deadline - 5:
        # Pick an order
        if len(orders) > 0:
            order = orders.pop(0)
        else:
            order = list(moduli)
            random.shuffle(order)
            
        sol = try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            
        # If we found a very good solution, maybe stop?
        # Max possible is 127. Unlikely.
        # But if we get > 20, it's good.
        
    return best_solution if best_solution else [[1, 2]]
