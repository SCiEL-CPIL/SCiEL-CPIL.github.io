"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute all products for a given sequence efficiently
    def compute_all_products(values):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                products.add(prod)
        return products
    
    def compute_tails(values, x):
        """Compute all consecutive-block products ending at x for the sequence values+[x]."""
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        return tails
    
    def is_valid(values, products, x):
        """Check if x can be appended to values without product collision."""
        tails = compute_tails(values, x)
        for t in tails:
            if t in products:
                return False, tails
        return True, tails
    
    def greedy_from(values, products, start):
        """Continue greedily from current state. Returns (values, products)."""
        for x in range(start, 129):
            if time.time() > deadline:
                break
            valid, tails = is_valid(values, products, x)
            if valid:
                values.append(x)
                products.update(tails)
        return values, products
    
    # Greedy baseline
    values, products = [], set()
    values, products = greedy_from(values, products, 2)
    best = list(values)
    best_len = len(best)
    
    # Simulated annealing approach: try random swaps and insertions
    # Start from the greedy solution and try to improve
    
    # First, let's try some random restarts with different starting points
    random.seed(42)
    
    # Try starting with different initial values
    for start_val in range(2, 20):
        if time.time() > deadline:
            break
        
        test_values = [start_val]
        test_products = compute_all_products(test_values)
        test_values, test_products = greedy_from(test_values, test_products, start_val + 1)
        
        if len(test_values) > best_len:
            best = list(test_values)
            best_len = len(best)
    
    # Now try simulated annealing on the best solution found so far
    # We'll try to swap elements, insert new ones, or remove and re-add
    
    current = list(best)
    current_products = compute_all_products(current)
    current_len = len(current)
    
    # Simulated annealing parameters
    T = 10.0
    cooling = 0.95
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = list(current)
        neighbor_products = set(current_products)
        
        move_type = random.choice(['swap', 'insert', 'remove', 'replace'])
        
        if move_type == 'swap' and len(current) >= 2:
            # Swap two adjacent elements
            i = random.randint(0, len(current) - 2)
            j = i + 1
            # Check if swapping maintains strictly increasing order
            if i > 0 and current[i-1] >= current[j]:
                continue
            if j < len(current) - 1 and current[i] >= current[j+1]:
                continue
            neighbor[i], neighbor[j] = current[j], current[i]
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                continue
            
        elif move_type == 'insert':
            # Try to insert a new element
            pos = random.randint(0, len(current))
            # Find a value to insert
            if pos == 0:
                lo, hi = 2, current[0] - 1 if current else 128
            elif pos == len(current):
                lo, hi = current[-1] + 1, 128
            else:
                lo, hi = current[pos-1] + 1, current[pos] - 1
            
            if lo > hi:
                continue
            
            new_val = random.randint(lo, hi)
            neighbor = current[:pos] + [new_val] + current[pos:]
            
            # Check strictly increasing
            for k in range(len(neighbor) - 1):
                if neighbor[k] >= neighbor[k+1]:
                    neighbor = list(current)
                    break
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                neighbor = list(current)
                neighbor_products = set(current_products)
                
        elif move_type == 'remove' and len(current) > 1:
            # Remove a random element
            pos = random.randint(0, len(current) - 1)
            neighbor = current[:pos] + current[pos+1:]
            neighbor_products = compute_all_products(neighbor)
            
        elif move_type == 'replace' and len(current) >= 1:
            # Replace a random element with another valid value
            pos = random.randint(0, len(current) - 1)
            
            if pos == 0:
                lo, hi = 2, (current[1] - 1 if len(current) > 1 else 128)
            elif pos == len(current) - 1:
                lo, hi = (current[pos-1] + 1 if pos > 0 else 2), 128
            else:
                lo, hi = current[pos-1] + 1, current[pos+1] - 1
            
            if lo > hi:
                continue
            
            new_val = random.randint(lo, hi)
            if new_val == current[pos]:
                continue
            
            neighbor = list(current)
            neighbor[pos] = new_val
            
            # Check strictly increasing
            valid_inc = True
            for k in range(len(neighbor) - 1):
                if neighbor[k] >= neighbor[k+1]:
                    valid_inc = False
                    break
            if not valid_inc:
                continue
            
            # Recompute products
            neighbor_products = compute_all_products(neighbor)
            # Check for duplicates
            if len(neighbor_products) != sum(len(neighbor) - i for i in range(len(neighbor))):
                neighbor = list(current)
                neighbor_products = set(current_products)
        
        # Accept or reject based on simulated annealing
        delta = len(neighbor) - current_len
        
        if delta > 0:
            # Always accept improvements
            current = neighbor
            current_products = neighbor_products
            current_len = len(current)
            
            if current_len > best_len:
                best = list(current)
                best_len = current_len
        
        else:
            # Accept worse solutions with probability exp(delta/T)
            if random.random() < (2.718281828 ** (delta / T)):
                current = neighbor
                current_products = neighbor_products
                current_len = len(current)
        
        T *= cooling
    
    # Final greedy polish: try to extend the best solution found
    if time.time() < deadline - 5:
        polish_values = list(best)
        polish_products = compute_all_products(polish_values)
        polish_values, polish_products = greedy_from(polish_values, polish_products, 
                                                      polish_values[-1] + 1 if polish_values else 2)
        if len(polish_values) > best_len:
            best = polish_values
            best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
