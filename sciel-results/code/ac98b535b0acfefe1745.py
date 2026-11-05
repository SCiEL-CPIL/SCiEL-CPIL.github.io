# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s
    
    def get_tail_products(values, x):
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        return tails
    
    def is_valid(values, products, x):
        tails = get_tail_products(values, x)
        return not products.intersection(tails)
    
    def greedy_fill(start_val, values, products):
        """Continue greedily from start_val."""
        for x in range(start_val, 129):
            if is_valid(values, products, x):
                values.append(x)
                products.update(get_tail_products(values, x))
        return values, products
    
    # Initial greedy
    values = []
    products = set()
    for x in range(2, 129):
        if is_valid(values, products, x):
            values.append(x)
            products.update(get_tail_products(values, x))
    
    best = list(values)
    best_len = len(best)
    
    # Local search: try removing each element and refilling from that position
    for i in range(len(best)):
        if time.time() > deadline:
            break
        
        # Remove element at position i
        removed = best[i]
        new_values = best[:i] + best[i+1:]
        
        # Recompute products for new_values
        new_products = set()
        for j in range(len(new_values)):
            for k in range(j, len(new_values)):
                prod = 1
                for m in range(j, k+1):
                    prod *= new_values[m]
                new_products.add(prod)
        
        # Continue greedily from removed value (try to insert something >= removed)
        start_val = removed
        
        new_values2, new_products2 = greedy_fill(start_val, new_values, new_products)
        
        if len(new_values2) > best_len:
            best = new_values2
            best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
