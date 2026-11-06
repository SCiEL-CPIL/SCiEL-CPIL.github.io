import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    values = []
    products = set()
    suffixes = [1]
    
    while True:
        if time.time() > deadline:
            break
        
        last_val = values[-1] if values else 1
        best_candidate = None
        best_new_products = None
        best_suffixes = None
        min_cost = float('inf')
        
        for x in range(last_val + 1, 129):
            # Calculate new products
            new_prods_list = [x * u for u in suffixes]
            new_prods_set = set(new_prods_list)
            
            if products.intersection(new_prods_set):
                continue
                
            cost = len(new_prods_set)
            
            if cost < min_cost:
                min_cost = cost
                best_candidate = x
                best_new_products = new_prods_set
                best_suffixes = new_prods_list
            elif cost == min_cost:
                if best_candidate is None or x < best_candidate:
                    best_candidate = x
                    best_new_products = new_prods_set
                    best_suffixes = new_prods_list
        
        if best_candidate is None:
            break
            
        values.append(best_candidate)
        products.update(best_new_products)
        suffixes = best_suffixes
        
    return values
