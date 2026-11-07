import sys
import time

def construct():
    candidates = list(range(2, 41))
    
    def search_max_k(k, order, time_limit=5):
        """
        Returns a list of k integers if found within time_limit, else None.
        """
        n = len(order)
        if k > n:
            return None
            
        start_time = time.time()
        nodes_visited = 0
        max_nodes = 200000 # Safety cap
        
        def dfs(start_idx, current_set, current_products):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > max_nodes or time.time() - start_time > time_limit:
                return None
            
            if len(current_set) == k:
                return current_set[:]
            
            # Pruning: Not enough candidates left
            if len(current_set) + (n - start_idx) < k:
                return None
                
            for i in range(start_idx, n):
                num = order[i]
                # Check collisions
                # We need to check if any p * num is in current_products
                # AND if any new product collides with another? No, p1*num != p2*num.
                # So just check against current_products.
                
                # Optimization: If num is large, products are large, less likely to collide?
                # Not necessarily.
                
                valid = True
                new_prods = []
                # Iterate through current products
                for p in current_products:
                    np = p * num
                    if np in current_products:
                        valid = False
                        break
                    new_prods.append(np)
                
                if not valid:
                    continue
                
                new_products = set(current_products)
                new_products.update(new_prods)
                
                res = dfs(i + 1, current_set + [num], new_products)
                if res is not None:
                    return res
                    
            return None
            
        return dfs(0, [], {1})

    orders = []
    orders.append(sorted(candidates))
    orders.append(sorted(candidates, reverse=True))
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [x for x in candidates if x not in primes]
    orders.append(primes + composites)
    orders.append(composites + primes)
    
    best_result = []
    
    # Try 18
    found = False
    for order in orders:
        res = search_max_k(18, order, time_limit=10)
        if res:
            best_result = res
            found = True
            break
            
    if not found:
        # Try 17
        for order in orders:
            res = search_max_k(17, order, time_limit=10)
            if res:
                best_result = res
                found = True
                break
                
    if not found:
        # Fallback to known 16
        best_result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return best_result
