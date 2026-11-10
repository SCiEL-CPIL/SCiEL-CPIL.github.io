import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Helper to check if inserting x at index i is valid
    # seq is sorted. i is the index where x would be inserted.
    def is_valid_insertion(seq, prods, x, i):
        L = seq[:i]
        R = seq[i:]
        
        # Calculate suffix products of L
        suf = [1] * (len(L) + 1)
        prod = 1
        for j in range(len(L) - 1, -1, -1):
            prod *= L[j]
            suf[j] = prod
        
        # Calculate prefix products of R
        pre = [1] * (len(R) + 1)
        prod = 1
        for k in range(len(R)):
            prod *= R[k]
            pre[k + 1] = prod
        
        new_prods = []
        for j in range(i + 1):
            p1 = suf[j]
            for k in range(len(R) + 1):
                p2 = pre[k]
                p = p1 * x * p2
                new_prods.append(p)
        
        # Check distinctness of new_prods
        if len(set(new_prods)) != len(new_prods):
            return False, None
        
        new_set = set(new_prods)
        
        # Check collision with existing prods
        if not new_set.isdisjoint(prods):
            return False, None
        
        return True, new_set

    # 1. Greedy Baseline
    best_seq, best_products = [], set()
    for x in range(2, 129):
        i = len(best_seq) # Since we iterate in increasing order, x is always larger than all in best_seq
        # So i is always len(best_seq)
        valid, new_prods_set = is_valid_insertion(best_seq, best_products, x, i)
        if valid:
            best_seq.append(x)
            best_products.update(new_prods_set)
            
    best_len = len(best_seq)
    best_seq_ls = best_seq[:]
    
    # 2. Local Search
    current_seq = best_seq[:]
    current_prods = best_products.copy()
    unused = set(range(2, 129))
    unused.difference_update(current_seq)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        unused_list = list(unused)
        random.shuffle(unused_list)
        
        for x in unused_list:
            i = bisect.bisect_left(current_seq, x)
            valid, new_prods_set = is_valid_insertion(current_seq, current_prods, x, i)
            if valid:
                current_seq.insert(i, x)
                current_prods.update(new_prods_set)
                unused.remove(x)
                improved = True
                no_improve_count = 0
                
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_ls = current_seq[:]
                break
        
        if not improved:
            no_improve_count += 1
            if len(current_seq) > 0:
                # Kick
                idx_to_remove = random.randint(0, len(current_seq) - 1)
                x_removed = current_seq.pop(idx_to_remove)
                unused.add(x_removed)
                
                # Rebuild prods
                current_prods = set()
                for i in range(len(current_seq)):
                    prod = 1
                    for j in range(i, len(current_seq)):
                        prod *= current_seq[j]
                        current_prods.add(prod)
            
            if no_improve_count > 100:
                # Restart
                current_seq = best_seq_ls[:]
                current_prods = set()
                for i in range(len(current_seq)):
                    prod = 1
                    for j in range(i, len(current_seq)):
                        prod *= current_seq[j]
                        current_prods.add(prod)
                unused = set(range(2, 129))
                unused.difference_update(current_seq)
                no_improve_count = 0

    return best_seq_ls
