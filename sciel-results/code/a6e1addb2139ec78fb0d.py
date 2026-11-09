import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    
    # Build conflict graph using bitsets for fast intersection checks
    # conflict[v] is an integer bitset where bit (u-1) is set if v and u conflict
    # i.e., (v*u) % (v+u) == 0 and v != u
    
    conflict = [0] * (N + 1)
    
    for a in range(1, N + 1):
        a_sq = a * a
        limit = int(math.isqrt(a_sq))
        for k in range(1, limit + 1):
            if a_sq % k == 0:
                d1 = k
                d2 = a_sq // k
                for d in (d1, d2):
                    if a < d <= a + N:
                        b = d - a
                        if 1 <= b <= N and b != a:
                            conflict[a] |= (1 << (b - 1))
                            conflict[b] |= (1 << (a - 1))
    
    # Helper functions
    def can_add(v, current_bits):
        return (conflict[v] & current_bits) == 0
    
    def greedy_independent_set(order):
        chosen_bits = 0
        chosen_list = []
        for v in order:
            if can_add(v, chosen_bits):
                chosen_bits |= (1 << (v - 1))
                chosen_list.append(v)
        return chosen_bits, chosen_list
    
    def local_search(bits, size, max_iters=30):
        """Improve an independent set by trying removals followed by greedy refill."""
        current_list = [v for v in range(1, N + 1) if (bits >> (v - 1)) & 1]
        current_bits = bits
        current_size = size
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            # Try removing each vertex and see if greedy refill improves
            best_gain = 0
            best_new_bits = current_bits
            best_new_list = current_list
            best_new_size = current_size
            
            for i, v in enumerate(current_list):
                if time.time() > deadline - 5:
                    break
                
                # Remove v
                temp_bits = current_bits & ~(1 << (v - 1))
                temp_list = current_list[:i] + current_list[i + 1:]
                
                # Greedy refill: try to add as many as possible
                new_bits = temp_bits
                new_list = temp_list[:]
                
                for u in range(1, N + 1):
                    if time.time() > deadline - 5:
                        break
                    if (new_bits >> (u - 1)) & 1:
                        continue
                    if can_add(u, new_bits):
                        new_bits |= (1 << (u - 1))
                        new_list.append(u)
                
                if len(new_list) > current_size:
                    gain = len(new_list) - current_size
                    if gain > best_gain:
                        best_gain = gain
                        best_new_bits = new_bits
                        best_new_list = new_list
                        best_new_size = len(new_list)
            
            if best_gain > 0:
                current_bits = best_new_bits
                current_list = best_new_list
                current_size = best_new_size
                improved = True
        
        return current_bits, current_list, current_size
    
    def two_swap_local_search(bits, size, max_iters=20):
        """Try removing two vertices and refilling to see if net gain is positive."""
        current_list = [v for v in range(1, N + 1) if (bits >> (v - 1)) & 1]
        current_bits = bits
        current_size = size
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            best_gain = 0
            best_new_bits = current_bits
            best_new_list = current_list
            best_new_size = current_size
            
            n = len(current_list)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    
                    # Remove vertices at positions i and j
                    v1 = current_list[i]
                    v2 = current_list[j]
                    temp_bits = current_bits & ~(1 << (v1 - 1)) & ~(1 << (v2 - 1))
                    temp_list = [x for k, x in enumerate(current_list) if k != i and k != j]
                    
                    # Greedy refill
                    new_bits = temp_bits
                    new_list = temp_list[:]
                    for u in range(1, N + 1):
                        if time.time() > deadline - 5:
                            break
                        if (new_bits >> (u - 1)) & 1:
                            continue
                        if can_add(u, new_bits):
                            new_bits |= (1 << (u - 1))
                            new_list.append(u)
                    
                    if len(new_list) > current_size:
                        gain = len(new_list) - current_size
                        if gain > best_gain:
                            best_gain = gain
                            best_new_bits = new_bits
                            best_new_list = new_list
                            best_new_size = len(new_list)
            
            if best_gain > 0:
                current_bits = best_new_bits
                current_list = best_new_list
                current_size = best_new_size
                improved = True
        
        return current_bits, current_list, current_size
    
    all_nodes = list(range(1, N + 1))
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for v in range(1, N + 1):
        degrees[v] = conflict[v].bit_count()
    
    best_bits = 0
    best_size = 0
    best_list = []
    
    # Strategy 1: Greedy by degree (ascending) + local search
    order_by_degree = sorted(all_nodes, key=lambda x: degrees[x])
    bits, lst = greedy_independent_set(order_by_degree)
    if len(lst) > best_size:
        best_size = len(lst)
        best_bits = bits
        best_list = lst
    new_bits, new_list, new_size = local_search(bits, len(lst))
    if new_size > best_size:
        best_size = new_size
        best_bits = new_bits
        best_list = new_list
    new_bits, new_list, new_size = two_swap_local_search(new_bits, new_size)
    if new_size > best_size:
        best_size = new_size
        best_bits = new_bits
        best_list = new_list
    
    # Strategy 2: Start from odd numbers (known good baseline)
    odd_bits = 0
    for v in range(1, N + 1, 2):
        odd_bits |= (1 << (v - 1))
    odd_list = list(range(1, N + 1, 2))
    if len(odd_list) > best_size:
        best_size = len(odd_list)
        best_bits = odd_bits
        best_list = odd_list
    new_bits, new_list, new_size = local_search(odd_bits, len(odd_list))
    if new_size > best_size:
        best_size = new_size
        best_bits = new_bits
        best_list = new_list
    new_bits, new_list, new_size = two_swap_local_search(new_bits, new_size)
    if new_size > best_size:
        best_size = new_size
        best_bits = new_bits
        best_list = new_list
    
    # Strategy 3: Multiple random restarts with greedy + local search
    random.seed(42)
    
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        random_order = all_nodes[:]
        random.shuffle(random_order)
        
        bits, lst = greedy_independent_set(random_order)
        
        # Local search
        new_bits, new_list, new_size = local_search(bits, len(lst), max_iters=20)
        if new_size > best_size:
            best_size = new_size
            best_bits = new_bits
            best_list = new_list
        
        # Two-swap local search
        new_bits, new_list, new_size = two_swap_local_search(new_bits, new_size, max_iters=10)
        if new_size > best_size:
            best_size = new_size
            best_bits = new_bits
            best_list = new_list
    
    # Final polish: try to add any remaining isolated nodes
    final_bits = best_bits
    final_list = list(best_list)
    for x in range(1, N + 1):
        if time.time() > deadline - 2:
            break
        if (final_bits >> (x - 1)) & 1:
            continue
        if can_add(x, final_bits):
            final_bits |= (1 << (x - 1))
            final_list.append(x)
    
    if len(final_list) > best_size:
        best_size = len(final_list)
        best_bits = final_bits
        best_list = final_list
    
    return best_list
