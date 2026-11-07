import time
import random

def construct():
    deadline = time.time() + 55.0
    best_size = 0
    best_elements = []
    
    def is_valid_divisibility_fast(sums, max_sum):
        """
        Check if no two distinct sums in the set divide each other.
        """
        for a in sums:
            if a == 0:
                continue
            k = 2
            # Optimization: if a > max_sum // 2, no multiple exists
            if a > max_sum // 2:
                continue
            while a * k <= max_sum:
                if (a * k) in sums:
                    return False
                k += 1
        return True

    def try_greedy_random():
        nonlocal best_size, best_elements
        
        current_elements = []
        current_sums = set()
        current_diffs = set()
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current_max_sum = 0
        
        for x in candidates:
            if time.time() > deadline:
                return
                
            if x in current_sums:
                continue
            if x in current_diffs:
                continue
                
            new_sums = set(current_sums)
            new_max_sum = current_max_sum
            for s in current_sums:
                ns = s + x
                new_sums.add(ns)
                if ns > new_max_sum:
                    new_max_sum = ns
            new_sums.add(x)
            if x > new_max_sum:
                new_max_sum = x
            
            if not is_valid_divisibility_fast(new_sums, new_max_sum):
                continue
                
            current_elements.append(x)
            current_sums = new_sums
            current_max_sum = new_max_sum
            
            # Update diffs
            # Only recompute if size is small enough?
            # If size > 10, N > 1000, diff calc is slow.
            # But we need it to prune future candidates.
            s_list = list(current_sums)
            new_diffs = set()
            # Use nested loops
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i+1, len(s_list)):
                    b = s_list[j]
                    d = a - b
                    if d > 0:
                        new_diffs.add(d)
            current_diffs = new_diffs
            
            if len(current_elements) > best_size:
                best_size = len(current_elements)
                best_elements = list(current_elements)

    # Run restarts
    while time.time() < deadline:
        try_greedy_random()
        
    return best_elements

if __name__ == "__main__":
    print(construct())
