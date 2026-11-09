# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin (limit 60s)
    
    def get_subset_sums(elements):
        """Returns a set of all non-empty subset sums."""
        sums = {0}
        for x in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
            sums.discard(0)
        return sums

    def is_valid_fast(sums_set):
        """Check if no two sums divide each other using Numpy."""
        if not sums_set:
            return True
        
        max_sum = max(sums_set)
        size = max_sum + 1
        arr = np.zeros(size, dtype=bool)
        for s in sums_set:
            arr[s] = True
        
        if 1 in sums_set and len(sums_set) > 1:
            return False
            
        for a in sums_set:
            if a == 1:
                continue
            if np.any(arr[2*a::a]):
                return False
        return True

    def try_add(x, current_sums, current_elements):
        """
        Attempts to add x to the set.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in current_elements:
            return False, None
            
        # Check for collisions (duplicates)
        if x in current_sums:
            return False, None
        
        # x + s must not be in S for any s in S
        for s in current_sums:
            if (s + x) in current_sums:
                return False, None
        
        # Construct new_sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        if not is_valid_fast(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_construct(ordering):
        """
        Greedy construction using a specific ordering of candidates.
        """
        elements = []
        sums = set()
        
        for x in ordering:
            if x in elements:
                continue
            ok, new_sums = try_add(x, sums, elements)
            if ok:
                elements.append(x)
                sums = new_sums
                
        return elements, sums

    def local_search(initial_elements, initial_sums, deadline):
        """
        Local search to improve the set size.
        """
        current_elements = list(initial_elements)
        current_sums = set(initial_sums)
        best_elements = list(current_elements)
        best_sums = set(current_sums)
        
        all_candidates = list(range(1, 257))
        
        while time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add a random element
            random.shuffle(all_candidates)
            for x in all_candidates:
                if x in current_elements:
                    continue
                ok, new_sums = try_add(x, current_sums, current_elements)
                if ok:
                    current_elements.append(x)
                    current_sums = new_sums
                    if len(current_elements) > len(best_elements):
                        best_elements = list(current_elements)
                        best_sums = set(current_sums)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. If no add possible, try a swap
            if not current_elements:
                break
                
            idx = random.randint(0, len(current_elements) - 1)
            x_remove = current_elements[idx]
            
            new_elements = [e for e in current_elements if e != x_remove]
            new_sums = get_subset_sums(new_elements)
            
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                    
                available = [x for x in all_candidates if x not in new_elements]
                if not available:
                    break
                x_add = random.choice(available)
                
                ok, new_sums_after_add = try_add(x_add, new_sums, new_elements)
                if ok:
                    final_elements = new_elements + [x_add]
                    final_sums = new_sums_after_add
                    
                    # Try to extend this new set greedily
                    random.shuffle(all_candidates)
                    for x_next in all_candidates:
                        if x_next in final_elements:
                            continue
                        ok2, new_sums2 = try_add(x_next, final_sums, final_elements)
                        if ok2:
                            final_elements.append(x_next)
                            final_sums = new_sums2
                            if len(final_elements) > len(best_elements):
                                best_elements = list(final_elements)
                                best_sums = set(final_sums)
                                improved = True
                                current_elements = final_elements
                                current_sums = final_sums
                                break
                    if improved:
                        break
                    else:
                        current_elements = final_elements
                        current_sums = final_sums
                        break
                        
        return best_elements, best_sums

    # Main logic
    best_elements = []
    best_sums = set()
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Binary representation approach: prioritize numbers with high 2-adic valuations
        # and large values to ensure distinct subset sums and avoid divisibility
        if restart_count % 4 == 1:
            # Pure descending (large numbers are safer for divisibility)
            order = list(range(256, 0, -1))
        elif restart_count % 4 == 2:
            # Sort by 2-adic valuation descending, then by value descending
            # Numbers with higher 2-adic valuation are "more independent"
            def two_adic_val(x):
                v = 0
                while x % 2 == 0:
                    v += 1
                    x //= 2
                return v
            order = list(range(1, 257))
            order.sort(key=lambda x: (-two_adic_val(x), -x))
        elif restart_count % 4 == 3:
            # Random shuffle with bias toward large numbers
            large = list(range(128, 257))
            small = list(range(1, 128))
            random.shuffle(large)
            random.shuffle(small)
            order = large + small
        else:
            # Pure random
            order = list(range(1, 257))
            random.shuffle(order)
        
        elements, sums = greedy_construct(order)
        
        if len(elements) > len(best_elements):
            best_elements = elements
            best_sums = sums
            
        # Local search
        new_best_elements, new_best_sums = local_search(best_elements, best_sums, deadline)
        if len(new_best_elements) > len(best_elements):
            best_elements = new_best_elements
            best_sums = new_best_sums

    # Final verification and cleanup
    result = [int(x) for x in best_elements]
    
    final_sums = get_subset_sums(result)
    if not is_valid_fast(final_sums):
        # Fallback: simple greedy descending
        fallback_elements = []
        fallback_sums = set()
        for x in range(256, 0, -1):
            ok, new_sums = try_add(x, fallback_sums, fallback_elements)
            if ok:
                fallback_elements.append(x)
                fallback_sums = new_sums
        result = [int(x) for x in fallback_elements]
        
    return result

# EVOLVE-BLOCK-END
