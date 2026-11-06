import numpy as np
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    MASK = (1 << 201) - 1
    
    def can_add(current_bits, x):
        """Check if adding x would make 200 reachable, and return new bits if safe."""
        new_bits = current_bits | (current_bits << x)
        new_bits &= MASK
        if (new_bits >> 200) & 1:
            return None
        return new_bits
    
    def compute_bits(element_set):
        bits = 1
        for x in element_set:
            bits |= (bits << x)
            bits &= MASK
        return bits
    
    def greedy(order, start_set=None):
        if start_set is not None:
            selected = list(start_set)
            bits = compute_bits(selected)
        else:
            selected = []
            bits = 1
        for x in order:
            if x in selected:
                continue
            new_bits = can_add(bits, x)
            if new_bits is not None:
                selected.append(x)
                bits = new_bits
        return selected
    
    best = []
    
    # Proven safe base: {66, 67, ..., 100} = 35 elements
    # Any subset of size >=3 sums to >200, any pair sums to <=199, single <=100
    base = list(range(66, 101))
    small = list(range(1, 66))
    
    # Try adding small elements in various orders to the base
    result = greedy(small[::-1], start_set=base)
    if len(result) > len(best):
        best = result
    
    result = greedy(small, start_set=base)
    if len(result) > len(best):
        best = result
    
    # Also try full-range greedy
    result = greedy(list(range(100, 0, -1)))
    if len(result) > len(best):
        best = result
    
    result = greedy(list(range(1, 101)))
    if len(result) > len(best):
        best = result
    
    # Local search: try to add excluded elements
    def local_search_improve(current_set):
        current_set = set(current_set)
        bits = compute_bits(current_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in excluded[:]:
                new_bits = can_add(bits, x)
                if new_bits is not None:
                    current_set.add(x)
                    bits = new_bits
                    excluded.remove(x)
                    improved = True
                    break
        return current_set
    
    best_set = local_search_improve(best)
    if len(best_set) > len(best):
        best = list(best_set)
    
    # (1,k)-exchange local search: remove one element, try to add multiple others
    def exchange_search(current_set):
        current_set = set(current_set)
        bits = compute_bits(current_set)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for rem in list(current_set):
                if time.time() >= deadline - 5:
                    break
                test_set = current_set - {rem}
                test_bits = compute_bits(test_set)
                
                excluded = [x for x in range(1, 101) if x not in test_set]
                for x in excluded:
                    new_bits = can_add(test_bits, x)
                    if new_bits is not None:
                        test_set.add(x)
                        test_bits = new_bits
                
                if len(test_set) > len(current_set):
                    current_set = test_set
                    bits = test_bits
                    improved = True
                    break
        return current_set
    
    if time.time() < deadline - 15:
        result = exchange_search(best)
        if len(result) > len(best):
            best = list(result)
    
    # Random restarts with the base set
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() >= deadline - 5:
                break
            order = small[:]
            random.shuffle(order)
            result = greedy(order, start_set=base)
            if len(result) > len(best):
                best = result
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 2:
        current_set = set(best)
        bits = compute_bits(current_set)
        for x in range(1, 101):
            if x not in current_set:
                new_bits = can_add(bits, x)
                if new_bits is not None:
                    current_set.add(x)
                    bits = new_bits
        if len(current_set) > len(best):
            best = list(current_set)
    
    return best
