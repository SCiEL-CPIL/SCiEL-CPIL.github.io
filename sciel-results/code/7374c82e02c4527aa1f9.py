import time
import random

def construct():
    deadline = time.time() + 140
    prescribed = frozenset({1, 2, 5, 11})
    N = 284
    
    def compute_sums(elements):
        s = set()
        n = len(elements)
        for i in range(n):
            ei = elements[i]
            for j in range(i, n):
                s.add(ei + elements[j])
        return s
    
    def is_valid(elements):
        if not prescribed.issubset(set(elements)):
            return False
        s = compute_sums(elements)
        return len(s) == len(elements) * (len(elements) + 1) // 2
    
    best_set = None
    best_size = 0
    
    def update_best(elements):
        nonlocal best_set, best_size
        if len(elements) > best_size:
            if is_valid(elements):
                best_size = len(elements)
                best_set = list(elements)
    
    def greedy_init():
        elements = sorted(prescribed)
        elem_set = set(elements)
        sums = compute_sums(elements)
        for x in range(12, N + 1):
            if x in elem_set:
                continue
            new_sums = {x + y for y in elements} | {2 * x}
            if len(new_sums) == len(elements) + 1 and not (new_sums & sums):
                elements.append(x)
                elements.sort()
                elem_set.add(x)
                sums |= new_sums
        return elements, elem_set, sums
    
    elements, elem_set, sums = greedy_init()
    update_best(elements)
    
    if time.time() > deadline:
        return best_set if best_set else sorted(prescribed)
    
    def local_add(elements, elem_set, sums):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x in elem_set:
                    continue
                new_sums = {x + y for y in elements} | {2 * x}
                if len(new_sums) == len(elements) + 1 and not (new_sums & sums):
                    elements = sorted(elements + [x])
                    elem_set.add(x)
                    sums |= new_sums
                    update_best(elements)
                    improved = True
                    break
        return elements, elem_set, sums
    
    elements, elem_set, sums = local_add(elements, elem_set, sums)
    
    if time.time() > deadline:
        return best_set if best_set else sorted(prescribed)
    
    def sa_search(elements, elem_set, sums, max_iters=10000):
        nonlocal best_set, best_size
        temp = 3.0
        cooling = 0.9997
        
        for it in range(max_iters):
            if time.time() > deadline or temp < 0.005:
                break
            n = len(elements)
            r = random.random()
            
            if r < 0.55:  # Add
                candidates = [x for x in range(1, N + 1) if x not in elem_set]
                if not candidates:
                    temp *= cooling
                    continue
                x = random.choice(candidates)
                new_sums = {x + y for y in elements} | {2 * x}
                if not (new_sums & sums):
                    elements = sorted(elements + [x])
                    elem_set = elem_set | {x}
                    sums = sums | new_sums
                    update_best(elements)
            elif r < 0.80 and n > 4:  # Remove
                removable = [x for x in elements if x not in prescribed]
                if not removable:
                    temp *= cooling
                    continue
                x = random.choice(removable)
                idx = elements.index(x)
                elements = elements[:idx] + elements[idx+1:]
                elem_set = elem_set - {x}
                sums = compute_sums(elements)
            else:  # Swap
                if n <= 4:
                    temp *= cooling
                    continue
                removable = [x for x in elements if x not in prescribed]
                if not removable:
                    temp *= cooling
                    continue
                x = random.choice(removable)
                candidates = [y for y in range(1, N + 1) if y not in elem_set]
                if not candidates:
                    temp *= cooling
                    continue
                y = random.choice(candidates)
                idx = elements.index(x)
                new_elements = elements[:idx] + elements[idx+1:] + [y]
                new_elements.sort()
                new_sums = compute_sums(new_elements)
                if len(new_sums) == len(new_elements) * (len(new_elements) + 1) // 2:
                    elements = new_elements
                    elem_set = elem_set - {x} | {y}
                    sums = new_sums
                    update_best(elements)
            
            temp *= cooling
        
        return elements, elem_set, sums
    
    restart = 0
    while time.time() < deadline:
        elements, elem_set, sums = sa_search(elements, elem_set, sums, max_iters=8000)
        restart += 1
        if restart % 4 == 0:
            elements, elem_set, sums = greedy_init()
            elements, elem_set, sums = local_add(elements, elem_set, sums)
    
    if best_set is None:
        best_set = sorted(prescribed)
    
    return best_set
