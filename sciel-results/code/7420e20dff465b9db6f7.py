from fractions import Fraction
import random

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    def compute_total(s):
        t = Fraction(0, 1)
        for i in s:
            t += recips[i]
        return t
    
    # Greedy initial solution
    best_set = set()
    best_deficit = Fraction(1, 1)
    
    current_set = set()
    current_total = Fraction(0, 1)
    for i in range(n):
        if current_total + recips[i] < 1:
            current_set.add(i)
            current_total += recips[i]
    deficit = 1 - current_total
    if deficit < best_deficit:
        best_deficit = deficit
        best_set = set(current_set)
    
    random.seed(42)
    
    for restart in range(100):
        if restart == 0:
            current_set = set(best_set)
        else:
            current_set = set(best_set)
            for _ in range(random.randint(1, 3)):
                if current_set:
                    in_list = list(current_set)
                    out_list = [i for i in range(n) if i not in current_set]
                    if in_list and out_list:
                        r = random.choice(in_list)
                        a = random.choice(out_list)
                        current_set.remove(r)
                        current_set.add(a)
        
        current_total = compute_total(current_set)
        if current_total >= 1:
            continue
        current_deficit = 1 - current_total
        
        improved = True
        while improved:
            improved = False
            
            # Try adding
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < current_deficit:
                            current_set.add(i)
                            current_total = new_total
                            current_deficit = new_deficit
                            improved = True
                            break
            if improved:
                continue
            
            # Try best swap
            best_swap_i = -1
            best_swap_j = -1
            best_swap_deficit = current_deficit
            for i in list(current_set):
                for j in range(n):
                    if j not in current_set:
                        new_total = current_total - recips[i] + recips[j]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_swap_deficit:
                                best_swap_deficit = new_deficit
                                best_swap_i = i
                                best_swap_j = j
            if best_swap_i >= 0:
                current_set.remove(best_swap_i)
                current_set.add(best_swap_j)
                current_total = current_total - recips[best_swap_i] + recips[best_swap_j]
                current_deficit = best_swap_deficit
                improved = True
        
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_set = set(current_set)
    
    return sorted(best_set)
