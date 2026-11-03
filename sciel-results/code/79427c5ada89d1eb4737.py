"""Mixed-radix digit-set construction for maximizing the sumset/difference-set exponent."""
import time
import math
import random


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        if n == 0:
            return 0.0

        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)

        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def build_mixed_radix(radices, digit_sets):
        """Build a set using mixed-radix representation.
        
        radices: list of radix values for each digit position
        digit_sets: list of digit sets for each position
        
        The set is {sum(d_i * multiplier_i) for d_i in digit_sets[i]}
        where multiplier_0 = 1, multiplier_i = product(radices[0..i-1])
        """
        if not radices or not digit_sets:
            return [0]
        
        n_digits = len(radices)
        multipliers = [1] * n_digits
        for i in range(1, n_digits):
            multipliers[i] = multipliers[i - 1] * radices[i - 1]
        
        result = {0}
        # For each digit position, expand the set
        for i in range(n_digits):
            new_result = set()
            for d in digit_sets[i]:
                offset = d * multipliers[i]
                for val in result:
                    new_result.add(val + offset)
            result = new_result
        
        return sorted(result)

    def find_good_digit_set(radix, max_val, time_limit):
        """Find a digit set D ⊆ [0, radix) that maximizes |D-D|/|D+D|."""
        best_set = None
        best_ratio = 0.0
        deadline_local = start_time + time_limit
        
        # Try all subsets for small radices
        if radix <= 12:
            for mask in range(1, 1 << radix):
                if time.time() > deadline_local:
                    break
                D = [i for i in range(radix) if mask & (1 << i)]
                if 0 not in D:
                    D = D + [0]
                    D = list(set(D))
                
                D_set = set(D)
                sumset = set()
                diffset = set()
                for a in D_set:
                    for b in D_set:
                        sumset.add(a + b)
                        diffset.add(a - b)
                
                ratio = len(diffset) / len(sumset)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_set = D
        else:
            # For larger radices, use greedy/randomized approach
            # Start with {0} and try adding elements
            current_D = [0]
            for _ in range(50):  # iterations
                if time.time() > deadline_local:
                    break
                # Try adding a random element
                new_elem = random.randint(1, radix - 1)
                new_D = list(set(current_D + [new_elem]))
                
                D_set = set(new_D)
                sumset = set()
                diffset = set()
                for a in D_set:
                    for b in D_set:
                        sumset.add(a + b)
                        diffset.add(a - b)
                
                ratio = len(diffset) / len(sumset)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_set = new_D
                    current_D = new_D
            
            # Also try some structured sets
            for size in range(2, min(radix, 10) + 1):
                if time.time() > deadline_local:
                    break
                # Try evenly spaced sets
                D = list(range(0, radix, max(1, radix // size)))
                D = list(set(D))
                if 0 not in D:
                    D.append(0)
                
                D_set = set(D)
                sumset = set()
                diffset = set()
                for a in D_set:
                    for b in D_set:
                        sumset.add(a + b)
                        diffset.add(a - b)
                
                ratio = len(diffset) / len(sumset)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_set = D
        
        return best_set if best_set else [0]

    # Start with a simple construction
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Try different mixed-radix constructions
    # Strategy: use 2-3 digit positions with carefully chosen radices and digit sets
    
    # Try small 2-digit constructions first
    for radix1 in range(2, 8):
        if time.time() > deadline:
            break
        
        digit_set1 = find_good_digit_set(radix1, radix1, 0.5)
        
        for radix2 in range(2, 8):
            if time.time() > deadline:
                break
            
            digit_set2 = find_good_digit_set(radix2, radix2, 0.5)
            
            U = build_mixed_radix([radix1, radix2], [digit_set1, digit_set2])
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Try 3-digit constructions
    for r1 in range(2, 6):
        if time.time() > deadline:
            break
        ds1 = find_good_digit_set(r1, r1, 0.3)
        
        for r2 in range(2, 6):
            if time.time() > deadline:
                break
            ds2 = find_good_digit_set(r2, r2, 0.3)
            
            for r3 in range(2, 6):
                if time.time() > deadline:
                    break
                ds3 = find_good_digit_set(r3, r3, 0.3)
                
                U = build_mixed_radix([r1, r2, r3], [ds1, ds2, ds3])
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # Try larger constructions with more digits
    for num_digits in range(4, 7):
        if time.time() > deadline:
            break
        
        radices = []
        digit_sets = []
        for i in range(num_digits):
            if time.time() > deadline:
                break
            r = random.choice([2, 3, 4, 5])
            ds = find_good_digit_set(r, r, 0.2)
            radices.append(r)
            digit_sets.append(ds)
        
        if len(radices) == num_digits:
            U = build_mixed_radix(radices, digit_sets)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Local search: try modifying the best set
    if time.time() < deadline - 5:
        current_set = best_set.copy()
        current_score = best_score
        no_improve_count = 0
        
        while time.time() < deadline - 2 and no_improve_count < 50:
            # Try adding or removing an element
            if len(current_set) < 50 and random.random() < 0.5:
                # Add
                max_val = max(current_set)
                candidate = random.randint(1, max_val + 10)
                if candidate not in current_set:
                    new_set = sorted(current_set + [candidate])
            else:
                # Remove (not 0)
                removable = [x for x in current_set if x != 0]
                if not removable:
                    break
                elem = random.choice(removable)
                new_set = [x for x in current_set if x != elem]
            
            if not new_set or 0 not in new_set:
                new_set = sorted(set(new_set + [0]))
            
            s = score_set(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
            else:
                no_improve_count += 1
    
    return best_set
