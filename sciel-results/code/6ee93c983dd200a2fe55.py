"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to stay under 90s
    
    best_max_den = float('inf')
    best_solution = None
    
    # Greedy baseline first
    r = target
    greedy_vals = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    if greedy_vals:
        best_max_den = max(greedy_vals)
        best_solution = list(greedy_vals)
    
    # Try to improve by exploring small sets of initial denominators
    # Strategy: pick k denominators from a small range, then greedily complete
    def try_complete(remaining, used_set, current_max, max_terms):
        """Complete the expansion greedily, returning solution or None."""
        nonlocal best_max_den, best_solution
        if remaining == 0:
            return list(used_set)
        if len(used_set) >= max_terms:
            return None
        
        # Greedy step: choose smallest denominator > current_max that helps
        # d must satisfy: 1/d <= remaining, so d >= 1/remaining
        # Also d > current_max for distinctness (we'll enforce in caller)
        min_d = max(current_max + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Try the greedy choice first
        d = min_d
        if d > 10000000:
            return None
        new_remaining = remaining - Fraction(1, d)
        if new_remaining < 0:
            return None
        used_set.add(d)
        result = try_complete(new_remaining, used_set, d, max_terms)
        used_set.discard(d)
        return result
    
    # Search over small initial denominators
    # Try 1, 2, or 3 initial denominators from a small range
    for num_initial in [1, 2, 3]:
        if time.time() > deadline:
            break
        
        if num_initial == 1:
            for d1 in range(8, 60):
                if time.time() > deadline:
                    break
                r1 = target - Fraction(1, d1)
                if r1 <= 0:
                    continue
                # Complete greedily
                vals = [d1]
                used = {d1}
                rem = r1
                cur_max = d1
                success = True
                for _ in range(255):
                    if rem == 0:
                        break
                    if time.time() > deadline:
                        success = False
                        break
                    min_d = max(cur_max + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
                    if min_d > 10000000:
                        success = False
                        break
                    d = min_d
                    vals.append(d)
                    used.add(d)
                    rem -= Fraction(1, d)
                    cur_max = d
                if success and rem == 0 and len(vals) <= 256:
                    m = max(vals)
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = vals
        
        elif num_initial == 2:
            for d1 in range(8, 40):
                if time.time() > deadline:
                    break
                for d2 in range(d1 + 1, 60):
                    if time.time() > deadline:
                        break
                    r2 = target - Fraction(1, d1) - Fraction(1, d2)
                    if r2 <= 0:
                        continue
                    vals = [d1, d2]
                    used = {d1, d2}
                    rem = r2
                    cur_max = d2
                    success = True
                    for _ in range(254):
                        if rem == 0:
                            break
                        if time.time() > deadline:
                            success = False
                            break
                        min_d = max(cur_max + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
                        if min_d > 10000000:
                            success = False
                            break
                        d = min_d
                        vals.append(d)
                        used.add(d)
                        rem -= Fraction(1, d)
                        cur_max = d
                    if success and rem == 0 and len(vals) <= 256:
                        m = max(vals)
                        if m < best_max_den:
                            best_max_den = m
                            best_solution = vals
        
        elif num_initial == 3:
            for d1 in range(8, 30):
                if time.time() > deadline:
                    break
                for d2 in range(d1 + 1, 45):
                    if time.time() > deadline:
                        break
                    for d3 in range(d2 + 1, 60):
                        if time.time() > deadline:
                            break
                        r3 = target - Fraction(1, d1) - Fraction(1, d2) - Fraction(1, d3)
                        if r3 <= 0:
                            continue
                        vals = [d1, d2, d3]
                        used = {d1, d2, d3}
                        rem = r3
                        cur_max = d3
                        success = True
                        for _ in range(253):
                            if rem == 0:
                                break
                            if time.time() > deadline:
                                success = False
                                break
                            min_d = max(cur_max + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
                            if min_d > 10000000:
                                success = False
                                break
                            d = min_d
                            vals.append(d)
                            used.add(d)
                            rem -= Fraction(1, d)
                            cur_max = d
                        if success and rem == 0 and len(vals) <= 256:
                            m = max(vals)
                            if m < best_max_den:
                                best_max_den = m
                                best_solution = vals
    
    if best_solution is None:
        return greedy_vals
    return best_solution
# EVOLVE-BLOCK-END
