"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    float_vals = [1.0 / d for d in denominators]
    
    def get_deficit(selected):
        """Compute the deficit 1 - sum(1/d_i) for selected indices."""
        total = Fraction(0)
        for i in selected:
            total += fracs[i]
        return 1 - total
    
    def best_fit_greedy(order):
        """Best-fit greedy: at each step, pick the available fraction that maximizes sum while staying < 1."""
        selected = set()
        total_float = 0.0
        available = set(range(n))
        
        while available:
            best_i = None
            best_new_total = total_float
            for i in available:
                new_total = total_float + float_vals[i]
                if new_total < 0.9999999999 and new_total > best_new_total:
                    best_new_total = new_total
                    best_i = i
            if best_i is None:
                break
            selected.add(best_i)
            total_float = best_new_total
            available.discard(best_i)
        
        return selected, total_float
    
    # Generate multiple initial solutions
    best_selected = set()
    best_deficit = Fraction(1)
    best_total_float = 0.0
    
    # Best-fit greedy (order-independent since we pick best at each step)
    selected, total_float = best_fit_greedy(list(range(n)))
    if selected:
        deficit = get_deficit(selected)
        if deficit < best_deficit:
            best_deficit = deficit
            best_selected = set(selected)
            best_total_float = total_float
    
    # Also try simple ascending greedy for comparison
    selected2 = set()
    total2 = Fraction(0)
    for i in range(n):
        if total2 + fracs[i] < 1:
            selected2.add(i)
            total2 += fracs[i]
    if selected2:
        deficit2 = get_deficit(selected2)
        if deficit2 < best_deficit:
            best_deficit = deficit2
            best_selected = set(selected2)
            best_total_float = sum(float_vals[i] for i in selected2)
    
    # Local search: 1-swap moves (remove one, add one)
    selected = set(best_selected)
    total_float = best_total_float
    total_frac = sum(fracs[i] for i in selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all 1-swap moves
        for rem in list(selected):
            if time.time() > deadline:
                break
            rem_float = total_float - float_vals[rem]
            for add in range(n):
                if time.time() > deadline:
                    break
                if add not in selected:
                    new_float = rem_float + float_vals[add]
                    if new_float < 0.9999999999 and new_float > total_float:
                        new_selected = (selected - {rem}) | {add}
                        # Verify with exact arithmetic
                        new_total_frac = total_frac - fracs[rem] + fracs[add]
                        if new_total_frac < 1:
                            new_deficit = 1 - new_total_frac
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                best_selected = new_selected
                                selected = new_selected
                                total_float = new_float
                                total_frac = new_total_frac
                                improved = True
                                break
        
        # Try adding single elements
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            if i not in selected:
                new_float = total_float + float_vals[i]
                if new_float < 0.9999999999 and new_float > total_float:
                    new_selected = selected | {i}
                    new_total_frac = total_frac + fracs[i]
                    if new_total_frac < 1:
                        new_deficit = 1 - new_total_frac
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = new_selected
                            selected = new_selected
                            total_float = new_float
                            total_frac = new_total_frac
                            improved = True
                            break
    
    # Try 2-swap moves if time permits (remove two, add two)
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            selected_list = list(selected)
            for i in range(len(selected_list)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(selected_list)):
                    if time.time() > deadline:
                        break
                    rem1, rem2 = selected_list[i], selected_list[j]
                    rem_float = total_float - float_vals[rem1] - float_vals[rem2]
                    rem_frac = total_frac - fracs[rem1] - fracs[rem2]
                    for add1 in range(n):
                        if time.time() > deadline:
                            break
                        if add1 in selected:
                            continue
                        for add2 in range(n):
                            if time.time() > deadline:
                                break
                            if add2 in selected or add2 == add1:
                                continue
                            new_float = rem_float + float_vals[add1] + float_vals[add2]
                            if new_float < 0.9999999999 and new_float > total_float:
                                new_selected = (selected - {rem1, rem2}) | {add1, add2}
                                new_total_frac = rem_frac + fracs[add1] + fracs[add2]
                                if new_total_frac < 1:
                                    new_deficit = 1 - new_total_frac
                                    if new_deficit < best_deficit:
                                        best_deficit = new_deficit
                                        best_selected = new_selected
                                        selected = new_selected
                                        total_float = new_float
                                        total_frac = new_total_frac
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final verification
    final_total = sum(fracs[i] for i in best_selected)
    if final_total >= 1:
        # Fallback: use the first valid greedy solution
        best_selected = {0}  # Just index 0 (denominator 2) is always valid
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
