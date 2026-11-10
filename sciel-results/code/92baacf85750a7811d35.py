"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    denominators = list(range(2, 49))
    n = len(denominators)
    inv_denoms = [Fraction(1, d) for d in denominators]
    
    # Start with greedy solution
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    
    best_selected = selected[:]
    best_total = total
    best_deficit = Fraction(1) - best_total
    
    # Local search: try to improve by adding, removing, or swapping elements
    def try_add_remove_swap(best_sel, best_tot, best_def):
        nonlocal deadline
        improved = True
        while improved:
            improved = False
            sel = list(best_sel)
            tot = best_tot
            # Try adding elements
            for i in range(n):
                if i not in sel:
                    if tot + inv_denoms[i] < 1:
                        new_sel = sel + [i]
                        new_tot = tot + inv_denoms[i]
                        new_def = Fraction(1) - new_tot
                        if new_def < best_def:
                            best_sel = new_sel
                            best_tot = new_tot
                            best_def = new_def
                            improved = True
                            break
            if time.time() > deadline:
                break
            if improved:
                continue
            # Try removing elements
            for i in sel[:]:
                if len(sel) > 1:
                    new_sel = [x for x in sel if x != i]
                    new_tot = tot - inv_denoms[i]
                    new_def = Fraction(1) - new_tot
                    if new_def < best_def:
                        best_sel = new_sel
                        best_tot = new_tot
                        best_def = new_def
                        improved = True
                        break
            if time.time() > deadline:
                break
            if improved:
                continue
            # Try swapping elements
            for i in sel[:]:
                for j in range(n):
                    if j not in sel:
                        new_sel = [x for x in sel if x != i] + [j]
                        new_tot = tot - inv_denoms[i] + inv_denoms[j]
                        if new_tot < 1:
                            new_def = Fraction(1) - new_tot
                            if new_def < best_def:
                                best_sel = new_sel
                                best_tot = new_tot
                                best_def = new_def
                                improved = True
                                break
                if improved or time.time() > deadline:
                    break
        return best_sel, best_tot, best_def
    
    best_selected, best_total, best_deficit = try_add_remove_swap(best_selected, best_total, best_deficit)
    
    return best_selected
# EVOLVE-BLOCK-END
