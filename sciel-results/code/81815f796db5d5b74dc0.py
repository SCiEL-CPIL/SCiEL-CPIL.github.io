"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Greedy construction prioritizing small numbers (high reciprocal weight)
    # A 3-term AP {a, b, c} with a < b < c satisfies 2b = a + c
    # We build the set by considering numbers in order of decreasing 1/x (i.e., increasing x)
    # but we want to maximize sum of 1/x, so we prefer small x.
    # However, greedy by smallest x first may block many small x's.
    # Let's try a different approach: local search / hill climbing on the set.
    
    # First, generate a good initial set using a greedy heuristic that favors small numbers
    # but is more careful about conflicts.
    
    # Strategy: Start with all numbers, then remove elements that are part of APs,
    # prioritizing removal of large numbers (low weight) first.
    
    # Actually, let's use a greedy approach that adds numbers in order of weight (small to large)
    # but checks if adding creates an AP. This is the same as the seed but let's verify.
    
    # The seed does: for x in 1..256, add x if no AP formed.
    # This is optimal for a greedy that processes in order 1, 2, ..., 256.
    # But maybe a different order gives a better set?
    # Since we want to maximize sum(1/x), we definitely want small numbers.
    # The greedy "add if possible" in order 1..256 is a natural baseline.
    
    # Let's try local search: start with the greedy set, then try to swap elements
    # or remove/add elements to improve the score.
    
    def has_ap3(set_list):
        s = set(set_list)
        for a in s:
            for b in s:
                if a < b:
                    c = 2*b - a
                    if c in s:
                        return True
        return False
    
    # Greedy construction (same as seed but with timing)
    selected = []
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Check if adding x creates a 3-term AP
        # x can be the first, middle, or last term
        # If x is first: need a, b in selected such that x + b = 2a -> b = 2a - x
        # If x is middle: need a, c in selected such that a + c = 2x -> c = 2x - a
        # If x is last: need a, b in selected such that a + x = 2b -> a = 2b - x
        # Actually, simpler: for any a, b in selected with a < b, check if 2b-a is x or a+b-x is in selected...
        # The seed's check: all(2*b-a != x for a in selected for b in selected if a<b)
        # This checks if x is the third term of an AP (a, b, x).
        # But it also needs to check if x is the first or second term!
        # Wait, if we process in increasing order, and we've only added numbers < x,
        # then x can only be the largest element in any AP involving it.
        # So checking 2b-a == x for a < b < x is sufficient.
        # The seed's loop `for a in selected for b in selected if a<b` checks all pairs.
        # Since selected only contains numbers < x, 2b-a could be > x, = x, or < x.
        # If 2b-a < x, it's already in selected, which would mean an AP existed before? No.
        # Actually, if a, b are in selected and 2b-a is also in selected, that AP was already formed.
        # But we ensure no AP exists in selected. So 2b-a cannot be in selected.
        # The only new AP formed by adding x is if x is part of it.
        # Since x is the largest, x = 2b - a for some a, b in selected with a < b < x.
        # So the check is correct.
        
        valid = True
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                if 2*b - a == x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    # Now local search to improve
    current_score = sum(Fraction(1, x) for x in selected)
    selected_set = set(selected)
    
    # Try swapping: remove a large x, add a smaller y that was previously rejected
    # Or remove two, add one, etc.
    
    # Let's try a simple hill climbing: try to add any number from 1..256 not in set,
    # if it doesn't create an AP. If it does, try removing one element to make it fit.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, 257):
            if time.time() > deadline - 5:
                break
            if x in selected_set:
                continue
            # Try adding x
            # Check if x creates AP
            creates_ap = False
            for i, a in enumerate(selected):
                for b in selected[i+1:]:
                    if 2*b - a == x or 2*x - a == b:  # x is last or middle
                        creates_ap = True
                        break
                if creates_ap:
                    break
            # Also check if x is first: 2*a - x in selected for some a in selected, a > x
            # But we process in order, so selected may have elements > x? No, selected is built from 1..256.
            # Wait, selected is not necessarily sorted? It is, because we add in order.
            # But after swaps, it might not be. Let's keep it sorted.
            
            if not creates_ap:
                selected.append(x)
                selected_set.add(x)
                current_score += Fraction(1, x)
                improved = True
            else:
                # Try to remove one element to allow adding x
                # This is expensive. Let's skip for now.
                pass
        
        # Try removing an element and adding a smaller one
        for x in range(256, 0, -1):
            if time.time() > deadline - 5:
                break
            if x not in selected_set:
                continue
            # Try removing x
            selected.remove(x)
            selected_set.remove(x)
            old_score = current_score
            current_score -= Fraction(1, x)
            
            # Try adding smaller numbers
            for y in range(1, x):
                if y in selected_set:
                    continue
                # Check if y can be added
                creates_ap = False
                for i, a in enumerate(selected):
                    for b in selected[i+1:]:
                        if 2*b - a == y or 2*y - a == b:
                            creates_ap = True
                            break
                    if creates_ap:
                        break
                if not creates_ap:
                    selected.append(y)
                    selected_set.add(y)
                    current_score += Fraction(1, y)
                    if current_score > old_score:
                        improved = True
                        break
                    else:
                        # Revert
                        selected.remove(y)
                        selected_set.remove(y)
                        current_score -= Fraction(1, y)
            
            # If we didn't improve by adding y, keep x removed? Or put it back?
            # If we put it back, we lost nothing. If we keep it removed, we might have space for more.
            # Let's put it back if we didn't improve.
            if not improved:
                selected.append(x)
                selected_set.add(x)
                current_score += Fraction(1, x)
            
            selected.sort()
    
    # Final check for APs
    if has_ap3(selected):
        # Remove elements to fix APs (shouldn't happen if logic is correct)
        # Simple fix: rebuild greedily
        selected = []
        selected_set = set()
        for x in range(1, 257):
            valid = True
            for i, a in enumerate(selected):
                for b in selected[i+1:]:
                    if 2*b - a == x:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
    
    return selected
# EVOLVE-BLOCK-END
