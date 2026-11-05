import math

def construct():
    # Initial set: all even moduli with first valid residue
    selected = []
    for m in range(20, 201, 2):
        found = False
        for r in range(m):
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                selected.append([m, r])
                found = True
                break
    
    # Calculate initial score
    current_score = sum(1.0 / m for m, r in selected)
    
    # Try to improve by swapping out conflicting evens for odd moduli
    # We iterate through odd moduli. 
    # For each odd m, find evens in selected that are coprime to m.
    # If we remove them, can we add m?
    # We also need to ensure m is disjoint from the REMAINING evens.
    
    # To be efficient, we can sort odds by 1/m descending (small m first)
    # But removing large evens (small 1/m) to add small odds (large 1/m) is good.
    
    odds = list(range(21, 200, 2))
    # Sort by potential gain? Just iterate.
    
    improved = True
    while improved:
        improved = False
        for m in odds:
            # Check if m is already in selected (shouldn't be, we only added evens initially, but maybe added in previous iter?)
            if any(m == sm for sm, sr in selected):
                continue
                
            # Find evens in selected that conflict with m due to gcd=1
            conflicts = []
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if g == 1:
                    conflicts.append((m2, r2))
            
            # If no conflicts, we can try to add m directly
            # If conflicts, we must remove them.
            
            # Tentative new set
            new_selected = [item for item in selected if item not in conflicts]
            
            # Check if m can be added to new_selected
            valid_r = None
            for r in range(m):
                ok = True
                for m2, r2 in new_selected:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    valid_r = r
                    break
            
            if valid_r is not None:
                # Calculate score change
                loss = sum(1.0 / m2 for m2, r2 in conflicts)
                gain = 1.0 / m
                if gain > loss + 1e-9:
                    # Perform swap
                    selected = new_selected
                    selected.append([m, valid_r])
                    improved = True
                    # Restart loop as set changed
                    break
        # If no improvement in this pass, stop
        # Note: We might need multiple passes as removing one even might allow another odd
        # But the "break" inside the for loop restarts the while loop.
    
    return selected
