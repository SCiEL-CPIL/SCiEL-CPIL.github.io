"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 85.0
    
    # Start with the proven valid interval [171, 256]
    selected = set(range(171, 257))
    sel_list = sorted(selected)
    
    def can_add(x, sel_list, selected):
        """Check if x can be added to selected set."""
        # Case 1: x is the smallest (a=x), check x doesn't divide b+c for any b,c in sel with b>x, c>=b
        for i in range(len(sel_list)):
            b = sel_list[i]
            if b <= x:
                continue
            for j in range(i, len(sel_list)):
                c = sel_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is in the middle or largest, some a < x in sel
        # For each a in sel with a < x:
        #   - x as b: for c in sel with c >= x: check a divides x+c
        #   - x as c: for b in sel with a < b <= x: check a divides b+x
        for a in sel_list:
            if a >= x:
                break
            # x as b (b=x), c in sel with c >= x
            for c in sel_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            # x as c (c=x), b in sel with a < b <= x
            for b in sel_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Try to add smaller numbers
    for candidate in range(170, 0, -1):
        if time.time() > deadline:
            break
        if can_add(candidate, sel_list, selected):
            selected.add(candidate)
            sel_list.append(candidate)
            sel_list.sort()
    
    return sorted(selected)
# EVOLVE-BLOCK-END
