"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55
    
    # Start with S = {65, ..., 128}
    current = set(range(65, 129))
    
    # Greedily try to add elements from 1..64
    # For each candidate x, check if adding it creates a violation
    # A violation: 3 distinct a,b,c in set with a+b, a+c, b+c all in set
    
    a_elements = []  # elements added from 1..64, sorted
    
    for x in range(1, 65):
        if time.time() > deadline:
            break
        
        # Tentatively add x
        # Check all triples involving x and existing elements
        conflict = False
        
        # Get all current elements sorted
        # Triples with x: (x, y, z) where y, z are in current, y != z, y,z != x
        # Since x <= 64 and current has {65..128} plus some {1..64}
        
        # For efficiency, categorize:
        # S_elements = {65..128}
        # A_elements = a_elements (subset of {1..64})
        
        # Triples:
        # (x, y, z) all in A: y,z in a_elements
        # (x, y in A, z in S)
        # (x in A, y,z in S): y+z >= 65+66=131 > 128, so y+z not in set. SAFE.
        # (x,y,z all in S): already safe.
        
        # Case A: x, y, z all in A (y, z from a_elements, y < z)
        for i in range(len(a_elements)):
            if conflict:
                break
            for j in range(i+1, len(a_elements)):
                y, z = a_elements[i], a_elements[j]
                s1, s2, s3 = x+y, x+z, y+z
                # Check if all three are in current ∪ {x}
                if s1 in current or s1 == x:
                    if s2 in current or s2 == x:
                        if s3 in current or s3 == x:
                            conflict = True
                            break
        
        # Case B: x, y in A, z in S
        if not conflict:
            for y in a_elements:
                if conflict:
                    break
                s1 = x + y
                # s1 must be in current ∪ {x}
                if s1 in current or s1 == x:
                    # Need z in S such that x+z in current and y+z in current
                    # x+z in current iff x+z <= 128 (since x+z >= 66 > 64)
                    # y+z in current iff y+z <= 128
                    # z in {65..128}
                    # x+z <= 128 => z <= 128-x
                    # y+z <= 128 => z <= 128-y
                    # So z in {65..min(128-x, 128-y)}
                    max_z = min(128 - x, 128 - y)
                    if max_z >= 65:
                        conflict = True
                        break
        
        if not conflict:
            current.add(x)
            a_elements.append(x)
    
    return sorted(current)
# EVOLVE-BLOCK-END
