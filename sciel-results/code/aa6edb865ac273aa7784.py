import time

def construct():
    # Deadline guard (though this computation is instantaneous)
    start_time = time.time()
    
    # Use the smallest known prime p=67 for the Paley tournament
    # which is known to satisfy the condition that every 4 vertices
    # have a common dominator.
    p = 67
    
    # Compute quadratic residues modulo p
    # i -> j iff j - i is a quadratic residue mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                # i -> j is 0
                orientation.append(0)
            else:
                # j -> i is 1
                orientation.append(1)
                
    # Safety check (optional, but good practice if unsure, though 67 is known valid)
    # For n=67, number of pairs is 67*66/2 = 2211
    if len(orientation) != p * (p - 1) // 2:
        raise ValueError("Orientation length mismatch")
        
    return {'n': p, 'orientation': orientation}
