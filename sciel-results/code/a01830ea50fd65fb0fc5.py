import time

def construct() -> dict:
    # The smallest known valid tournament for the 4-domination property (Schütte's problem k=4)
    # is the Paley tournament on p=67 vertices.
    # p=67 is a prime such that p ≡ 3 (mod 4).
    # The orientation is determined by quadratic residues modulo p.
    # i -> j if (j - i) mod p is a quadratic residue.
    
    p = 67
    
    # Precompute quadratic residues modulo p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
        
    # Construct the orientation bits
    # Bit for pair (i, j) with i < j is at index idx in the list.
    # idx = i * (p - 1 - i) // 2 + (j - i - 1) is not quite right for flat list.
    # Standard lexicographic order:
    # (0,1), (0,2), ..., (0,p-1), (1,2), (1,3), ..., (1,p-1), ...
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in residues:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': p, 'orientation': orientation}
