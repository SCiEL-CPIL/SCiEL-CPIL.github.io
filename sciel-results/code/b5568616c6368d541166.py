import time

def construct():
    deadline = time.time() + 80  # safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Enumerate all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                residue = (residue + inv[x - 1]) % 101
        all_subsets.append((size, residue, mask))
    
    # Group by size
    by_size = {}
    for size, residue, mask in all_subsets:
        by_size.setdefault(size, []).append((residue, mask))
    
    # Find minimum k such that all 101 residues are covered by subsets of size <= k
    # Try k = 1, 2, 3, ...
    for k in range(1, 11):
        covered = set()
        candidates = []  # list of (residue, mask) for subsets of size <= k
        for s in range(1, k + 1):
            for residue, mask in by_size.get(s, []):
                covered.add(residue)
                candidates.append((residue, mask))
        
        if len(covered) == 101:
            # All residues covered with max size k
            # Assign each residue to a subset of size <= k
            # Prefer smallest size for each residue
            assignment = {}
            for s in range(1, k + 1):
                for residue, mask in by_size.get(s, []):
                    if residue not in assignment:
                        assignment[residue] = mask
            
            # Build the result
            result = []
            for r in range(101):
                mask = assignment[r]
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                result.append(terms)
            
            return result
    
    # Fallback: should not reach here, but just in case
    # Use any valid assignment
    assignment = {}
    for size, residue, mask in all_subsets:
        if residue not in assignment:
            assignment[residue] = mask
    
    result = []
    for r in range(101):
        mask = assignment[r]
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        result.append(terms)
    
    return result
