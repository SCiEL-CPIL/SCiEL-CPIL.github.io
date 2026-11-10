import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # For each size s from 1 to 10, compute all residues achievable
    # and store ALL subsets for each residue (to allow better assignment)
    from itertools import combinations
    
    # residues_by_size[s] = dict mapping residue -> list of subsets
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            if time.time() > deadline - 5:
                break
            res = sum(inv[x] for x in combo) % 101
            if res not in residues_by_size[s]:
                residues_by_size[s][res] = []
            residues_by_size[s][res].append(list(combo))
    
    # Find minimum k such that union of residues for sizes 1..k covers all 101
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            covered.update(residues_by_size[s].keys())
        if len(covered) == 101:
            # Construct assignment: for each residue, pick smallest size subset
            result = []
            for r in range(101):
                found = None
                for s in range(1, k + 1):
                    if r in residues_by_size[s]:
                        found = residues_by_size[s][r][0]
                        break
                if found is None:
                    # Should not happen, but fallback
                    found = [1]  # placeholder, won't be used
                result.append(found)
            return result
    
    # Fallback: use original approach
    representations = {}
    for mask in range(1, 1 << 10):
        if time.time() > deadline - 2:
            break
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        representations.setdefault(residue, terms)
    return [representations.get(r, [1]) for r in range(101)]
