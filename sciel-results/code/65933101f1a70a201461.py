import time
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            if time.time() > deadline - 5:
                break
            res = sum(inv[x] for x in combo) % 101
            if res not in residues_by_size[s]:
                residues_by_size[s][res] = list(combo)
    
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            covered.update(residues_by_size[s].keys())
        if len(covered) == 101:
            result = []
            for r in range(101):
                found = None
                for s in range(1, k + 1):
                    if r in residues_by_size[s]:
                        found = residues_by_size[s][r]
                        break
                if found is None:
                    found = [1]
                result.append(found)
            return result
    
    representations = {}
    for mask in range(1, 1 << 10):
        if time.time() > deadline - 2:
            break
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        representations.setdefault(residue, terms)
    return [representations.get(r, [1]) for r in range(101)]
