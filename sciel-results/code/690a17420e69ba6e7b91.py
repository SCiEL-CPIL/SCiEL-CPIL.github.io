import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    denoms = list(range(1, 11))
    
    # For each size s, compute all (residue, frozenset) pairs
    # residues_by_size[s] = list of (residue, tuple_of_denominators)
    residues_by_size = [[] for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(denoms, s):
            r = sum(inv[x] for x in combo) % 101
            residues_by_size[s].append((r, combo))
        if time.time() > deadline - 5:
            break
    
    # Find minimum size s where all 101 residues are achievable
    min_s = None
    for s in range(1, 11):
        if time.time() > deadline - 5:
            break
        achievable = set(r for r, _ in residues_by_size[s])
        if len(achievable) == 101:
            min_s = s
            break
    
    if min_s is None:
        # Fallback: use all sizes, pick shortest available
        # Build a combined list sorted by length
        all_pairs = []
        for s in range(1, 11):
            for r, combo in residues_by_size[s]:
                all_pairs.append((r, combo))
        # For each residue, keep the shortest
        best_for_residue = {}
        for r, combo in all_pairs:
            if r not in best_for_residue or len(combo) < len(best_for_residue[r]):
                best_for_residue[r] = combo
        result = [best_for_residue[r] for r in range(101)]
        return result
    
    # Greedy set-cover: pick subsets to cover all 101 residues
    # Each "set" is a subset of denominators; it covers the residue it produces
    # We need to assign one subset to each residue
    # Since each subset produces exactly one residue, this is just: for each residue, pick one subset of size min_s that produces it
    
    # Build mapping: residue -> list of combos of size min_s
    res_to_combos = {}
    for r, combo in residues_by_size[min_s]:
        if r not in res_to_combos:
            res_to_combos[r] = []
        res_to_combos[r].append(combo)
    
    # For each residue, just pick the first available combo
    result = []
    for r in range(101):
        if r in res_to_combos:
            result.append(list(res_to_combos[r][0]))
        else:
            # Should not happen since we verified all residues are covered
            # Fallback: find any combo
            for s in range(min_s, 11):
                for rr, combo in residues_by_size[s]:
                    if rr == r:
                        result.append(list(combo))
                        break
                else:
                    continue
                break
            else:
                result.append([1])  # last resort
    
    return result
